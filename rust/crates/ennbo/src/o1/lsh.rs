use rand::rngs::StdRng;
use rayon::prelude::*;
use rand::{Rng, SeedableRng};
use std::collections::HashMap;

use crate::backend::MmapColumnStore;
use crate::error::ENNError;

pub(crate) const O1_MAX_BITS: usize = 40;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct O1Params {
    pub num_tables: usize,
    pub bucket_target: usize,
    pub bucket_cap: usize,
    pub num_probes: usize,
}

impl Default for O1Params {
    fn default() -> Self {
        Self {
            num_tables: 8,
            bucket_target: 8,
            bucket_cap: 32,
            num_probes: 4,
        }
    }
}

fn standard_normal(rng: &mut StdRng) -> f64 {
    let u1 = rng.gen::<f64>().clamp(1e-12, 1.0 - 1e-12);
    let u2 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

#[derive(Default)]
struct Bucket {
    arrivals: u64,
    ids: Vec<u32>,
}

struct Hyperplane {
    normal: Vec<f64>,
    anchor_row: usize,
    offset: f64,
}

pub(crate) struct LshTables {
    params: O1Params,
    dim: usize,
    planes: Vec<Vec<Hyperplane>>,
    bits: usize,
    hash_scale: Vec<f64>,
    inv_scale: Vec<f64>,
    keys: Vec<u64>,
    buckets: Vec<HashMap<u64, Bucket>>,
    rng: StdRng,
}

pub(crate) fn target_bits(n: usize, bucket_target: usize) -> usize {
    let mut bits = 0;
    while bits < O1_MAX_BITS && (bucket_target << bits) < n {
        bits += 1;
    }
    bits
}

pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = [0.0f64; 4];
    let body = a.len() - a.len() % 4;
    for (ca, cb) in a[..body].chunks_exact(4).zip(b[..body].chunks_exact(4)) {
        for (l, slot) in acc.iter_mut().enumerate() {
            *slot += ca[l] * cb[l];
        }
    }
    let tail: f64 = (body..a.len()).map(|j| a[j] * b[j]).sum();
    acc.iter().sum::<f64>() + tail
}

fn scaled(x: &[f64], inv_scale: &[f64]) -> Vec<f64> {
    x.iter().zip(inv_scale).map(|(v, w)| v * w).collect()
}

fn bucket_insert(bucket: &mut Bucket, id: u32, cap: usize, rng: &mut StdRng) {
    bucket.arrivals += 1;
    if bucket.ids.len() < cap {
        bucket.ids.push(id);
        return;
    }
    let slot = rng.gen_range(0..bucket.arrivals);
    if (slot as usize) < cap {
        bucket.ids[slot as usize] = id;
    }
}

impl LshTables {
    pub(crate) fn new(dim: usize, seed: u64, params: O1Params) -> Self {
        Self {
            params,
            dim,
            planes: (0..params.num_tables).map(|_| Vec::new()).collect(),
            bits: 0,
            hash_scale: vec![1.0; dim],
            inv_scale: vec![1.0; dim],
            keys: Vec::new(),
            buckets: (0..params.num_tables).map(|_| HashMap::new()).collect(),
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub(crate) fn bits(&self) -> usize {
        self.bits
    }

    pub(crate) fn num_rows(&self) -> usize {
        self.keys.len() / self.params.num_tables
    }

    pub(crate) fn hash_scale(&self) -> &[f64] {
        &self.hash_scale
    }

    pub(crate) fn memory_bytes(&self) -> usize {
        let planes = self.params.num_tables * self.bits * (self.dim + 2) * 8;
        let ids: usize = self
            .buckets
            .iter()
            .flat_map(|t| t.values())
            .map(|b| 24 + 4 * b.ids.capacity())
            .sum();
        planes + ids + 8 * self.keys.capacity()
    }

    fn key_of(&self, table: usize, z: &[f64]) -> u64 {
        let mut key = 0u64;
        for plane in &self.planes[table] {
            key = (key << 1) | u64::from(dot(&plane.normal, z) > plane.offset);
        }
        key
    }

    fn row_keys(&self, x: &[f64]) -> Vec<u64> {
        let z = scaled(x, &self.inv_scale);
        (0..self.params.num_tables).map(|t| self.key_of(t, &z)).collect()
    }

    fn all_row_keys(&self, rows: &MmapColumnStore, start: usize) -> Result<Vec<u64>, ENNError> {
        let per_row: Result<Vec<Vec<u64>>, ENNError> = (start..rows.nrows)
            .into_par_iter()
            .map(|i| rows.mmap_row_slice(i).map(|x| self.row_keys(x)))
            .collect();
        Ok(per_row?.into_iter().flatten().collect())
    }

    pub(crate) fn query_keys(&self, x: &[f64]) -> Vec<(u64, Vec<usize>)> {
        let z = scaled(x, &self.inv_scale);
        (0..self.params.num_tables)
            .map(|t| {
                let mut key = 0u64;
                let mut margins = Vec::with_capacity(self.bits);
                for (b, plane) in self.planes[t].iter().enumerate() {
                    let gap = dot(&plane.normal, &z) - plane.offset;
                    key = (key << 1) | u64::from(gap > 0.0);
                    margins.push((gap.abs(), self.bits - 1 - b));
                }
                margins.sort_by(|a, b| a.0.total_cmp(&b.0));
                let flips = margins.iter().take(self.params.num_probes).map(|m| m.1).collect();
                (key, flips)
            })
            .collect()
    }

    pub(crate) fn bucket_ids(&self, table: usize, key: u64) -> &[u32] {
        self.buckets[table]
            .get(&key)
            .map(|b| b.ids.as_slice())
            .unwrap_or(&[])
    }

    pub(crate) fn insert_rows(
        &mut self,
        rows: &MmapColumnStore,
        start: usize,
    ) -> Result<(), ENNError> {
        if u32::try_from(rows.nrows).is_err() {
            return Err(ENNError::InvalidParameter(
                "O1 store holds at most 2^32 rows".into(),
            ));
        }
        let new_keys = self.all_row_keys(rows, start)?;
        for (offset, &key) in new_keys.iter().enumerate() {
            let id = (start + offset / self.params.num_tables) as u32;
            let t = offset % self.params.num_tables;
            bucket_insert(self.buckets[t].entry(key).or_default(), id, self.params.bucket_cap, &mut self.rng);
        }
        self.keys.extend(new_keys);
        while self.bits < target_bits(rows.nrows, self.params.bucket_target) {
            self.add_bit(rows)?;
        }
        Ok(())
    }

    fn draw_plane(&mut self, rows: &MmapColumnStore) -> Result<Hyperplane, ENNError> {
        let normal: Vec<f64> = (0..self.dim)
            .map(|_| standard_normal(&mut self.rng))
            .collect();
        let anchor_row = self.rng.gen_range(0..rows.nrows.max(1));
        let offset = if rows.nrows == 0 {
            0.0
        } else {
            dot(&normal, &scaled(rows.mmap_row_slice(anchor_row)?, &self.inv_scale))
        };
        Ok(Hyperplane {
            normal,
            anchor_row,
            offset,
        })
    }

    fn add_bit(&mut self, rows: &MmapColumnStore) -> Result<(), ENNError> {
        for t in 0..self.params.num_tables {
            let plane = self.draw_plane(rows)?;
            self.planes[t].push(plane);
        }
        self.bits += 1;
        let fresh: Vec<&Hyperplane> = self.planes.iter().map(|p| &p[self.bits - 1]).collect();
        let n = self.num_rows();
        let sides: Result<Vec<Vec<bool>>, ENNError> = (0..n)
            .into_par_iter()
            .map(|i| {
                let z = scaled(rows.mmap_row_slice(i)?, &self.inv_scale);
                Ok(fresh.iter().map(|p| dot(&p.normal, &z) > p.offset).collect())
            })
            .collect();
        for (i, row_sides) in sides?.into_iter().enumerate() {
            for (t, side) in row_sides.into_iter().enumerate() {
                let slot = i * self.params.num_tables + t;
                self.keys[slot] = (self.keys[slot] << 1) | u64::from(side);
            }
        }
        self.rebuild_buckets();
        Ok(())
    }

    fn rebuild_buckets(&mut self) {
        for table in &mut self.buckets {
            table.clear();
        }
        for (slot, &key) in self.keys.iter().enumerate() {
            let (i, t) = (slot / self.params.num_tables, slot % self.params.num_tables);
            bucket_insert(
                self.buckets[t].entry(key).or_default(),
                i as u32,
                self.params.bucket_cap,
                &mut self.rng,
            );
        }
    }

    pub(crate) fn rehash(
        &mut self,
        rows: &MmapColumnStore,
        hash_scale: &[f64],
    ) -> Result<(), ENNError> {
        self.hash_scale = hash_scale.to_vec();
        self.inv_scale = hash_scale.iter().map(|s| 1.0 / s).collect();
        for t in 0..self.params.num_tables {
            for b in 0..self.bits {
                let anchor = scaled(rows.mmap_row_slice(self.planes[t][b].anchor_row)?, &self.inv_scale);
                self.planes[t][b].offset = dot(&self.planes[t][b].normal, &anchor);
            }
        }
        let n = self.num_rows();
        if n > 0 {
            self.keys = self.all_row_keys(rows, 0)?;
            self.keys.truncate(n * self.params.num_tables);
        }
        self.rebuild_buckets();
        Ok(())
    }
}
