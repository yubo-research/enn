use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::backend::MmapColumnStore;
use crate::error::ENNError;

pub(crate) const O1_MAX_BITS: usize = 40;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct O1Params {
    pub num_tables: usize,
    pub bucket_target: usize,
    pub bucket_cap: usize,
    pub num_probes: usize,
    pub build_budget_bytes: usize,
}

impl Default for O1Params {
    fn default() -> Self {
        Self {
            num_tables: 8,
            bucket_target: 8,
            bucket_cap: 32,
            num_probes: 4,
            build_budget_bytes: 32 << 20,
        }
    }
}

pub(crate) fn io_error(e: impl std::fmt::Display) -> ENNError {
    ENNError::InvalidParameter(e.to_string())
}

pub(crate) fn mix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

pub(crate) fn table_seed(seed: u64, table: usize) -> u64 {
    mix64(seed ^ mix64(table as u64 + 1))
}

fn plane_seed(seed: u64, table: usize, bit: usize) -> u64 {
    mix64(table_seed(seed, table) ^ mix64((bit as u64 + 1) << 20))
}

fn standard_normal(rng: &mut StdRng) -> f64 {
    let u1 = rng.gen::<f64>().clamp(1e-12, 1.0 - 1e-12);
    let u2 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
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

struct Hyperplane {
    normal: Vec<f64>,
    anchor_row: usize,
    offset: f64,
}

pub(crate) struct Planes {
    seed: u64,
    dim: usize,
    planes: Vec<Vec<Hyperplane>>,
    hash_scale: Vec<f64>,
    inv_scale: Vec<f64>,
}

impl Planes {
    pub(crate) fn new(dim: usize, seed: u64, num_tables: usize) -> Self {
        Self {
            seed,
            dim,
            planes: (0..num_tables).map(|_| Vec::new()).collect(),
            hash_scale: vec![1.0; dim],
            inv_scale: vec![1.0; dim],
        }
    }

    pub(crate) fn cleared(&self) -> Self {
        let mut fresh = Self::new(self.dim, self.seed, self.planes.len());
        fresh.hash_scale.clone_from(&self.hash_scale);
        fresh.inv_scale.clone_from(&self.inv_scale);
        fresh
    }

    pub(crate) fn bits(&self) -> usize {
        self.planes.first().map_or(0, Vec::len)
    }

    pub(crate) fn hash_scale(&self) -> &[f64] {
        &self.hash_scale
    }

    pub(crate) fn anchors(&self) -> Vec<Vec<usize>> {
        self.planes
            .iter()
            .map(|t| t.iter().map(|p| p.anchor_row).collect())
            .collect()
    }

    pub(crate) fn memory_bytes(&self) -> usize {
        self.planes.len() * self.bits() * (self.dim + 2) * 8 + 16 * self.dim
    }

    fn normal(&self, table: usize, bit: usize) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(plane_seed(self.seed, table, bit));
        (0..self.dim).map(|_| standard_normal(&mut rng)).collect()
    }

    fn anchor_offset(&self, normal: &[f64], rows: &MmapColumnStore, anchor_row: usize) -> Result<f64, ENNError> {
        if rows.nrows == 0 {
            return Ok(0.0);
        }
        Ok(dot(normal, &self.scaled(rows.mmap_row_slice(anchor_row)?)))
    }

    pub(crate) fn push_bit(&mut self, rows: &MmapColumnStore) -> Result<(), ENNError> {
        let bit = self.bits();
        for t in 0..self.planes.len() {
            let normal = self.normal(t, bit);
            let draw = mix64(plane_seed(self.seed, t, bit) ^ 0xa5a5);
            let anchor_row = (draw % rows.nrows.max(1) as u64) as usize;
            let offset = self.anchor_offset(&normal, rows, anchor_row)?;
            self.planes[t].push(Hyperplane { normal, anchor_row, offset });
        }
        Ok(())
    }

    pub(crate) fn restore(&mut self, anchors: &[Vec<usize>], rows: &MmapColumnStore) -> Result<(), ENNError> {
        for (t, table_anchors) in anchors.iter().enumerate() {
            for (bit, &anchor_row) in table_anchors.iter().enumerate() {
                let normal = self.normal(t, bit);
                self.planes[t].push(Hyperplane { normal, anchor_row, offset: 0.0 });
            }
        }
        let scale = self.hash_scale.clone();
        self.set_scale(&scale, rows)
    }

    pub(crate) fn set_scale(&mut self, hash_scale: &[f64], rows: &MmapColumnStore) -> Result<(), ENNError> {
        self.hash_scale = hash_scale.to_vec();
        self.inv_scale = hash_scale.iter().map(|s| 1.0 / s).collect();
        for t in 0..self.planes.len() {
            for b in 0..self.planes[t].len() {
                let plane = &self.planes[t][b];
                let offset = self.anchor_offset(&plane.normal, rows, plane.anchor_row)?;
                self.planes[t][b].offset = offset;
            }
        }
        Ok(())
    }

    pub(crate) fn scaled(&self, x: &[f64]) -> Vec<f64> {
        x.iter().zip(&self.inv_scale).map(|(v, w)| v * w).collect()
    }

    pub(crate) fn key_bits(&self, table: usize, z: &[f64], from_bit: usize) -> u64 {
        let mut key = 0u64;
        for plane in &self.planes[table][from_bit..] {
            key = (key << 1) | u64::from(dot(&plane.normal, z) > plane.offset);
        }
        key
    }

    pub(crate) fn query_keys(&self, x: &[f64], num_probes: usize) -> Vec<(u64, Vec<usize>)> {
        let z = self.scaled(x);
        let bits = self.bits();
        self.planes
            .iter()
            .map(|table| {
                let mut key = 0u64;
                let mut margins = Vec::with_capacity(bits);
                for (b, plane) in table.iter().enumerate() {
                    let gap = dot(&plane.normal, &z) - plane.offset;
                    key = (key << 1) | u64::from(gap > 0.0);
                    margins.push((gap.abs(), bits - 1 - b));
                }
                margins.sort_by(|a, b| a.0.total_cmp(&b.0));
                let flips = margins.iter().take(num_probes).map(|m| m.1).collect();
                (key, flips)
            })
            .collect()
    }
}
