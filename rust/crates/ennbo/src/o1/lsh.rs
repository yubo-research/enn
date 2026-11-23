use ndarray::{s, Array2, ArrayView2};
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
    pub pending_budget_bytes: usize,
}

impl Default for O1Params {
    fn default() -> Self {
        Self {
            num_tables: 8,
            bucket_target: 8,
            bucket_cap: 32,
            num_probes: 4,
            build_budget_bytes: 32 << 20,
            pending_budget_bytes: 32 << 20,
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

struct Hyperplane {
    anchor_row: usize,
}

pub(crate) struct Planes {
    seed: u64,
    dim: usize,
    planes: Vec<Vec<Hyperplane>>,
    hash_scale: Vec<f64>,
    proj: Array2<f64>,
    offsets: Vec<f64>,
}

impl Planes {
    pub(crate) fn new(dim: usize, seed: u64, num_tables: usize) -> Self {
        Self {
            seed,
            dim,
            planes: (0..num_tables).map(|_| Vec::new()).collect(),
            hash_scale: vec![1.0; dim],
            proj: Array2::zeros((dim, 0)),
            offsets: Vec::new(),
        }
    }

    pub(crate) fn cleared(&self) -> Self {
        let mut fresh = Self::new(self.dim, self.seed, self.planes.len());
        fresh.hash_scale.clone_from(&self.hash_scale);
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
        self.planes.len() * self.bits() * (self.dim + 2) * 8 + 8 * self.dim
    }

    fn normal(&self, table: usize, bit: usize) -> Vec<f64> {
        let mut rng = StdRng::seed_from_u64(plane_seed(self.seed, table, bit));
        (0..self.dim).map(|_| standard_normal(&mut rng)).collect()
    }

    fn refresh(&mut self, rows: &MmapColumnStore) -> Result<(), ENNError> {
        let bits = self.bits();
        let mut proj = Array2::zeros((self.dim, self.planes.len() * bits));
        for (t, table) in self.planes.iter().enumerate() {
            for b in 0..table.len() {
                for (j, (v, s)) in self.normal(t, b).iter().zip(&self.hash_scale).enumerate() {
                    proj[[j, t * bits + b]] = v / s;
                }
            }
        }
        self.proj = proj;
        self.offsets = vec![0.0; self.proj.ncols()];
        if rows.nrows == 0 {
            return Ok(());
        }
        for (t, table) in self.planes.iter().enumerate() {
            for (b, plane) in table.iter().enumerate() {
                let c = t * bits + b;
                let x = rows.mmap_rows_view(plane.anchor_row, plane.anchor_row + 1)?;
                self.offsets[c] = x.dot(&self.proj.slice(s![.., c..c + 1]))[[0, 0]];
            }
        }
        Ok(())
    }

    pub(crate) fn push_bit(&mut self, rows: &MmapColumnStore) -> Result<(), ENNError> {
        let bit = self.bits();
        for t in 0..self.planes.len() {
            let draw = mix64(plane_seed(self.seed, t, bit) ^ 0xa5a5);
            let anchor_row = (draw % rows.nrows.max(1) as u64) as usize;
            self.planes[t].push(Hyperplane { anchor_row });
        }
        self.refresh(rows)
    }

    pub(crate) fn restore(&mut self, anchors: &[Vec<usize>], rows: &MmapColumnStore) -> Result<(), ENNError> {
        for (t, table_anchors) in anchors.iter().enumerate() {
            for &anchor_row in table_anchors {
                self.planes[t].push(Hyperplane { anchor_row });
            }
        }
        self.refresh(rows)
    }

    pub(crate) fn set_scale(&mut self, hash_scale: &[f64], rows: &MmapColumnStore) -> Result<(), ENNError> {
        self.hash_scale = hash_scale.to_vec();
        self.refresh(rows)
    }

    fn gaps(&self, x: &ArrayView2<f64>, from_bit: usize) -> Array2<f64> {
        let bits = self.bits();
        let width = bits - from_bit;
        let cols: Vec<usize> = (0..self.planes.len())
            .flat_map(|t| (from_bit..bits).map(move |b| t * bits + b))
            .collect();
        let mut g = if width == bits {
            x.dot(&self.proj)
        } else {
            x.dot(&self.proj.select(ndarray::Axis(1), &cols))
        };
        for (j, &c) in cols.iter().enumerate() {
            g.column_mut(j).mapv_inplace(|v| v - self.offsets[c]);
        }
        g
    }

    pub(crate) fn row_keys(&self, x: &ArrayView2<f64>, from_bit: usize) -> Vec<u64> {
        let width = self.bits() - from_bit;
        let g = self.gaps(x, from_bit);
        let mut out = Vec::with_capacity(x.nrows() * self.planes.len());
        for row in g.rows() {
            for t in 0..self.planes.len() {
                let key = (0..width).fold(0u64, |k, b| (k << 1) | u64::from(row[t * width + b] > 0.0));
                out.push(key);
            }
        }
        out
    }

    pub(crate) fn query_keys(&self, x: &[f64], num_probes: usize) -> Vec<(u64, Vec<usize>)> {
        let bits = self.bits();
        let view = ArrayView2::from_shape((1, x.len()), x).expect("query row");
        let g = self.gaps(&view, 0);
        let row = g.row(0);
        (0..self.planes.len())
            .map(|t| {
                let mut key = 0u64;
                let mut margins = Vec::with_capacity(bits);
                for b in 0..bits {
                    let gap = row[t * bits + b];
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
