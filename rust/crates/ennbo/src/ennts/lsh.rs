use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::error::ENNError;

pub(super) struct LshFamily {
    width: f64,
    num_tables: usize,
    num_projections: usize,
    window_dim: usize,
    projections: Vec<f64>,
    offsets: Vec<f64>,
}

impl LshFamily {
    pub(super) fn create(
        window_dim: usize,
        num_tables: usize,
        num_projections: usize,
        width: f64,
        seed: u64,
    ) -> Result<Self, ENNError> {
        if window_dim == 0
            || num_tables == 0
            || num_projections == 0
            || !(width.is_finite() && width > 0.0)
        {
            return Err(ENNError::InvalidParameter(
                "LSH requires a positive window, tables, projections, and bucket width".into(),
            ));
        }
        let n_vec = num_tables * num_projections;
        let mut rng = StdRng::seed_from_u64(seed);
        let mut spare = None;
        let mut projections = Vec::with_capacity(n_vec * window_dim);
        for _ in 0..(n_vec * window_dim) {
            projections.push(standard_normal(&mut rng, &mut spare));
        }
        let offsets: Vec<f64> = (0..n_vec).map(|_| rng.gen::<f64>() * width).collect();
        Ok(Self {
            width,
            num_tables,
            num_projections,
            window_dim,
            projections,
            offsets,
        })
    }

    pub(super) fn hash_table(&self, table: usize, window: &[f64]) -> Result<Vec<i32>, ENNError> {
        if window.len() != self.window_dim || table >= self.num_tables {
            return Err(ENNError::InvalidParameter(
                "LSH hash received a window with the wrong length".into(),
            ));
        }
        let mut key = Vec::with_capacity(self.num_projections);
        for proj in 0..self.num_projections {
            let dot = dot(self.projection(table, proj), window);
            key.push(bucket_index(dot, self.offset(table, proj), self.width)?);
        }
        Ok(key)
    }

    fn projection(&self, table: usize, proj: usize) -> &[f64] {
        let start = (table * self.num_projections + proj) * self.window_dim;
        &self.projections[start..start + self.window_dim]
    }

    fn offset(&self, table: usize, proj: usize) -> f64 {
        self.offsets[table * self.num_projections + proj]
    }
}

pub(super) fn probed_keys(base: &[i32], probe: i32) -> Vec<Vec<i32>> {
    let mut keys = vec![base.to_vec()];
    if probe <= 0 || base.is_empty() {
        return keys;
    }
    for delta in 1..=probe {
        let mut up = base.to_vec();
        let mut down = base.to_vec();
        up[0] = up[0].saturating_add(delta);
        down[0] = down[0].saturating_sub(delta);
        keys.push(up);
        keys.push(down);
    }
    keys
}

fn dot(projection: &[f64], window: &[f64]) -> f64 {
    projection
        .iter()
        .zip(window.iter())
        .map(|(a, b)| a * b)
        .sum()
}

fn standard_normal(rng: &mut StdRng, spare: &mut Option<f64>) -> f64 {
    if let Some(z) = spare.take() {
        return z;
    }
    let u1 = rng.gen::<f64>().clamp(1e-12, 1.0 - 1e-12);
    let u2 = rng.gen::<f64>();
    let radius = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    *spare = Some(radius * theta.sin());
    radius * theta.cos()
}

fn bucket_index(dot: f64, offset: f64, width: f64) -> Result<i32, ENNError> {
    let raw = (dot + offset) / width;
    if !raw.is_finite() {
        return Err(ENNError::InvalidParameter(
            "non-finite locality-sensitive hash projection".into(),
        ));
    }
    Ok(raw.floor().clamp(i32::MIN as f64, i32::MAX as f64) as i32)
}
