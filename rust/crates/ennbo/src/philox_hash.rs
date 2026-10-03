//! NumPy Philox + Cephes `ndtri` reference hash.

use ndarray::{Array, ArrayD, IxDyn};

use crate::hash::{unique_index_inverse, HashError};
use crate::ndtri::ndtri;
use crate::philox::NumpyPhilox;

const SEED_PRIME: u64 = 1_000_003;
const CLIP_MIN: f64 = 1e-10;
const CLIP_MAX: f64 = 1.0 - 1e-10;

fn philox_normal(seed: u64, index: i64, metric: usize) -> f64 {
    let combined = seed
        .wrapping_mul(SEED_PRIME)
        .wrapping_add(index as u64)
        .wrapping_mul(SEED_PRIME)
        .wrapping_add(metric as u64);
    let mut rng = NumpyPhilox::from_seed(combined);
    let u = rng.random().clamp(CLIP_MIN, CLIP_MAX);
    ndtri(u)
}

pub fn philox_normal_hash(function_seeds: &[i64], data_indices: &[i64], num_metrics: i64) -> Result<ArrayD<f64>, HashError> {
    if num_metrics <= 0 {
        return Err(HashError::InvalidNumMetrics(num_metrics));
    }
    let num_metrics = num_metrics as usize;
    let (unique, inverse) = unique_index_inverse(data_indices);
    let mut flat = vec![0.0; function_seeds.len() * data_indices.len() * num_metrics];
    for (si, &seed) in function_seeds.iter().enumerate() {
        let mut cache = vec![0.0; unique.len() * num_metrics];
        for (ui, &idx) in unique.iter().enumerate() {
            for metric in 0..num_metrics {
                cache[ui * num_metrics + metric] = philox_normal(seed as u64, idx, metric);
            }
        }
        for (di, &inv) in inverse.iter().enumerate() {
            let dst = ((si * data_indices.len()) + di) * num_metrics;
            let src = inv * num_metrics;
            flat[dst..dst + num_metrics].copy_from_slice(&cache[src..src + num_metrics]);
        }
    }
    Ok(Array::from_shape_vec(IxDyn(&[function_seeds.len(), data_indices.len(), num_metrics]), flat).expect("shape"))
}
