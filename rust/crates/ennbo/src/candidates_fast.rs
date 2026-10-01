//! `RAASPDriver.FAST`: binomial perturbation count, uniform in the trust region.

use ndarray::{Array1, Array2, ArrayView1};
use rand::Rng;

use crate::error::ENNError;

fn binomial<R: Rng + ?Sized>(rng: &mut R, n: usize, p: f64) -> usize {
    (0..n).filter(|_| rng.gen::<f64>() < p).count()
}

fn choose<R: Rng + ?Sized>(rng: &mut R, n: usize, k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    for i in 0..k {
        let j = i + rng.gen_range(0..(n - i));
        idx.swap(i, j);
    }
    idx.truncate(k);
    idx
}

pub fn generate_tr_candidates_fast<R: Rng + ?Sized>(
    x_center: &ArrayView1<f64>,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    num_candidates: usize,
    rng: &mut R,
    num_pert: usize,
) -> Result<Array2<f64>, ENNError> {
    let num_dim = x_center.len();
    if num_dim == 0 || num_candidates == 0 {
        return Ok(Array2::zeros((num_candidates, num_dim)));
    }
    let prob = (num_pert as f64 / num_dim as f64).min(1.0);
    let mut candidates = Array2::zeros((num_candidates, num_dim));
    for i in 0..num_candidates {
        for j in 0..num_dim {
            candidates[[i, j]] = x_center[j];
        }
        let k = binomial(rng, num_dim, prob).max(1);
        for dim in choose(rng, num_dim, k) {
            let u: f64 = rng.gen();
            candidates[[i, dim]] = lower[dim] + (upper[dim] - lower[dim]) * u;
        }
    }
    Ok(candidates)
}
