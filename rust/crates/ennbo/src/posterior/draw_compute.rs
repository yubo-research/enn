//! Posterior draw computation helpers.

use ndarray::Array3;
use rayon::prelude::*;

use crate::draw::DrawInternals;
use crate::error::ENNError;
use crate::hash::{normal_for_seed_index_metric, unique_index_inverse};
use crate::model::EpistemicNearestNeighbors;

pub(crate) fn draw_from_internals(
    model: &EpistemicNearestNeighbors,
    internals: &DrawInternals,
    function_seeds: &[i64],
) -> Result<Array3<f64>, ENNError> {
    let n = internals.mu.nrows();
    let k = internals.n_neighbors();
    let m = model.num_metrics();
    let num_seeds = function_seeds.len();

    if k == 0 {
        let mut draws = Array3::zeros((num_seeds, n, m));
        for s in 0..num_seeds {
            for i in 0..n {
                for j in 0..m {
                    draws[[s, i, j]] = internals.mu[[i, j]];
                }
            }
        }
        return Ok(draws);
    }

    // Fuse hash → weighted sum per seed: never materialize u[s,i,k,j].
    let idx_array: Vec<i64> = internals.idx.iter().flatten().map(|&i| i as i64).collect();
    let (unique_indices, inverse) = unique_index_inverse(&idx_array);
    let n_unique = unique_indices.len();

    if m == 1 {
        return draw_from_internals_m1(
            internals,
            function_seeds,
            &unique_indices,
            &inverse,
            n,
            k,
            n_unique,
        );
    }

    let out_stride = n * m;
    let mut scale = vec![0.0f64; n * m];
    let mut w_flat = vec![0.0f64; n * k * m];
    let mut mu_flat = vec![0.0f64; n * m];
    for i in 0..n {
        for j in 0..m {
            let l2_safe = internals.l2[[i, j]].max(1e-12);
            scale[i * m + j] = internals.se[[i, j]] / l2_safe;
            mu_flat[i * m + j] = internals.mu[[i, j]];
            for ki in 0..k {
                w_flat[(i * k + ki) * m + j] = internals.w_normalized[[i, ki, j]];
            }
        }
    }

    let mut draws = Array3::zeros((num_seeds, n, m));
    let draws_flat = draws
        .as_slice_memory_order_mut()
        .ok_or_else(|| ENNError::InvalidParameter("draws must be contiguous".into()))?;

    draws_flat
        .par_chunks_mut(out_stride)
        .zip(function_seeds.par_iter())
        .for_each_init(
            || vec![0.0f64; n_unique * m],
            |unique_cache, (out_seed, &seed)| {
                let seed_u64 = seed as u64;
                for (ui, &unique_idx) in unique_indices.iter().enumerate() {
                    for j in 0..m {
                        unique_cache[ui * m + j] =
                            normal_for_seed_index_metric(seed_u64, unique_idx, j);
                    }
                }
                for i in 0..n {
                    for j in 0..m {
                        let mut weighted_u = 0.0;
                        for ki in 0..k {
                            let inv = inverse[i * k + ki];
                            let w = w_flat[(i * k + ki) * m + j];
                            weighted_u += w * unique_cache[inv * m + j];
                        }
                        out_seed[i * m + j] =
                            mu_flat[i * m + j] + scale[i * m + j] * weighted_u;
                    }
                }
            },
        );

    Ok(draws)
}

/// Specialized fused draw path for the common single-metric case.
fn draw_from_internals_m1(
    internals: &DrawInternals,
    function_seeds: &[i64],
    unique_indices: &[i64],
    inverse: &[usize],
    n: usize,
    k: usize,
    n_unique: usize,
) -> Result<Array3<f64>, ENNError> {
    let num_seeds = function_seeds.len();
    let mut scale = vec![0.0f64; n];
    let mut mu_flat = vec![0.0f64; n];
    let mut w_flat = vec![0.0f64; n * k];
    for i in 0..n {
        let l2_safe = internals.l2[[i, 0]].max(1e-12);
        scale[i] = internals.se[[i, 0]] / l2_safe;
        mu_flat[i] = internals.mu[[i, 0]];
        for ki in 0..k {
            w_flat[i * k + ki] = internals.w_normalized[[i, ki, 0]];
        }
    }

    let mut draws = Array3::zeros((num_seeds, n, 1));
    let draws_flat = draws
        .as_slice_memory_order_mut()
        .ok_or_else(|| ENNError::InvalidParameter("draws must be contiguous".into()))?;

    draws_flat
        .par_chunks_mut(n)
        .zip(function_seeds.par_iter())
        .for_each_init(
            || vec![0.0f64; n_unique],
            |unique_cache, (out_seed, &seed)| {
                let seed_u64 = seed as u64;
                for (ui, &unique_idx) in unique_indices.iter().enumerate() {
                    unique_cache[ui] = normal_for_seed_index_metric(seed_u64, unique_idx, 0);
                }
                for i in 0..n {
                    let mut weighted_u = 0.0;
                    let base = i * k;
                    for ki in 0..k {
                        weighted_u += w_flat[base + ki] * unique_cache[inverse[base + ki]];
                    }
                    out_seed[i] = mu_flat[i] + scale[i] * weighted_u;
                }
            },
        );

    Ok(draws)
}
