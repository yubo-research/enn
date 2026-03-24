//! Posterior draw computation helpers.

use ndarray::Array3;

use crate::draw::DrawInternals;
use crate::error::ENNError;
use crate::hash::normal_hash_batch_multi_seed_fast;
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

    let idx_array: Vec<i64> = internals
        .idx
        .iter()
        .flatten()
        .map(|&i| i as i64)
        .collect();
    let u = normal_hash_batch_multi_seed_fast(function_seeds, &idx_array, m as i64)
        .map_err(|e| ENNError::InvalidParameter(format!("Hash error: {}", e)))?;

    let u = u.into_shape_with_order((num_seeds, n, k, m))?;

    let mut draws = Array3::zeros((num_seeds, n, m));
    for s in 0..num_seeds {
        for i in 0..n {
            for j in 0..m {
                let weighted_u: f64 = (0..k)
                    .map(|ki| internals.w_normalized[[i, ki, j]] * u[[s, i, ki, j]])
                    .sum();
                let l2_safe = internals.l2[[i, j]].max(1e-12);
                draws[[s, i, j]] = internals.mu[[i, j]] + internals.se[[i, j]] * weighted_u / l2_safe;
            }
        }
    }

    Ok(draws)
}
