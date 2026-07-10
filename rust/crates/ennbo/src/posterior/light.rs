//! Fused posterior fast path when yvar is absent and observation_noise is off.

use ndarray::{Array2, ArrayView2, Axis};

use crate::error::{ENNError, EPS_VAR};
use crate::model::EpistemicNearestNeighbors;
use crate::params::{ENNParams, PosteriorFlags};

use super::{empty_posterior_internals, index_search};

pub(crate) type PosteriorLightOut = (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>, Array2<i64>);

pub(crate) fn idx_nested_to_array2(idx: &[Vec<usize>]) -> Array2<i64> {
    let n_query = idx.len();
    let k = idx.first().map(|r| r.len()).unwrap_or(0);
    let mut out = Array2::from_elem((n_query, k), -1i64);
    for (i, row) in idx.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            out[[i, j]] = v as i64;
        }
    }
    out
}

fn neighbor_search_k(
    model: &EpistemicNearestNeighbors,
    params: &ENNParams,
    exclude_nearest: bool,
) -> Result<usize, ENNError> {
    if exclude_nearest && model.num_obs() <= 1 {
        return Err(ENNError::InvalidParameter(format!(
            "exclude_nearest=True requires at least 2 observations, got {}",
            model.num_obs()
        )));
    }
    Ok(if exclude_nearest {
        (params.k_num_neighbors as usize + 1).min(model.num_obs())
    } else {
        (params.k_num_neighbors as usize).min(model.num_obs())
    })
}

fn fuse_neighbors_to_mu_se(
    model: &EpistemicNearestNeighbors,
    dist2s: &ArrayView2<f64>,
    idx: &ArrayView2<i64>,
    params: &ENNParams,
) -> PosteriorLightOut {
    let n_query = dist2s.nrows();
    let k = dist2s.ncols();
    let num_metrics = model.num_metrics();
    let y_scale = model.y_scale();
    let epistemic_scale = params.epistemic_variance_scale;
    let aleatoric_scale = params.aleatoric_variance_scale;

    let mut mu = Array2::zeros((n_query, num_metrics));
    let mut se = Array2::zeros((n_query, num_metrics));
    let mut se_epi = Array2::zeros((n_query, num_metrics));
    let mut se_ale = Array2::zeros((n_query, num_metrics));
    let mut idx_out = Array2::from_elem((n_query, k), -1i64);
    let mut w = vec![0.0f64; k];

    for i in 0..n_query {
        let dist_row = dist2s.row(i);
        let idx_row = idx.row(i);

        let mut norm = 0.0;
        for j in 0..k {
            let var_total = EPS_VAR + epistemic_scale * dist_row[j] + aleatoric_scale;
            w[j] = 1.0 / var_total;
            norm += w[j];
        }
        let inv_norm = 1.0 / norm;
        let se_base = inv_norm.max(EPS_VAR).sqrt();

        for j in 0..k {
            idx_out[[i, j]] = idx_row[j];
        }

        for m in 0..num_metrics {
            let mut mu_val = 0.0;
            let y_scale_m = y_scale[m];
            if let Some(train_y_view) = model.train_y_view_opt() {
                for j in 0..k {
                    let w_norm = w[j] * inv_norm;
                    mu_val += w_norm * train_y_view[[idx_row[j] as usize, m]];
                }
            } else {
                for j in 0..k {
                    let w_norm = w[j] * inv_norm;
                    let y_row = model.rows().row_y(idx_row[j] as usize).expect("row_y");
                    mu_val += w_norm * y_row[m];
                }
            }
            mu[[i, m]] = mu_val;
            let se_val = se_base * y_scale_m;
            se[[i, m]] = se_val;
            se_epi[[i, m]] = se_val;
            se_ale[[i, m]] = 0.0;
        }
    }

    (mu, se, se_epi, se_ale, idx_out)
}

/// Fused index_search + mu/se for the no-yvar, no-observation-noise posterior path.
pub(crate) fn compute_posterior_light(
    model: &EpistemicNearestNeighbors,
    x: &ArrayView2<f64>,
    params: &ENNParams,
    flags: &PosteriorFlags,
) -> Result<PosteriorLightOut, ENNError> {
    if x.ncols() != model.num_dim() {
        return Err(ENNError::InvalidShape {
            expected: vec![x.nrows(), model.num_dim()],
            got: x.shape().to_vec(),
        });
    }

    let batch_size = x.nrows();
    if model.num_obs() == 0 {
        let internals = empty_posterior_internals(model, batch_size);
        return Ok((
            internals.mu,
            internals.se,
            internals.se_epi,
            internals.se_ale,
            idx_nested_to_array2(&internals.idx),
        ));
    }

    let search_k = neighbor_search_k(model, params, flags.exclude_nearest)?;
    if search_k == 0 {
        let internals = empty_posterior_internals(model, batch_size);
        return Ok((
            internals.mu,
            internals.se,
            internals.se_epi,
            internals.se_ale,
            idx_nested_to_array2(&internals.idx),
        ));
    }

    let (dist2s_full, idx_full) = index_search(
        model,
        x,
        search_k as i32,
        flags.exclude_nearest,
        flags.tie_break_neighbors,
    )?;

    let available_k = if flags.exclude_nearest {
        search_k.saturating_sub(1)
    } else {
        search_k
    };
    let k = (params.k_num_neighbors as usize).min(available_k);
    if k == 0 {
        let internals = empty_posterior_internals(model, batch_size);
        return Ok((
            internals.mu,
            internals.se,
            internals.se_epi,
            internals.se_ale,
            idx_nested_to_array2(&internals.idx),
        ));
    }

    let dist2s = dist2s_full.slice_axis(Axis(1), ndarray::Slice::from(..k));
    let idx = idx_full.slice_axis(Axis(1), ndarray::Slice::from(..k));
    Ok(fuse_neighbors_to_mu_se(model, &dist2s, &idx, params))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::IndexDriver;
    use crate::model::EpistemicNearestNeighbors;
    use crate::posterior::compute_posterior_internals;
    use crate::test_helpers::test_epistemic_model_exact_unit_square as create_test_model;
    use ndarray::array;

    #[test]
    fn test_posterior_light_matches_weighted_path_on_self_search() {
        let n = 32;
        let d = 4;
        let m = 2;
        let train_x =
            Array2::from_shape_fn((n, d), |(i, j)| (i as f64 * 0.1) + (j as f64 * 0.01));
        let train_y = Array2::from_shape_fn((n, m), |(i, j)| (i as f64) + (j as f64));
        let model = EpistemicNearestNeighbors::new(
            train_x.clone(),
            train_y,
            None,
            false,
            IndexDriver::Exact,
        )
        .unwrap();
        let params = ENNParams::new(5, 1.0, 0.1).unwrap();
        let flags = PosteriorFlags::new();

        let (mu_light, se_light, se_epi_light, se_ale_light, idx_light) =
            compute_posterior_light(&model, &train_x.view(), &params, &flags).unwrap();
        let full = compute_posterior_internals(&model, &train_x.view(), &params, &flags).unwrap();

        assert_eq!(mu_light.shape(), full.mu.shape());
        assert_eq!(se_light.shape(), full.se.shape());
        assert!((mu_light - &full.mu).mapv(f64::abs).iter().all(|&d| d < 1e-12));
        assert!((se_light - &full.se).mapv(f64::abs).iter().all(|&d| d < 1e-12));
        assert!((se_epi_light - &full.se_epi).mapv(f64::abs).iter().all(|&d| d < 1e-12));
        assert!((se_ale_light - &full.se_ale).mapv(f64::abs).iter().all(|&d| d < 1e-12));
        assert_eq!(idx_light, idx_nested_to_array2(&full.idx));
    }

    #[test]
    fn test_neighbor_search_k_exclude_nearest_requires_two_observations() {
        let model = EpistemicNearestNeighbors::new(
            array![[0.0, 0.0]],
            array![[1.0]],
            None,
            false,
            IndexDriver::Exact,
        )
        .unwrap();
        let params = ENNParams::new(1, 1.0, 0.1).unwrap();
        assert!(neighbor_search_k(&model, &params, true).is_err());
        assert_eq!(neighbor_search_k(&model, &params, false).unwrap(), 1);
    }

    #[test]
    fn test_idx_nested_to_array2_round_trip() {
        let nested = vec![vec![0usize, 2], vec![1, 3]];
        let arr = idx_nested_to_array2(&nested);
        assert_eq!(arr[[0, 0]], 0);
        assert_eq!(arr[[1, 1]], 3);
        let empty = idx_nested_to_array2(&[]);
        assert_eq!(empty.shape(), &[0, 0]);
    }

    #[test]
    fn test_compute_posterior_light_empty_query() {
        let model = create_test_model();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let flags = PosteriorFlags::new();
        let empty_query: Array2<f64> = Array2::zeros((0, 2));
        let (mu, se, _se_epi, _se_ale, idx) =
            compute_posterior_light(&model, &empty_query.view(), &params, &flags).unwrap();
        assert_eq!(mu.nrows(), 0);
        assert_eq!(se.nrows(), 0);
        assert!(idx.is_empty());
    }

    #[test]
    fn test_compute_posterior_light_invalid_query_shape() {
        let model = create_test_model();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let flags = PosteriorFlags::new();
        let bad_query = array![[0.5]];
        assert!(compute_posterior_light(&model, &bad_query.view(), &params, &flags).is_err());
    }

    #[test]
    fn test_compute_posterior_light_empty_model() {
        let model = EpistemicNearestNeighbors::new(
            Array2::zeros((0, 2)),
            Array2::zeros((0, 1)),
            None,
            false,
            IndexDriver::Exact,
        )
        .unwrap();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let flags = PosteriorFlags::new();
        let query = array![[0.5, 0.5]];
        let (mu, se, _se_epi, _se_ale, idx) =
            compute_posterior_light(&model, &query.view(), &params, &flags).unwrap();
        assert_eq!(mu.nrows(), 1);
        assert_eq!(se.nrows(), 1);
        assert_eq!(idx.nrows(), 1);
    }

    #[test]
    fn test_fuse_neighbors_to_mu_se_on_small_batch() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let train_y = array![[1.0], [2.0], [3.0], [4.0]];
        let model =
            EpistemicNearestNeighbors::new(train_x, train_y, None, false, IndexDriver::Exact)
                .unwrap();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let flags = PosteriorFlags::new();
        let all: Vec<usize> = (0..model.len()).collect();
        let (tx, _, _) = model.rows().train_rows_at(&all).unwrap();
        let (mu_light, se_light, _se_epi_light, _se_ale_light, idx_light) =
            compute_posterior_light(&model, &tx.view(), &params, &flags).unwrap();
        let dist2s = array![[0.0, 1.0], [1.0, 0.0], [2.0, 1.0], [1.0, 2.0]];
        let idx = array![[0, 1], [1, 0], [2, 3], [3, 2]];
        let (mu, se, _se_epi, _se_ale, idx_out) =
            fuse_neighbors_to_mu_se(&model, &dist2s.view(), &idx.view(), &params);
        assert_eq!(mu.nrows(), 4);
        assert_eq!(se.nrows(), 4);
        assert_eq!(idx_out.shape(), &[4, 2]);
        assert!(mu.iter().all(|v| v.is_finite()));
        assert!(se.iter().all(|&v| v > 0.0));
        assert_eq!(mu_light.nrows(), 4);
        assert_eq!(se_light.nrows(), 4);
        assert_eq!(idx_light.nrows(), 4);
    }

    #[test]
    fn test_compute_posterior_light_multiple_metrics() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let train_y = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let model =
            EpistemicNearestNeighbors::new(train_x, train_y, None, false, IndexDriver::Exact)
                .unwrap();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let flags = PosteriorFlags::new();
        let all: Vec<usize> = (0..model.len()).collect();
        let (tx, _, _) = model.rows().train_rows_at(&all).unwrap();
        let (mu, se, _se_epi, _se_ale, idx) = compute_posterior_light(
            &model,
            &tx.view(),
            &params,
            &flags,
        )
        .unwrap();
        assert_eq!(mu.ncols(), 2);
        assert_eq!(se.ncols(), 2);
        assert_eq!(idx.ncols(), 2);
        assert!(mu.iter().all(|v| v.is_finite()));
        assert!(se.iter().all(|&v| v > 0.0));
    }
}
