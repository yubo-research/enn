//! Posterior computation for ENN model.

mod draw_compute;
mod neighbor;

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2, Axis};

use self::draw_compute::draw_from_internals;
use self::neighbor::{get_conditional_neighbor_data, get_neighbor_data};
use crate::draw::DrawInternals;
use crate::error::{ENNError, EPS_VAR};
use crate::model::EpistemicNearestNeighbors;
use crate::params::{ENNNormal, ENNParams, PosteriorFlags};
use crate::stats::WeightedStats;
use crate::traits::PosteriorComputation;

impl PosteriorComputation for EpistemicNearestNeighbors {
    fn posterior(
        &self,
        x: &ArrayView2<f64>,
        params: &ENNParams,
        flags: &PosteriorFlags,
    ) -> Result<ENNNormal, ENNError> {
        let internals = compute_posterior_internals(self, x, params, flags)?;
        Ok(ENNNormal::new(
            internals.mu.into_dyn(),
            internals.se.into_dyn(),
            Some(internals.idx),
        ))
    }

    fn batch_posterior(
        &self,
        x: &ArrayView2<f64>,
        paramss: &[ENNParams],
        flags: &PosteriorFlags,
    ) -> Result<ENNNormal, ENNError> {
        if paramss.is_empty() {
            return Err(ENNError::InvalidParameter(
                "paramss must be non-empty".to_string(),
            ));
        }

        let batch_size = x.nrows();
        let num_params = paramss.len();

        let mut mu_all = Array3::zeros((num_params, batch_size, self.num_metrics()));
        let mut se_all = Array3::zeros((num_params, batch_size, self.num_metrics()));

        let k_values: std::collections::HashSet<i32> =
            paramss.iter().map(|p| p.k_num_neighbors).collect();

        if k_values.len() == 1 && self.num_obs() > 0 {
            compute_batch_with_shared_neighbors(self, x, paramss, flags, &mut mu_all, &mut se_all)?;
        } else {
            compute_batch_separate_neighbors(self, x, paramss, flags, &mut mu_all, &mut se_all)?;
        }

        Ok(ENNNormal::new(mu_all.into_dyn(), se_all.into_dyn(), None))
    }

    fn posterior_function_draw(
        &self,
        x: &ArrayView2<f64>,
        params: &ENNParams,
        function_seeds: &[i64],
        flags: &PosteriorFlags,
    ) -> Result<(Array3<f64>, Vec<Vec<usize>>), ENNError> {
        let internals = compute_posterior_internals(self, x, params, flags)?;
        let draws = draw_from_internals(self, &internals, function_seeds)?;
        Ok((draws, internals.idx))
    }

    fn conditional_posterior(
        &self,
        x_whatif: &ArrayView2<f64>,
        y_whatif: &ArrayView2<f64>,
        x: &ArrayView2<f64>,
        params: &ENNParams,
        flags: &PosteriorFlags,
    ) -> Result<ENNNormal, ENNError> {
        let internals =
            compute_conditional_posterior_internals(self, x, x_whatif, y_whatif, params, flags)?;
        Ok(ENNNormal::new(
            internals.mu.into_dyn(),
            internals.se.into_dyn(),
            Some(internals.idx),
        ))
    }

    fn conditional_posterior_function_draw(
        &self,
        x_whatif: &ArrayView2<f64>,
        y_whatif: &ArrayView2<f64>,
        x: &ArrayView2<f64>,
        params: &ENNParams,
        function_seeds: &[i64],
        flags: &PosteriorFlags,
    ) -> Result<(Array3<f64>, Vec<Vec<usize>>), ENNError> {
        let internals =
            compute_conditional_posterior_internals(self, x, x_whatif, y_whatif, params, flags)?;
        let draws = draw_from_internals(self, &internals, function_seeds)?;
        Ok((draws, internals.idx))
    }
}

fn compute_batch_with_shared_neighbors(
    model: &EpistemicNearestNeighbors,
    x: &ArrayView2<f64>,
    paramss: &[ENNParams],
    flags: &PosteriorFlags,
    mu_all: &mut Array3<f64>,
    se_all: &mut Array3<f64>,
) -> Result<(), ENNError> {
    let neighbor_data = get_neighbor_data(model, x, &paramss[0], flags.exclude_nearest)?;

    if let Some(data) = neighbor_data {
        let wp_data = WeightedPosteriorData {
            dist2s: &data.dist2s.view(),
            idx: &data.idx,
            y_neighbors: &data.y_neighbors.view(),
            params: &paramss[0],
            observation_noise: flags.observation_noise,
            yvar_neighbors_override: None,
        };

        for (i, params) in paramss.iter().enumerate() {
            let data_with_params = WeightedPosteriorData { params, ..wp_data };
            let internals = compute_weighted_posterior(model, data_with_params, None)?;
            assign_posterior_results(&internals, mu_all, se_all, i);
        }
    } else {
        let batch_size = x.nrows();
        let internals = empty_posterior_internals(model, batch_size);
        for i in 0..paramss.len() {
            assign_posterior_results(&internals, mu_all, se_all, i);
        }
    }
    Ok(())
}

fn compute_batch_separate_neighbors(
    model: &EpistemicNearestNeighbors,
    x: &ArrayView2<f64>,
    paramss: &[ENNParams],
    flags: &PosteriorFlags,
    mu_all: &mut Array3<f64>,
    se_all: &mut Array3<f64>,
) -> Result<(), ENNError> {
    for (i, params) in paramss.iter().enumerate() {
        let internals = compute_posterior_internals(model, x, params, flags)?;
        assign_posterior_results(&internals, mu_all, se_all, i);
    }
    Ok(())
}

fn assign_posterior_results(
    internals: &DrawInternals,
    mu_all: &mut Array3<f64>,
    se_all: &mut Array3<f64>,
    index: usize,
) {
    let slice = ndarray::Slice::from(index..index + 1);
    mu_all
        .slice_axis_mut(Axis(0), slice)
        .assign(&internals.mu.slice_axis(Axis(0), ndarray::Slice::from(..)));
    se_all
        .slice_axis_mut(Axis(0), slice)
        .assign(&internals.se.slice_axis(Axis(0), ndarray::Slice::from(..)));
}

/// Data for weighted posterior computation.
pub struct WeightedPosteriorData<'a> {
    pub dist2s: &'a ArrayView2<'a, f64>,
    pub idx: &'a [Vec<usize>],
    pub y_neighbors: &'a ArrayView2<'a, f64>,
    pub params: &'a ENNParams,
    pub observation_noise: bool,
    /// When set, use this instead of gathering from model (for conditional with whatif).
    pub yvar_neighbors_override: Option<&'a Array2<f64>>,
}

pub fn compute_weighted_posterior(
    model: &EpistemicNearestNeighbors,
    data: WeightedPosteriorData<'_>,
    y_scale_override: Option<&ArrayView1<'_, f64>>,
) -> Result<DrawInternals, ENNError> {
    let y_scale: Array1<f64> = y_scale_override
        .map(|v| v.to_owned())
        .unwrap_or_else(|| model.y_scale().clone());

    let yvar_neighbors: Option<Array2<f64>> = if let Some(ov) = data.yvar_neighbors_override {
        Some(ov.clone())
    } else if let Some(yvar) = model.train_yvar() {
        let n_query = data.dist2s.nrows();
        // Handle empty query case (n_query == 0 or data.idx is empty)
        let k = if data.idx.is_empty() {
            0
        } else {
            data.idx[0].len()
        };
        let n_train = model.num_obs();
        let mut yvar_neighbors = Array2::zeros((n_query * k, model.num_metrics()));
        for i in 0..n_query {
            for (j, &neighbor_idx) in data.idx[i].iter().enumerate() {
                if neighbor_idx < n_train {
                    let yvar_row = yvar.row(neighbor_idx);
                    for m in 0..model.num_metrics() {
                        yvar_neighbors[[i * k + j, m]] = yvar_row[m];
                    }
                }
                // else: whatif point, keep 0
            }
        }
        Some(yvar_neighbors)
    } else {
        None
    };

    let stats = compute_weighted_stats_impl(
        data.dist2s,
        data.y_neighbors,
        yvar_neighbors.as_ref().map(|v| v.view()),
        data.params,
        data.observation_noise,
        &y_scale.view(),
    )?;

    Ok(DrawInternals::new(
        data.idx.to_vec(),
        stats.w_normalized,
        stats.l2,
        stats.mu,
        stats.se,
    ))
}

pub fn compute_weighted_stats_impl(
    dist2s: &ArrayView2<f64>,
    y_neighbors: &ArrayView2<f64>,
    yvar_neighbors: Option<ArrayView2<f64>>,
    params: &ENNParams,
    observation_noise: bool,
    y_scale: &ArrayView1<f64>,
) -> Result<WeightedStats, ENNError> {
    let n_query = dist2s.nrows();
    let k = dist2s.ncols();
    let num_metrics = y_scale.len();

    // Hoist constants outside loops
    let epistemic_scale = params.epistemic_variance_scale;
    let aleatoric_scale = params.aleatoric_variance_scale;
    let y_scale_sq: Vec<f64> = y_scale.iter().map(|&v| v * v).collect();

    // Pre-compute var_epi using iterator-based zip for better cache efficiency
    let mut var_epi = Array2::zeros((n_query, k));
    for (i, mut row) in var_epi.rows_mut().into_iter().enumerate() {
        let dist_row = dist2s.row(i);
        for (j, v) in row.iter_mut().enumerate() {
            *v = epistemic_scale * dist_row[j];
        }
    }

    // Compute weights w with hoisted constants and pre-allocated storage
    let mut w = Array2::zeros((n_query * k, num_metrics));
    let yvar_ref = yvar_neighbors.as_ref();

    for i in 0..n_query {
        let var_epi_row = var_epi.row(i);
        for j in 0..k {
            let var_epi_ij = var_epi_row[j];
            let idx = i * k + j;
            for m in 0..num_metrics {
                let var_y = if let Some(yv) = yvar_ref {
                    yv[[idx, m]] / y_scale_sq[m]
                } else {
                    0.0
                };
                let var_total = EPS_VAR + var_epi_ij + aleatoric_scale + var_y;
                w[[idx, m]] = 1.0 / var_total;
            }
        }
    }

    let mut w_normalized = Array3::zeros((n_query, k, num_metrics));
    let mut l2 = Array2::zeros((n_query, num_metrics));
    let mut mu = Array2::zeros((n_query, num_metrics));
    let mut se = Array2::zeros((n_query, num_metrics));

    // Process each query row and metric with optimized inner loops
    for i in 0..n_query {
        for m in 0..num_metrics {
            // Compute normalization factor (sum of weights for this query/metric)
            let base_idx = i * k;
            let norm: f64 = (0..k).map(|j| w[[base_idx + j, m]]).sum();
            let inv_norm = 1.0 / norm;

            // Normalize weights and compute l2 norm in a single pass when possible
            let mut l2_sq = 0.0;
            let mut mu_val = 0.0;

            for j in 0..k {
                let w_norm = w[[base_idx + j, m]] * inv_norm;
                w_normalized[[i, j, m]] = w_norm;
                l2_sq += w_norm * w_norm;
                mu_val += w_norm * y_neighbors[[base_idx + j, m]];
            }

            l2[[i, m]] = l2_sq.sqrt();
            mu[[i, m]] = mu_val;

            let epistemic_var = inv_norm;

            let aleatoric_var = if observation_noise {
                let mut sum = 0.0;
                for j in 0..k {
                    let var_ale_j = aleatoric_scale
                        + if let Some(yv) = yvar_ref {
                            yv[[base_idx + j, m]] / y_scale_sq[m]
                        } else {
                            0.0
                        };
                    sum += w_normalized[[i, j, m]] * var_ale_j;
                }
                sum
            } else {
                0.0
            };

            se[[i, m]] = (epistemic_var + aleatoric_var).max(EPS_VAR).sqrt() * y_scale[m];
        }
    }

    Ok(WeightedStats::new(w_normalized, l2, mu, se))
}

pub fn compute_posterior_internals(
    model: &EpistemicNearestNeighbors,
    x: &ArrayView2<f64>,
    params: &ENNParams,
    flags: &PosteriorFlags,
) -> Result<DrawInternals, ENNError> {
    if x.ncols() != model.num_dim() {
        return Err(ENNError::InvalidShape {
            expected: vec![x.nrows(), model.num_dim()],
            got: x.shape().to_vec(),
        });
    }

    let batch_size = x.nrows();

    if model.num_obs() == 0 {
        return Ok(empty_posterior_internals(model, batch_size));
    }

    let neighbor_data = get_neighbor_data(model, x, params, flags.exclude_nearest)?;

    if let Some(data) = neighbor_data {
        let wp_data = WeightedPosteriorData {
            dist2s: &data.dist2s.view(),
            idx: &data.idx,
            y_neighbors: &data.y_neighbors.view(),
            params,
            observation_noise: flags.observation_noise,
            yvar_neighbors_override: None,
        };
        compute_weighted_posterior(model, wp_data, None)
    } else {
        Ok(empty_posterior_internals(model, batch_size))
    }
}

fn compute_scale_for_conditional(
    train_y: &ArrayView2<f64>,
    y_whatif: &ArrayView2<f64>,
) -> Array1<f64> {
    let n1 = train_y.nrows();
    let n2 = y_whatif.nrows();
    let m = train_y.ncols();
    let mut stacked = Array2::zeros((n1 + n2, m));
    stacked.slice_mut(ndarray::s![..n1, ..]).assign(train_y);
    stacked.slice_mut(ndarray::s![n1.., ..]).assign(y_whatif);

    if stacked.nrows() < 2 {
        return Array1::ones(stacked.ncols());
    }
    let mut scale = Array1::zeros(stacked.ncols());
    for j in 0..stacked.ncols() {
        let col = stacked.column(j);
        let var = col.var(0.0);
        let std = var.sqrt();
        scale[j] = if std.is_finite() && std > 0.0 {
            std
        } else {
            1.0
        };
    }
    scale
}

pub fn compute_conditional_posterior_internals(
    model: &EpistemicNearestNeighbors,
    x: &ArrayView2<f64>,
    x_whatif: &ArrayView2<f64>,
    y_whatif: &ArrayView2<f64>,
    params: &ENNParams,
    flags: &PosteriorFlags,
) -> Result<DrawInternals, ENNError> {
    if x_whatif.nrows() == 0 {
        return compute_posterior_internals(model, x, params, flags);
    }

    let validation_err: Option<ENNError> = if x.iter().any(|v| !v.is_finite())
        || x_whatif.iter().any(|v| !v.is_finite())
        || y_whatif.iter().any(|v| !v.is_finite())
    {
        Some(ENNError::InvalidParameter(
            "NaN or Inf not allowed in x, x_whatif, or y_whatif".to_string(),
        ))
    } else if x.ncols() != model.num_dim() {
        Some(ENNError::InvalidShape {
            expected: vec![x.nrows(), model.num_dim()],
            got: x.shape().to_vec(),
        })
    } else if x_whatif.ncols() != model.num_dim() {
        Some(ENNError::InvalidShape {
            expected: vec![x_whatif.nrows(), model.num_dim()],
            got: x_whatif.shape().to_vec(),
        })
    } else if y_whatif.ncols() != model.num_metrics() {
        Some(ENNError::InvalidShape {
            expected: vec![y_whatif.nrows(), model.num_metrics()],
            got: y_whatif.shape().to_vec(),
        })
    } else if x_whatif.nrows() != y_whatif.nrows() {
        Some(ENNError::InvalidParameter(
            "x_whatif and y_whatif must have same number of rows".to_string(),
        ))
    } else {
        None
    };
    if let Some(e) = validation_err {
        return Err(e);
    }

    let batch_size = x.nrows();
    let neighbor_data = get_conditional_neighbor_data(model, x, x_whatif, y_whatif, params, flags)?;

    if let Some(data) = neighbor_data {
        let y_scale_cond = compute_scale_for_conditional(&model.train_y().view(), y_whatif);
        let wp_data = WeightedPosteriorData {
            dist2s: &data.dist2s.view(),
            idx: &data.idx,
            y_neighbors: &data.y_neighbors.view(),
            params,
            observation_noise: flags.observation_noise,
            yvar_neighbors_override: None,
        };
        compute_weighted_posterior(model, wp_data, Some(&y_scale_cond.view()))
    } else {
        Ok(empty_posterior_internals(model, batch_size))
    }
}

pub fn empty_posterior_internals(
    model: &EpistemicNearestNeighbors,
    batch_size: usize,
) -> DrawInternals {
    DrawInternals::new(
        vec![vec![]; batch_size],
        Array3::zeros((batch_size, 0, model.num_metrics())),
        Array2::ones((batch_size, model.num_metrics())),
        Array2::zeros((batch_size, model.num_metrics())),
        Array2::ones((batch_size, model.num_metrics())),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::IndexDriver;
    use crate::model::EpistemicNearestNeighbors;
    use crate::test_helpers::test_epistemic_model_exact_unit_square as create_test_model;
    use ndarray::array;
    use ndarray::ArrayView2;

    fn assert_batch_neighbor_fill<F>(
        model: &EpistemicNearestNeighbors,
        paramss: Vec<ENNParams>,
        mut run: F,
    ) where
        F: FnMut(
            &EpistemicNearestNeighbors,
            &ArrayView2<f64>,
            &[ENNParams],
            &PosteriorFlags,
            &mut Array3<f64>,
            &mut Array3<f64>,
        ) -> Result<(), ENNError>,
    {
        let flags = PosteriorFlags::new();
        let query = array![[0.5, 0.5]];
        let (bs, np) = (query.nrows(), paramss.len());
        let mut mu_all = Array3::zeros((np, bs, model.num_metrics()));
        let mut se_all = Array3::zeros((np, bs, model.num_metrics()));
        assert!(run(
            model,
            &query.view(),
            &paramss,
            &flags,
            &mut mu_all,
            &mut se_all
        )
        .is_ok());
        assert_eq!(mu_all.shape(), &[np, bs, 1]);
        assert_eq!(se_all.shape(), &[np, bs, 1]);
    }

    #[test]
    fn test_posterior_computation() {
        let model = create_test_model();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let flags = PosteriorFlags::new();
        let query = array![[0.5, 0.5]];

        let result = model.posterior(&query.view(), &params, &flags);
        assert!(result.is_ok());

        let posterior = result.unwrap();
        assert_eq!(posterior.mu.len(), 1);
        assert!(posterior.mu[[0, 0]] > 0.0 && posterior.mu[[0, 0]] < 2.0);
        assert!(posterior.se[[0, 0]] > 0.0);
    }

    #[test]
    fn test_batch_posterior() {
        let model = create_test_model();
        let params1 = ENNParams::new(2, 1.0, 0.1).unwrap();
        let params2 = ENNParams::new(2, 2.0, 0.2).unwrap();
        let paramss = vec![params1, params2];
        let flags = PosteriorFlags::new();
        let query = array![[0.5, 0.5]];

        let result = model.batch_posterior(&query.view(), &paramss, &flags);
        assert!(result.is_ok());

        let posterior = result.unwrap();
        assert_eq!(posterior.mu.shape(), &[2, 1, 1]);
    }

    #[test]
    fn test_posterior_function_draw() {
        let model = create_test_model();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let flags = PosteriorFlags::new();
        let query = array![[0.5, 0.5]];
        let seeds = vec![1i64, 2, 3];

        let result = model.posterior_function_draw(&query.view(), &params, &seeds, &flags);
        assert!(result.is_ok());

        let (draws, idx) = result.unwrap();
        assert_eq!(draws.shape(), &[3, 1, 1]);
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn test_empty_posterior_internals() {
        let model = create_test_model();
        let internals = empty_posterior_internals(&model, 5);

        assert_eq!(internals.idx.len(), 5);
        assert!(internals.idx.iter().all(|v| v.is_empty()));
        assert_eq!(internals.mu.shape(), &[5, 1]);
        assert_eq!(internals.se.shape(), &[5, 1]);
    }

    #[test]
    fn test_compute_posterior_internals() {
        let model = create_test_model();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let flags = PosteriorFlags::new();
        let query = array![[0.5, 0.5]];

        let result = compute_posterior_internals(&model, &query.view(), &params, &flags);
        assert!(result.is_ok());

        let internals = result.unwrap();
        assert_eq!(internals.mu.nrows(), 1);
        assert_eq!(internals.mu.ncols(), 1);
    }

    #[test]
    fn test_compute_posterior_internals_empty_model() {
        let train_x = array![[0.0, 0.0]];
        let train_y = array![[0.0]];
        let model =
            EpistemicNearestNeighbors::new(train_x, train_y, None, false, IndexDriver::Exact)
                .unwrap();

        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let flags = PosteriorFlags::new();
        let query = array![[100.0, 100.0]];

        let result = compute_posterior_internals(&model, &query.view(), &params, &flags);
        assert!(result.is_ok());
    }

    #[test]
    fn test_batch_posterior_empty_params() {
        let model = create_test_model();
        let flags = PosteriorFlags::new();
        let query = array![[0.5, 0.5]];
        let paramss: Vec<ENNParams> = vec![];

        let result = model.batch_posterior(&query.view(), &paramss, &flags);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("paramss must be non-empty"));
    }

    #[test]
    fn test_get_neighbor_data() {
        let model = create_test_model();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let query = array![[0.5, 0.5]];

        let result = get_neighbor_data(&model, &query.view(), &params, false);
        assert!(result.is_ok());
        assert!(result.unwrap().is_some());
    }

    #[test]
    fn test_compute_weighted_stats_impl() {
        let model = create_test_model();
        let dist2s = array![[0.1, 0.2]];
        let y_neighbors = array![[0.0], [1.0]];
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();

        let result = compute_weighted_stats_impl(
            &dist2s.view(),
            &y_neighbors.view(),
            None,
            &params,
            false,
            &model.y_scale().view(),
        );
        assert!(result.is_ok());

        let stats = result.unwrap();
        assert_eq!(stats.mu.shape(), &[1, 1]);
        assert_eq!(stats.se.shape(), &[1, 1]);
    }

    #[test]
    fn test_draw_from_internals_k0() {
        let model = create_test_model();
        let internals = empty_posterior_internals(&model, 3);
        let seeds = vec![1i64, 2];

        let result = draw_from_internals(&model, &internals, &seeds);
        assert!(result.is_ok());

        let draws = result.unwrap();
        assert_eq!(draws.shape(), &[2, 3, 1]);
    }

    #[test]
    fn test_draw_from_internals_with_neighbors() {
        let model = create_test_model();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let flags = PosteriorFlags::new();
        let query = array![[0.5, 0.5]];

        let internals =
            compute_posterior_internals(&model, &query.view(), &params, &flags).unwrap();
        let seeds = vec![1i64, 2];

        let result = draw_from_internals(&model, &internals, &seeds);
        assert!(result.is_ok());

        let draws = result.unwrap();
        assert_eq!(draws.shape(), &[2, 1, 1]);
        let one = draw_from_internals(&model, &internals, &[7i64]).unwrap();
        assert!((one[[0, 0, 0]] - 0.223_806_179_572_216_43).abs() < 1e-12);
    }

    #[test]
    fn test_compute_weighted_posterior() {
        let model = create_test_model();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let query = array![[0.5, 0.5]];

        let neighbor_data = get_neighbor_data(&model, &query.view(), &params, false)
            .unwrap()
            .unwrap();

        let data = WeightedPosteriorData {
            dist2s: &neighbor_data.dist2s.view(),
            idx: &neighbor_data.idx,
            y_neighbors: &neighbor_data.y_neighbors.view(),
            params: &params,
            observation_noise: false,
            yvar_neighbors_override: None,
        };

        let result = compute_weighted_posterior(&model, data, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_get_neighbor_data_exclude_nearest() {
        let model = create_test_model();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let query = array![[0.5, 0.5]];

        let result = get_neighbor_data(&model, &query.view(), &params, true);
        assert!(result.is_ok());
    }

    #[test]
    fn test_compute_weighted_stats_with_noise() {
        let model = create_test_model();
        let dist2s = array![[0.1, 0.2]];
        let y_neighbors = array![[0.0], [1.0]];
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();

        let result = compute_weighted_stats_impl(
            &dist2s.view(),
            &y_neighbors.view(),
            None,
            &params,
            true,
            &model.y_scale().view(),
        );
        assert!(result.is_ok());
    }

    // Tests for helper functions to achieve 100% coverage

    #[test]
    fn test_compute_batch_with_shared_neighbors_direct() {
        let model = create_test_model();
        let params1 = ENNParams::new(2, 1.0, 0.1).unwrap();
        let params2 = ENNParams::new(2, 2.0, 0.2).unwrap();
        let paramss = vec![params1, params2];
        assert_batch_neighbor_fill(&model, paramss, |m, q, p, f, mu, se| {
            compute_batch_with_shared_neighbors(m, q, p, f, mu, se)
        });
    }

    #[test]
    fn test_compute_batch_separate_neighbors_direct() {
        let model = create_test_model();
        let params1 = ENNParams::new(2, 1.0, 0.1).unwrap();
        let params2 = ENNParams::new(3, 2.0, 0.2).unwrap(); // Different k values
        let paramss = vec![params1, params2];
        assert_batch_neighbor_fill(&model, paramss, |m, q, p, f, mu, se| {
            compute_batch_separate_neighbors(m, q, p, f, mu, se)
        });
    }

    #[test]
    fn test_assign_posterior_results_direct() {
        let model = create_test_model();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let flags = PosteriorFlags::new();
        let query = array![[0.5, 0.5]];

        let internals =
            compute_posterior_internals(&model, &query.view(), &params, &flags).unwrap();

        let mut mu_all = Array3::zeros((3, 1, 1));
        let mut se_all = Array3::zeros((3, 1, 1));

        // Test assigning to different indices
        assign_posterior_results(&internals, &mut mu_all, &mut se_all, 0);
        assign_posterior_results(&internals, &mut mu_all, &mut se_all, 1);
        assign_posterior_results(&internals, &mut mu_all, &mut se_all, 2);

        // Verify the assignments were made (values should be non-zero from the computation)
        assert!(mu_all[[0, 0, 0]].is_finite());
        assert!(se_all[[0, 0, 0]].is_finite());
    }

    #[test]
    fn test_conditional_posterior_empty_whatif_delegates_to_posterior() {
        let model = create_test_model();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let flags = PosteriorFlags::new();
        let query = array![[0.5, 0.5]];
        let x_whatif: Array2<f64> = Array2::zeros((0, 2));
        let y_whatif: Array2<f64> = Array2::zeros((0, 1));

        let post = model.posterior(&query.view(), &params, &flags).unwrap();
        let cond = compute_conditional_posterior_internals(
            &model,
            &query.view(),
            &x_whatif.view(),
            &y_whatif.view(),
            &params,
            &flags,
        )
        .unwrap();

        assert_eq!(post.mu[[0, 0]], cond.mu[[0, 0]]);
        assert_eq!(post.se[[0, 0]], cond.se[[0, 0]]);
    }

    #[test]
    fn test_conditional_posterior_with_whatif() {
        let model = create_test_model();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let flags = PosteriorFlags::new();
        let query = array![[0.5, 0.5]];
        let x_whatif = array![[0.5, 0.5]];
        let y_whatif = array![[2.0]];

        let internals = compute_conditional_posterior_internals(
            &model,
            &query.view(),
            &x_whatif.view(),
            &y_whatif.view(),
            &params,
            &flags,
        )
        .unwrap();

        assert_eq!(internals.mu.nrows(), 1);
        assert_eq!(internals.mu.ncols(), 1);
        assert!(internals.mu[[0, 0]].is_finite());
        assert!(internals.se[[0, 0]].is_finite());
    }

    #[test]
    fn test_conditional_posterior_function_draw() {
        let model = create_test_model();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let flags = PosteriorFlags::new();
        let query = array![[0.5, 0.5]];
        let x_whatif = array![[0.5, 0.5]];
        let y_whatif = array![[1.5]];
        let seeds = vec![1i64, 2, 3];

        let internals = compute_conditional_posterior_internals(
            &model,
            &query.view(),
            &x_whatif.view(),
            &y_whatif.view(),
            &params,
            &flags,
        )
        .unwrap();
        let result = draw_from_internals(&model, &internals, &seeds);

        assert!(result.is_ok());
        let draws = result.unwrap();
        assert_eq!(draws.shape(), &[3, 1, 1]);
    }

    #[test]
    fn test_conditional_posterior_exclude_nearest() {
        let model = create_test_model();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let flags = PosteriorFlags::new().with_exclude_nearest(true);
        let query = array![[0.5, 0.5]];
        let x_whatif = array![[0.5, 0.5], [0.6, 0.6]];
        let y_whatif = array![[1.5], [2.0]];

        let internals = compute_conditional_posterior_internals(
            &model,
            &query.view(),
            &x_whatif.view(),
            &y_whatif.view(),
            &params,
            &flags,
        )
        .unwrap();
        assert!(internals.mu[[0, 0]].is_finite());
    }

    #[test]
    fn test_conditional_posterior_scaled_model() {
        let train_x = array![[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]];
        let train_y = array![[0.0], [1.0], [2.0]];
        let model =
            EpistemicNearestNeighbors::new(train_x, train_y, None, true, IndexDriver::Exact)
                .unwrap();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let flags = PosteriorFlags::new();
        let query = array![[1.0, 1.0]];
        let x_whatif = array![[3.0, 3.0]];
        let y_whatif = array![[1.5]];

        let internals = compute_conditional_posterior_internals(
            &model,
            &query.view(),
            &x_whatif.view(),
            &y_whatif.view(),
            &params,
            &flags,
        )
        .unwrap();
        assert!(internals.mu[[0, 0]].is_finite());
    }

    #[test]
    fn test_conditional_scale_single_row() {
        let train_x = array![[0.0, 0.0]];
        let train_y = array![[1.0]];
        let model =
            EpistemicNearestNeighbors::new(train_x, train_y, None, false, IndexDriver::Exact)
                .unwrap();
        let params = ENNParams::new(1, 1.0, 0.1).unwrap();
        let flags = PosteriorFlags::new();
        let query = array![[0.0, 0.0]];
        let x_whatif = array![[0.0, 0.0]];
        let y_whatif = array![[2.0]];

        let internals = compute_conditional_posterior_internals(
            &model,
            &query.view(),
            &x_whatif.view(),
            &y_whatif.view(),
            &params,
            &flags,
        )
        .unwrap();
        assert!(internals.mu[[0, 0]].is_finite());
    }

    #[test]
    fn test_compute_scale_for_conditional_branches() {
        use ndarray::Array2;

        let train = array![[1.0f64]];
        let whatif_empty: Array2<f64> = Array2::zeros((0, 1));
        let s0 = compute_scale_for_conditional(&train.view(), &whatif_empty.view());
        assert_eq!(s0.len(), 1);
        assert!((s0[0] - 1.0).abs() < 1e-12);

        let train_c = array![[1.0, 2.0], [1.0, 3.0]];
        let whatif_c = array![[1.0, 0.0]];
        let s1 = compute_scale_for_conditional(&train_c.view(), &whatif_c.view());
        assert_eq!(s1.len(), 2);
        assert!((s1[0] - 1.0).abs() < 1e-9, "constant column uses scale=1");
        assert!(s1[1] > 0.0);
    }

    #[test]
    fn test_conditional_posterior_y_whatif_shape_error_reports_y_whatif_shape() {
        let model = create_test_model();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let flags = PosteriorFlags::new();
        let query = array![[0.5, 0.5]];
        let x_whatif = array![[0.5, 0.5]];
        // y_whatif has wrong number of columns (2 instead of 1)
        let y_whatif = array![[1.0, 2.0]];

        let result = compute_conditional_posterior_internals(
            &model,
            &query.view(),
            &x_whatif.view(),
            &y_whatif.view(),
            &params,
            &flags,
        );
        let err = result.unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("Invalid shape"),
            "expected InvalidShape, got: {}",
            msg
        );
        assert!(
            msg.contains("[1, 2]"),
            "error should report y_whatif shape [1, 2], got: {}",
            msg
        );
    }

    #[test]
    fn test_conditional_posterior_nan_in_x_whatif_returns_error_not_panic() {
        let model = create_test_model();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let flags = PosteriorFlags::new();
        let query = array![[0.5, 0.5]];
        let x_whatif = array![[f64::NAN, 0.5]];
        let y_whatif = array![[1.0]];

        let result = compute_conditional_posterior_internals(
            &model,
            &query.view(),
            &x_whatif.view(),
            &y_whatif.view(),
            &params,
            &flags,
        );
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("NaN") || err.to_string().contains("invalid"),
            "expected error about NaN/invalid values, got: {}",
            err
        );
    }

    // Tests for empty-query cases (Bug fixes for panic issues)

    #[test]
    fn test_empty_query_posterior_with_yvar_no_panic() {
        // Bug fix: Empty query with train_yvar should not panic
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let train_y = array![[0.0], [1.0], [1.0], [2.0]];
        let train_yvar = array![[0.1], [0.1], [0.1], [0.1]];

        let model = EpistemicNearestNeighbors::new(
            train_x,
            train_y,
            Some(train_yvar),
            false,
            IndexDriver::Exact,
        )
        .unwrap();

        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let flags = PosteriorFlags::new();
        let empty_query: Array2<f64> = Array2::zeros((0, 2));

        let result = model.posterior(&empty_query.view(), &params, &flags);
        assert!(
            result.is_ok(),
            "Empty query posterior with yvar should not panic"
        );

        let posterior = result.unwrap();
        assert_eq!(posterior.mu.shape()[0], 0);
        assert_eq!(posterior.se.shape()[0], 0);
    }

    #[test]
    fn test_empty_query_conditional_posterior_no_panic() {
        // Bug fix: Empty query conditional posterior should not panic
        let model = create_test_model();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let flags = PosteriorFlags::new();

        let empty_query: Array2<f64> = Array2::zeros((0, 2));
        let x_whatif = array![[0.5, 0.5]];
        let y_whatif = array![[1.5]];

        let result = model.conditional_posterior(
            &x_whatif.view(),
            &y_whatif.view(),
            &empty_query.view(),
            &params,
            &flags,
        );
        assert!(
            result.is_ok(),
            "Empty query conditional posterior should not panic"
        );

        let posterior = result.unwrap();
        assert_eq!(posterior.mu.shape()[0], 0);
        assert_eq!(posterior.se.shape()[0], 0);
    }

    #[test]
    fn test_empty_query_conditional_posterior_with_yvar_no_panic() {
        // Bug fix: Empty query conditional posterior with train_yvar should not panic
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let train_y = array![[0.0], [1.0], [1.0], [2.0]];
        let train_yvar = array![[0.1], [0.1], [0.1], [0.1]];

        let model = EpistemicNearestNeighbors::new(
            train_x,
            train_y,
            Some(train_yvar),
            false,
            IndexDriver::Exact,
        )
        .unwrap();

        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let flags = PosteriorFlags::new();

        let empty_query: Array2<f64> = Array2::zeros((0, 2));
        let x_whatif = array![[0.5, 0.5]];
        let y_whatif = array![[1.5]];

        let result = model.conditional_posterior(
            &x_whatif.view(),
            &y_whatif.view(),
            &empty_query.view(),
            &params,
            &flags,
        );
        assert!(
            result.is_ok(),
            "Empty query conditional posterior with yvar should not panic"
        );

        let posterior = result.unwrap();
        assert_eq!(posterior.mu.shape()[0], 0);
        assert_eq!(posterior.se.shape()[0], 0);
    }

    #[test]
    fn test_compute_weighted_posterior_empty_idx() {
        // Direct test for compute_weighted_posterior with empty idx
        let model = create_test_model();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();

        let empty_dist2s: Array2<f64> = Array2::zeros((0, 2));
        let empty_y_neighbors: Array2<f64> = Array2::zeros((0, 1));
        let empty_idx: Vec<Vec<usize>> = vec![];

        let data = WeightedPosteriorData {
            dist2s: &empty_dist2s.view(),
            idx: &empty_idx,
            y_neighbors: &empty_y_neighbors.view(),
            params: &params,
            observation_noise: false,
            yvar_neighbors_override: None,
        };

        let result = compute_weighted_posterior(&model, data, None);
        assert!(
            result.is_ok(),
            "compute_weighted_posterior with empty idx should not panic"
        );
    }

    #[test]
    fn test_get_conditional_neighbor_data_empty_batch() {
        // Direct test for get_conditional_neighbor_data with empty batch
        let model = create_test_model();
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let flags = PosteriorFlags::new();

        let empty_x: Array2<f64> = Array2::zeros((0, 2));
        let x_whatif = array![[0.5, 0.5]];
        let y_whatif = array![[1.5]];

        let result = get_conditional_neighbor_data(
            &model,
            &empty_x.view(),
            &x_whatif.view(),
            &y_whatif.view(),
            &params,
            &flags,
        );
        assert!(
            result.is_ok(),
            "get_conditional_neighbor_data with empty batch should not panic"
        );
        assert!(result.unwrap().is_none());
    }
}
