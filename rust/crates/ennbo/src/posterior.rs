mod batch;
mod draw_compute;
mod dual_api;
mod light;
mod neighbor;
pub mod neighbor_dist;
pub(crate) use neighbor::index_search;

use ndarray::{Array1, Array2, Array3, ArrayView1, ArrayView2};

use self::batch::batch_posterior_warped;
#[cfg(test)]
use self::batch::{
    assign_posterior_results, compute_batch_separate_neighbors, compute_batch_with_shared_neighbors,
};
use self::draw_compute::draw_from_internals;
use self::neighbor::{get_conditional_neighbor_data, get_neighbor_data};
use crate::draw::DrawInternals;
use crate::error::{ENNError, EPS_VAR};
use crate::model::EpistemicNearestNeighbors;
use crate::params::{ENNNormal, ENNParams, PosteriorFlags};
use crate::stats::WeightedStats;
use crate::traits::PosteriorComputation;

pub(crate) fn se_from_variance_components(
    epistemic_var: f64,
    aleatoric_var: f64,
    y_scale: f64,
) -> (f64, f64, f64) {
    let sum = epistemic_var + aleatoric_var;
    if sum < EPS_VAR {
        let se = EPS_VAR.sqrt() * y_scale;
        (se, se, 0.0)
    } else {
        let se = sum.sqrt() * y_scale;
        let se_epi = epistemic_var.sqrt() * y_scale;
        let se_ale = aleatoric_var.sqrt() * y_scale;
        (se, se_epi, se_ale)
    }
}

impl PosteriorComputation for EpistemicNearestNeighbors {
    fn posterior(
        &self,
        x: &ArrayView2<f64>,
        params: &ENNParams,
        flags: &PosteriorFlags,
    ) -> Result<ENNNormal, ENNError> {
        self.posterior_warped(x, params, flags)
    }

    fn batch_posterior(
        &self,
        x: &ArrayView2<f64>,
        paramss: &[ENNParams],
        flags: &PosteriorFlags,
    ) -> Result<ENNNormal, ENNError> {
        batch_posterior_warped(self, x, paramss, flags)
    }

    fn posterior_function_draw(
        &self,
        x: &ArrayView2<f64>,
        params: &ENNParams,
        function_seeds: &[i64],
        flags: &PosteriorFlags,
    ) -> Result<(Array3<f64>, Vec<Vec<usize>>), ENNError> {
        EpistemicNearestNeighbors::posterior_function_draw(self, x, params, function_seeds, flags)
    }

    fn conditional_posterior(
        &self,
        x_whatif: &ArrayView2<f64>,
        y_whatif: &ArrayView2<f64>,
        x: &ArrayView2<f64>,
        params: &ENNParams,
        flags: &PosteriorFlags,
    ) -> Result<ENNNormal, ENNError> {
        self.conditional_posterior_warped(x_whatif, y_whatif, x, params, flags)
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
        if x_whatif.nrows() == 0 {
            return EpistemicNearestNeighbors::posterior_function_draw(
                self,
                x,
                params,
                function_seeds,
                flags,
            );
        }
        let (y_z, _) = self.warp_observations(y_whatif, None)?;
        let internals =
            compute_conditional_posterior_internals(self, x, x_whatif, &y_z.view(), params, flags)?;
        let draws = draw_from_internals(self, &internals, function_seeds)?;
        let draws = self.calibrate_published_draws(draws, params, |bare| {
            self.conditional_posterior(x_whatif, y_whatif, x, bare, flags)
        })?;
        Ok((draws, internals.idx))
    }
}

pub struct WeightedPosteriorData<'a> {
    pub dist2s: &'a ArrayView2<'a, f64>,
    pub idx: &'a [Vec<usize>],
    pub y_neighbors: &'a ArrayView2<'a, f64>,
    pub params: &'a ENNParams,
    pub observation_noise: bool,
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

    let counts = neighbor_counts(model, data.idx);
    let yvar_neighbors: Option<Array2<f64>> = if let Some(ov) = data.yvar_neighbors_override {
        Some(ov.clone())
    } else if model.has_yvar() {
        let n_query = data.dist2s.nrows();

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
                    let yvar_row = model
                        .neighbor_yvar(neighbor_idx)
                        .expect("row_yvar")
                        .expect("yvar");
                    for m in 0..model.num_metrics() {
                        yvar_neighbors[[i * k + j, m]] = yvar_row[m];
                    }
                }
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
        counts.as_deref(),
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
        stats.se_epi,
        stats.se_ale,
    ))
}

fn neighbor_counts(model: &EpistemicNearestNeighbors, idx: &[Vec<usize>]) -> Option<Vec<f64>> {
    if !model.aggregates_duplicates() {
        return None;
    }
    let n_train = model.num_obs();
    Some(
        idx.iter()
            .flatten()
            .map(|&id| {
                if id < n_train {
                    model.neighbor_count(id)
                } else {
                    1.0
                }
            })
            .collect(),
    )
}

#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn stats_from_neighbor_weights(
    w: &Array2<f64>,
    y_neighbors: &ArrayView2<f64>,
    yvar_neighbors: Option<ArrayView2<f64>>,
    n_query: usize,
    k: usize,
    num_metrics: usize,
    observation_noise: bool,
    aleatoric_scale: f64,
    y_scale: &ArrayView1<f64>,
    y_scale_sq: &[f64],
) -> (
    Array3<f64>,
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
    Array2<f64>,
) {
    let yvar_ref = yvar_neighbors.as_ref();
    let mut w_normalized = Array3::zeros((n_query, k, num_metrics));
    let mut l2 = Array2::zeros((n_query, num_metrics));
    let mut mu = Array2::zeros((n_query, num_metrics));
    let mut se = Array2::zeros((n_query, num_metrics));
    let mut se_epi = Array2::zeros((n_query, num_metrics));
    let mut se_ale = Array2::zeros((n_query, num_metrics));

    for i in 0..n_query {
        for m in 0..num_metrics {
            let base_idx = i * k;
            let norm: f64 = (0..k).map(|j| w[[base_idx + j, m]]).sum();
            let inv_norm = 1.0 / norm;

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

            let (se_val, se_epi_val, se_ale_val) =
                se_from_variance_components(epistemic_var, aleatoric_var, y_scale[m]);
            se[[i, m]] = se_val;
            se_epi[[i, m]] = se_epi_val;
            se_ale[[i, m]] = se_ale_val;
        }
    }

    (w_normalized, l2, mu, se, se_epi, se_ale)
}

pub fn compute_weighted_stats_impl(
    dist2s: &ArrayView2<f64>,
    y_neighbors: &ArrayView2<f64>,
    yvar_neighbors: Option<ArrayView2<f64>>,
    counts: Option<&[f64]>,
    params: &ENNParams,
    observation_noise: bool,
    y_scale: &ArrayView1<f64>,
) -> Result<WeightedStats, ENNError> {
    let n_query = dist2s.nrows();
    let k = dist2s.ncols();
    let num_metrics = y_scale.len();

    let epistemic_scale = params.epistemic_variance_scale;
    let aleatoric_scale = params.aleatoric_variance_scale;
    let y_scale_sq: Vec<f64> = y_scale.iter().map(|&v| v * v).collect();

    let mut var_epi = Array2::zeros((n_query, k));
    for (i, mut row) in var_epi.rows_mut().into_iter().enumerate() {
        let dist_row = dist2s.row(i);
        for (j, v) in row.iter_mut().enumerate() {
            *v = epistemic_scale * dist_row[j];
        }
    }

    let mut w = Array2::zeros((n_query * k, num_metrics));
    let yvar_ref = yvar_neighbors.as_ref();

    for i in 0..n_query {
        let var_epi_row = var_epi.row(i);
        for j in 0..k {
            let var_epi_ij = var_epi_row[j];
            let idx = i * k + j;
            for m in 0..num_metrics {
                let count = counts.map_or(1.0, |c| c[idx]);
                let var_y = if let Some(yv) = yvar_ref {
                    yv[[idx, m]] / y_scale_sq[m] / count
                } else {
                    0.0
                };
                let var_total = EPS_VAR + var_epi_ij + aleatoric_scale / count + var_y;
                w[[idx, m]] = 1.0 / var_total;
            }
        }
    }

    let (w_normalized, l2, mu, se, se_epi, se_ale) = stats_from_neighbor_weights(
        &w,
        y_neighbors,
        yvar_neighbors,
        n_query,
        k,
        num_metrics,
        observation_noise,
        aleatoric_scale,
        y_scale,
        &y_scale_sq,
    );

    Ok(WeightedStats::new(w_normalized, l2, mu, se, se_epi, se_ale))
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

    crate::util::check_finite_query(x)?;

    if model.num_obs() == 0 {
        return Ok(empty_posterior_internals(model, x.nrows()));
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
        Ok(empty_posterior_internals(model, x.nrows()))
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
        let all_y_indices: Vec<usize> = (0..model.num_obs()).collect();
        let (_, train_y_all, _) = model.rows().train_rows_at(&all_y_indices)?;
        let y_scale_cond = compute_scale_for_conditional(&train_y_all.view(), y_whatif);
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
        Array2::ones((batch_size, model.num_metrics())),
        Array2::zeros((batch_size, model.num_metrics())),
    )
}

#[cfg(test)]
#[path = "posterior/tests_core.rs"]
mod tests_core;

#[cfg(test)]
#[path = "posterior/tests_draw_noise.rs"]
mod tests_draw_noise;

#[cfg(test)]
#[path = "posterior/tests_conditional.rs"]
mod tests_conditional;

#[cfg(test)]
#[path = "posterior/tests_y_bounds.rs"]
mod tests_y_bounds;

#[cfg(test)]
#[path = "posterior/tests_duplicates.rs"]
mod tests_duplicates;

#[cfg(test)]
#[path = "posterior/tests_prefix.rs"]
mod tests_prefix;
