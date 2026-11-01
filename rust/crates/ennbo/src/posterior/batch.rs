use ndarray::{Array2, Array3, ArrayView2, Axis};

use super::light::idx_nested_to_array2;
use super::neighbor::get_neighbor_data;
use super::{
    compute_posterior_internals, compute_weighted_posterior, empty_posterior_internals,
    WeightedPosteriorData,
};
use crate::draw::DrawInternals;
use crate::error::ENNError;
use crate::model::EpistemicNearestNeighbors;
use crate::params::{ENNNormal, ENNParams, PosteriorFlags};

pub(super) fn batch_posterior_unblended(
    model: &EpistemicNearestNeighbors,
    x: &ArrayView2<f64>,
    paramss: &[ENNParams],
    flags: &PosteriorFlags,
) -> Result<ENNNormal, ENNError> {
    if paramss.is_empty() {
        return Err(ENNError::InvalidParameter(
            "paramss must be non-empty".to_string(),
        ));
    }

    let k_values: std::collections::HashSet<i32> =
        paramss.iter().map(|p| p.k_num_neighbors).collect();
    let batch_size = x.nrows();
    let num_params = paramss.len();

    let mut mu_all = Array3::zeros((num_params, batch_size, model.num_metrics()));
    let mut se_all = Array3::zeros((num_params, batch_size, model.num_metrics()));
    let mut se_epi_all = Array3::zeros((num_params, batch_size, model.num_metrics()));
    let mut se_ale_all = Array3::zeros((num_params, batch_size, model.num_metrics()));

    let idx = if k_values.len() == 1 && model.num_obs() > 0 {
        compute_batch_with_shared_neighbors(
            model,
            x,
            paramss,
            flags,
            &mut mu_all,
            &mut se_all,
            &mut se_epi_all,
            &mut se_ale_all,
        )?
    } else {
        compute_batch_separate_neighbors(
            model,
            x,
            paramss,
            flags,
            &mut mu_all,
            &mut se_all,
            &mut se_epi_all,
            &mut se_ale_all,
        )?;
        None
    };

    Ok(ENNNormal::new(
        mu_all.into_dyn(),
        se_all.into_dyn(),
        se_epi_all.into_dyn(),
        se_ale_all.into_dyn(),
        idx,
    ))
}

#[allow(clippy::too_many_arguments)]
pub(super) fn compute_batch_with_shared_neighbors(
    model: &EpistemicNearestNeighbors,
    x: &ArrayView2<f64>,
    paramss: &[ENNParams],
    flags: &PosteriorFlags,
    mu_all: &mut Array3<f64>,
    se_all: &mut Array3<f64>,
    se_epi_all: &mut Array3<f64>,
    se_ale_all: &mut Array3<f64>,
) -> Result<Option<Array2<i64>>, ENNError> {
    let neighbor_data = get_neighbor_data(model, x, &paramss[0], flags.exclude_nearest)?;

    if let Some(data) = neighbor_data {
        let idx = idx_nested_to_array2(&data.idx);
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
            assign_posterior_results(&internals, mu_all, se_all, se_epi_all, se_ale_all, i);
        }
        Ok(Some(idx))
    } else {
        let batch_size = x.nrows();
        let internals = empty_posterior_internals(model, batch_size);
        for i in 0..paramss.len() {
            assign_posterior_results(&internals, mu_all, se_all, se_epi_all, se_ale_all, i);
        }
        Ok(Some(idx_nested_to_array2(&internals.idx)))
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn compute_batch_separate_neighbors(
    model: &EpistemicNearestNeighbors,
    x: &ArrayView2<f64>,
    paramss: &[ENNParams],
    flags: &PosteriorFlags,
    mu_all: &mut Array3<f64>,
    se_all: &mut Array3<f64>,
    se_epi_all: &mut Array3<f64>,
    se_ale_all: &mut Array3<f64>,
) -> Result<(), ENNError> {
    for (i, params) in paramss.iter().enumerate() {
        let internals = compute_posterior_internals(model, x, params, flags)?;
        assign_posterior_results(&internals, mu_all, se_all, se_epi_all, se_ale_all, i);
    }
    Ok(())
}

pub(super) fn assign_posterior_results(
    internals: &DrawInternals,
    mu_all: &mut Array3<f64>,
    se_all: &mut Array3<f64>,
    se_epi_all: &mut Array3<f64>,
    se_ale_all: &mut Array3<f64>,
    index: usize,
) {
    let slice = ndarray::Slice::from(index..index + 1);
    mu_all
        .slice_axis_mut(Axis(0), slice)
        .assign(&internals.mu.slice_axis(Axis(0), ndarray::Slice::from(..)));
    se_all
        .slice_axis_mut(Axis(0), slice)
        .assign(&internals.se.slice_axis(Axis(0), ndarray::Slice::from(..)));
    se_epi_all.slice_axis_mut(Axis(0), slice).assign(
        &internals
            .se_epi
            .slice_axis(Axis(0), ndarray::Slice::from(..)),
    );
    se_ale_all.slice_axis_mut(Axis(0), slice).assign(
        &internals
            .se_ale
            .slice_axis(Axis(0), ndarray::Slice::from(..)),
    );
}
