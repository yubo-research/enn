//! Fit and apply an affine calibrator inside the TuRBO-ENN surrogate.

use ndarray::{Array2, Array3, ArrayView2};
use rand::Rng;

use crate::calibration::AffineCalibrator;
use crate::error::ENNError;
use crate::model::EpistemicNearestNeighbors;
use crate::params::{ENNParams, PosteriorFlags};

pub fn fit_calibrator<R: Rng>(
    model: &EpistemicNearestNeighbors,
    params: &ENNParams,
    num_fit_samples: usize,
    rng: &mut R,
    observation_noise: bool,
) -> Result<AffineCalibrator, ENNError> {
    let n = model.len();
    let metrics = model.num_outputs();
    if n < 2 || num_fit_samples == 0 {
        return Ok(AffineCalibrator::identity(metrics));
    }
    let order = crate::calibration::sample_prefix(n, num_fit_samples.min(n), rng);
    let (x, y, _) = model.train_rows_at(&order)?;
    let flags = PosteriorFlags::new()
        .with_exclude_nearest(true)
        .with_observation_noise(observation_noise);
    let post = model.posterior(&x.view(), params, &flags)?;
    let mu = to_2d(&post.mu)?;
    let se = to_2d(&post.se)?;
    AffineCalibrator::fit(mu.view(), y.view(), Some(se.view()))
}

pub fn apply_prediction(
    cal: &AffineCalibrator,
    mu: Array2<f64>,
    se: Array2<f64>,
    y_bounds: &Array2<f64>,
) -> Result<(Array2<f64>, Array2<f64>), ENNError> {
    let se_p =
        crate::calibration::scale_matrix_columns(&se, crate::calibration::coeff_slice(&cal.c)?)?;
    let mu_p = cal.map_mu(&mu, Some(y_bounds), Some(&se_p))?;
    Ok((mu_p, se_p))
}

pub fn apply_draws(
    cal: &AffineCalibrator,
    draws: &Array3<f64>,
    mu: &Array2<f64>,
    se: &Array2<f64>,
    y_bounds: &Array2<f64>,
) -> Result<Array3<f64>, ENNError> {
    let mut out = draws.clone();
    for seed in 0..draws.shape()[0] {
        let mapped = crate::calibration::map_draws_with(
            crate::calibration::coeff_slice(&cal.a)?,
            crate::calibration::coeff_slice(&cal.b)?,
            crate::calibration::coeff_slice(&cal.c)?,
            draws.slice(ndarray::s![seed, .., ..]),
            mu.view(),
            Some(y_bounds),
            Some(se),
        )?;
        out.slice_mut(ndarray::s![seed, .., ..]).assign(&mapped);
    }
    Ok(out)
}

pub fn calibrated_posterior(
    model: &EpistemicNearestNeighbors,
    x: &ArrayView2<f64>,
    params: &ENNParams,
    flags: &PosteriorFlags,
    cal: Option<&AffineCalibrator>,
) -> Result<crate::params::ENNNormal, ENNError> {
    let mut out = model.posterior(x, params, flags)?;
    let Some(cal) = cal else {
        return Ok(out);
    };
    let mu = to_2d(&out.mu)?;
    let se_epi = to_2d(&out.se_epi)?;
    let se_ale = to_2d(&out.se_ale)?;
    let (mu_p, se, epi, ale) = crate::calibration::apply_normal(
        crate::calibration::coeff_slice(&cal.a)?,
        crate::calibration::coeff_slice(&cal.b)?,
        crate::calibration::coeff_slice(&cal.c)?,
        mu.view(),
        se_epi.view(),
        se_ale.view(),
        Some(model.y_bounds()),
    )?;
    out.mu = mu_p.into_dyn();
    out.se = se.into_dyn();
    out.se_epi = epi.into_dyn();
    out.se_ale = ale.into_dyn();
    Ok(out)
}

pub fn calibrated_function_draw(
    model: &EpistemicNearestNeighbors,
    x: &ArrayView2<f64>,
    params: &ENNParams,
    function_seeds: &[i64],
    flags: &PosteriorFlags,
    cal: Option<&AffineCalibrator>,
) -> Result<(Array3<f64>, Vec<Vec<usize>>), ENNError> {
    let (mut draws, idx) = model.posterior_function_draw(x, params, function_seeds, flags)?;
    if let Some(cal) = cal {
        let post = model.posterior(x, params, flags)?;
        let mu = to_2d(&post.mu)?;
        let se = to_2d(&post.se)?;
        draws = apply_draws(cal, &draws, &mu, &se, model.y_bounds())?;
    }
    Ok((to_batch_metric_sample(draws), idx))
}

fn to_batch_metric_sample(draws: Array3<f64>) -> Array3<f64> {
    let (n_seed, n_batch, n_met) = draws.dim();
    let mut out = Array3::zeros((n_batch, n_met, n_seed));
    for s in 0..n_seed {
        for b in 0..n_batch {
            for m in 0..n_met {
                out[[b, m, s]] = draws[[s, b, m]];
            }
        }
    }
    out
}

fn to_2d(arr: &ndarray::ArrayD<f64>) -> Result<Array2<f64>, ENNError> {
    arr.clone()
        .into_dimensionality()
        .map_err(|e| ENNError::InvalidParameter(format!("Shape error: {e}")))
}

pub fn calibrated_sample(
    model: &EpistemicNearestNeighbors,
    params: &ENNParams,
    cal: &AffineCalibrator,
    x: &ArrayView2<f64>,
    num_samples: usize,
    rng: &mut dyn rand::RngCore,
) -> Result<Array3<f64>, ENNError> {
    let mut seed_bytes = [0u8; 8];
    rng.fill_bytes(&mut seed_bytes);
    let base = u64::from_le_bytes(seed_bytes) as i64;
    let seeds: Vec<i64> = (0..num_samples as i64).map(|i| base + i).collect();
    let flags = PosteriorFlags::new();
    let (draws, _) = model.posterior_function_draw(x, params, &seeds, &flags)?;
    let post = model.posterior(x, params, &flags)?;
    let mu = to_2d(&post.mu)?;
    let se = to_2d(&post.se)?;
    apply_draws(cal, &draws, &mu, &se, model.y_bounds())
}
