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
) -> Result<AffineCalibrator, ENNError> {
    let n = model.len();
    let metrics = model.num_outputs();
    if n < 2 || num_fit_samples == 0 {
        return Ok(AffineCalibrator::identity(metrics));
    }
    let p = num_fit_samples.min(n);
    let mut order: Vec<usize> = (0..n).collect();
    for i in 0..p {
        let j = i + rng.gen_range(0..(n - i));
        order.swap(i, j);
    }
    order.truncate(p);
    let (x, y, _) = model.train_rows_at(&order)?;
    let flags = PosteriorFlags::new().with_exclude_nearest(true);
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
) -> (Array2<f64>, Array2<f64>) {
    let mut se_p = se;
    for j in 0..se_p.ncols() {
        for i in 0..se_p.nrows() {
            se_p[[i, j]] *= cal.c[j];
        }
    }
    let mu_p = cal.map_mu(&mu, Some(y_bounds), Some(&se_p));
    (mu_p, se_p)
}

pub fn apply_draws(cal: &AffineCalibrator, draws: &Array3<f64>, mu: &Array2<f64>, se: &Array2<f64>, y_bounds: &Array2<f64>) -> Array3<f64> {
    let mut out = draws.clone();
    for s in 0..draws.shape()[0] {
        for i in 0..draws.shape()[1] {
            for j in 0..draws.shape()[2] {
                out[[s, i, j]] = cal.a[j] + cal.b[j] * mu[[i, j]] + cal.c[j] * (draws[[s, i, j]] - mu[[i, j]]);
            }
        }
    }
    for s in 0..out.shape()[0] {
        let mut slice = out.slice_mut(ndarray::s![s, .., ..]);
        let owned = slice.to_owned();
        let mut projected = owned;
        crate::calibration::project_mu(&mut projected, Some(y_bounds), Some(se));
        slice.assign(&projected);
    }
    out
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
        coeff_slice(&cal.a)?,
        coeff_slice(&cal.b)?,
        coeff_slice(&cal.c)?,
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
        draws = map_function_draws(cal, &draws, &mu, &se, model.y_bounds())?;
    }
    Ok((to_batch_metric_sample(draws), idx))
}

fn map_function_draws(
    cal: &AffineCalibrator,
    draws: &Array3<f64>,
    mu: &Array2<f64>,
    se: &Array2<f64>,
    y_bounds: &Array2<f64>,
) -> Result<Array3<f64>, ENNError> {
    let (n_seed, n_batch, n_met) = draws.dim();
    let rows = n_batch * n_seed;
    let mut flat = Array2::zeros((rows, n_met));
    let mut mu_flat = Array2::zeros((rows, n_met));
    let mut se_flat = Array2::zeros((rows, n_met));
    let mut row = 0;
    for b in 0..n_batch {
        for s in 0..n_seed {
            for m in 0..n_met {
                flat[[row, m]] = draws[[s, b, m]];
                mu_flat[[row, m]] = mu[[b, m]];
                se_flat[[row, m]] = se[[b, m]];
            }
            row += 1;
        }
    }
    let mapped = crate::calibration::map_draws_with(
        coeff_slice(&cal.a)?,
        coeff_slice(&cal.b)?,
        coeff_slice(&cal.c)?,
        flat.view(),
        mu_flat.view(),
        Some(y_bounds),
        Some(&se_flat),
    )?;
    let mut out = draws.clone();
    row = 0;
    for b in 0..n_batch {
        for s in 0..n_seed {
            for m in 0..n_met {
                out[[s, b, m]] = mapped[[row, m]];
            }
            row += 1;
        }
    }
    Ok(out)
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

fn coeff_slice(v: &ndarray::Array1<f64>) -> Result<&[f64], ENNError> {
    v.as_slice()
        .ok_or_else(|| ENNError::InvalidParameter("calibrator coefficient is not contiguous".to_string()))
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
    let (_mu_p, se_p) = apply_prediction(cal, mu.clone(), se, model.y_bounds());
    Ok(apply_draws(cal, &draws, &mu, &se_p, model.y_bounds()))
}
