use ndarray::{Array2, Array3};
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
    let uncalibrated = params.clone().with_calibrator(None);
    let post = model.posterior(&x.view(), &uncalibrated, &flags)?;
    let mu = to_2d(&post.mu)?;
    let se = to_2d(&post.se)?;
    AffineCalibrator::fit(mu.view(), y.view(), Some(se.view()))
}

pub(crate) struct NormalColumns {
    pub(crate) mu: Array2<f64>,
    pub(crate) se: Array2<f64>,
    pub(crate) se_epi: Array2<f64>,
    pub(crate) se_ale: Array2<f64>,
}

pub(crate) fn apply_normal_columns(
    cal: &AffineCalibrator,
    normal: NormalColumns,
    y_bounds: &Array2<f64>,
) -> Result<NormalColumns, ENNError> {
    let c = crate::calibration::coeff_slice(&cal.c)?;
    let se = crate::calibration::scale_matrix_columns(&normal.se, c)?;
    let mu = cal.map_mu(&normal.mu, Some(y_bounds), Some(&se))?;
    Ok(NormalColumns {
        mu,
        se,
        se_epi: crate::calibration::scale_matrix_columns(&normal.se_epi, c)?,
        se_ale: crate::calibration::scale_matrix_columns(&normal.se_ale, c)?,
    })
}

pub(crate) fn calibrate_normal(
    cal: &AffineCalibrator,
    out: &mut crate::params::ENNNormal,
    y_bounds: &Array2<f64>,
) -> Result<(), ENNError> {
    let scaled = apply_normal_columns(
        cal,
        NormalColumns {
            mu: to_2d(&out.mu)?,
            se: to_2d(&out.se)?,
            se_epi: to_2d(&out.se_epi)?,
            se_ale: to_2d(&out.se_ale)?,
        },
        y_bounds,
    )?;
    out.mu = scaled.mu.into_dyn();
    out.se = scaled.se.into_dyn();
    out.se_epi = scaled.se_epi.into_dyn();
    out.se_ale = scaled.se_ale.into_dyn();
    Ok(())
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

pub(crate) fn calibrate_natural_draws(
    cal: &AffineCalibrator,
    draws: Array3<f64>,
    uncalibrated: &crate::params::ENNNormal,
    y_bounds: &Array2<f64>,
) -> Result<Array3<f64>, ENNError> {
    let mu = to_2d(&uncalibrated.mu)?;
    let se = to_2d(&uncalibrated.se)?;
    apply_draws(cal, &draws, &mu, &se, y_bounds)
}

fn to_2d(arr: &ndarray::ArrayD<f64>) -> Result<Array2<f64>, ENNError> {
    arr.clone()
        .into_dimensionality()
        .map_err(|e| ENNError::InvalidParameter(format!("Shape error: {e}")))
}
