use ndarray::{Array2, Array3, ArrayView2};

pub(crate) fn transpose_num_samples_last(draws: Array3<f64>) -> Array3<f64> {
    let (samples, batch, metrics) = draws.dim();
    let mut out = Array3::zeros((batch, metrics, samples));
    for s in 0..samples {
        for b in 0..batch {
            for m in 0..metrics {
                out[[b, m, s]] = draws[[s, b, m]];
            }
        }
    }
    out
}

use super::draw_compute::draw_from_internals;
use super::light::{compute_posterior_light, idx_nested_to_array2};
use super::{compute_conditional_posterior_internals, compute_posterior_internals};
use crate::calibration::AffineCalibrator;
use crate::error::ENNError;
use crate::model::EpistemicNearestNeighbors;
use crate::params::{ENNNormal, ENNParams, PosteriorFlags};
use crate::traits::PosteriorComputation;
use crate::y_bounds::{inv_y, is_identity_bounds, naturalize_mu_se, warp_y};

impl EpistemicNearestNeighbors {
    pub(crate) fn posterior_warped(
        &self,
        x: &ArrayView2<f64>,
        params: &ENNParams,
        flags: &PosteriorFlags,
    ) -> Result<ENNNormal, ENNError> {
        crate::util::check_finite_query(x)?;
        let (mu, se, se_epi, se_ale, idx) = if !flags.observation_noise && !self.has_yvar() {
            compute_posterior_light(self, x, params, flags)?
        } else {
            let internals = compute_posterior_internals(self, x, params, flags)?;
            (
                internals.mu,
                internals.se,
                internals.se_epi,
                internals.se_ale,
                idx_nested_to_array2(&internals.idx),
            )
        };
        let mut out = ENNNormal::new(
            mu.into_dyn(),
            se.into_dyn(),
            se_epi.into_dyn(),
            se_ale.into_dyn(),
            Some(idx),
        );
        self.apply_normal_blend(&mut out, params)?;
        Ok(out)
    }

    fn apply_normal_blend(&self, out: &mut ENNNormal, params: &ENNParams) -> Result<(), ENNError> {
        match self.checked_blend(params)? {
            Some(blend) => blend.apply_normal(out),
            None => Ok(()),
        }
    }

    pub(super) fn apply_batch_blends(
        &self,
        out: &mut ENNNormal,
        paramss: &[ENNParams],
    ) -> Result<(), ENNError> {
        for (entry, params) in paramss.iter().enumerate() {
            if let Some(blend) = self.checked_blend(params)? {
                blend.apply_batch_entry(out, entry)?;
            }
        }
        Ok(())
    }

    pub(super) fn apply_draw_blend(
        &self,
        draws: &mut Array3<f64>,
        params: &ENNParams,
    ) -> Result<(), ENNError> {
        if let Some(blend) = self.checked_blend(params)? {
            blend.apply_draws(draws);
        }
        Ok(())
    }

    pub fn posterior(
        &self,
        x: &ArrayView2<f64>,
        params: &ENNParams,
        flags: &PosteriorFlags,
    ) -> Result<ENNNormal, ENNError> {
        let mut out = self.posterior_warped(x, params, flags)?;
        self.naturalize_enn_normal(&mut out)?;
        self.calibrate_normal(&mut out, params)?;
        Ok(out)
    }

    fn checked_calibrator<'a>(
        &self,
        params: &'a ENNParams,
    ) -> Result<Option<&'a AffineCalibrator>, ENNError> {
        let Some(cal) = params.calibrator.as_deref() else {
            return Ok(None);
        };
        if cal.a.len() != self.num_metrics() {
            return Err(ENNError::InvalidParameter(format!(
                "calibrator has {} outputs but the model has {}",
                cal.a.len(),
                self.num_metrics()
            )));
        }
        Ok(Some(cal))
    }

    fn calibrate_normal(&self, out: &mut ENNNormal, params: &ENNParams) -> Result<(), ENNError> {
        if let Some(cal) = self.checked_calibrator(params)? {
            crate::surrogate_affine::calibrate_normal(cal, out, self.y_bounds())?;
        }
        Ok(())
    }

    fn calibrate_batch(&self, out: &mut ENNNormal, paramss: &[ENNParams]) -> Result<(), ENNError> {
        for (entry, params) in paramss.iter().enumerate() {
            let Some(cal) = self.checked_calibrator(params)? else {
                continue;
            };
            let mut slice = out.batch_entry(entry);
            crate::surrogate_affine::calibrate_normal(cal, &mut slice, self.y_bounds())?;
            out.assign_batch_entry(entry, &slice);
        }
        Ok(())
    }

    pub(super) fn calibrate_published_draws(
        &self,
        mut draws: Array3<f64>,
        params: &ENNParams,
        uncalibrated_posterior: impl FnOnce(&ENNParams) -> Result<ENNNormal, ENNError>,
    ) -> Result<Array3<f64>, ENNError> {
        self.naturalize_draws_3d(&mut draws);
        if let Some(cal) = self.checked_calibrator(params)? {
            let post = uncalibrated_posterior(&params.clone().with_calibrator(None))?;
            draws = crate::surrogate_affine::calibrate_natural_draws(
                cal,
                draws,
                &post,
                self.y_bounds(),
            )?;
        }
        Ok(transpose_num_samples_last(draws))
    }

    pub(crate) fn posterior_function_draw_warped(
        &self,
        x: &ArrayView2<f64>,
        params: &ENNParams,
        function_seeds: &[i64],
        flags: &PosteriorFlags,
    ) -> Result<(Array3<f64>, Vec<Vec<usize>>), ENNError> {
        let internals = compute_posterior_internals(self, x, params, flags)?;
        let mut draws = draw_from_internals(self, &internals, function_seeds)?;
        self.apply_draw_blend(&mut draws, params)?;
        Ok((draws, internals.idx))
    }

    pub fn posterior_function_draw(
        &self,
        x: &ArrayView2<f64>,
        params: &ENNParams,
        function_seeds: &[i64],
        flags: &PosteriorFlags,
    ) -> Result<(Array3<f64>, Vec<Vec<usize>>), ENNError> {
        let (draws, idx) = self.posterior_function_draw_warped(x, params, function_seeds, flags)?;
        let draws = self.calibrate_published_draws(draws, params, |bare| {
            self.posterior(x, bare, flags)
        })?;
        Ok((draws, idx))
    }

    pub(crate) fn conditional_posterior_warped(
        &self,
        x_whatif: &ArrayView2<f64>,
        y_whatif: &ArrayView2<f64>,
        x: &ArrayView2<f64>,
        params: &ENNParams,
        flags: &PosteriorFlags,
    ) -> Result<ENNNormal, ENNError> {
        let (y_z, _) = self.warp_observations(y_whatif, None)?;
        let internals =
            compute_conditional_posterior_internals(self, x, x_whatif, &y_z.view(), params, flags)?;
        let mut out = ENNNormal::new(
            internals.mu.into_dyn(),
            internals.se.into_dyn(),
            internals.se_epi.into_dyn(),
            internals.se_ale.into_dyn(),
            Some(idx_nested_to_array2(&internals.idx)),
        );
        self.apply_normal_blend(&mut out, params)?;
        Ok(out)
    }

    pub fn conditional_posterior(
        &self,
        x_whatif: &ArrayView2<f64>,
        y_whatif: &ArrayView2<f64>,
        x: &ArrayView2<f64>,
        params: &ENNParams,
        flags: &PosteriorFlags,
    ) -> Result<ENNNormal, ENNError> {
        let mut out = self.conditional_posterior_warped(x_whatif, y_whatif, x, params, flags)?;
        self.naturalize_enn_normal(&mut out)?;
        self.calibrate_normal(&mut out, params)?;
        Ok(out)
    }

    pub(crate) fn naturalize_enn_normal(&self, out: &mut ENNNormal) -> Result<(), ENNError> {
        if is_identity_bounds(self.y_bounds()) {
            return Ok(());
        }
        let ndim = out.mu.ndim();
        if ndim == 2 {
            let mut mu = out
                .mu
                .clone()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| ENNError::ShapeError(e.to_string()))?;
            let mut se = out
                .se
                .clone()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| ENNError::ShapeError(e.to_string()))?;
            let mut se_epi = out
                .se_epi
                .clone()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| ENNError::ShapeError(e.to_string()))?;
            let mut se_ale = out
                .se_ale
                .clone()
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| ENNError::ShapeError(e.to_string()))?;
            naturalize_mu_se(&mut mu, &mut se, &mut se_epi, &mut se_ale, self.y_bounds());
            out.mu = mu.into_dyn();
            out.se = se.into_dyn();
            out.se_epi = se_epi.into_dyn();
            out.se_ale = se_ale.into_dyn();
        } else if ndim == 3 {
            let shape = out.mu.shape().to_vec();
            let p = shape[0];
            let b = shape[1];
            let m = shape[2];
            for pi in 0..p {
                let mut mu = Array2::zeros((b, m));
                let mut se = Array2::zeros((b, m));
                let mut se_epi = Array2::zeros((b, m));
                let mut se_ale = Array2::zeros((b, m));
                for bi in 0..b {
                    for mi in 0..m {
                        mu[[bi, mi]] = out.mu[[pi, bi, mi]];
                        se[[bi, mi]] = out.se[[pi, bi, mi]];
                        se_epi[[bi, mi]] = out.se_epi[[pi, bi, mi]];
                        se_ale[[bi, mi]] = out.se_ale[[pi, bi, mi]];
                    }
                }
                naturalize_mu_se(&mut mu, &mut se, &mut se_epi, &mut se_ale, self.y_bounds());
                for bi in 0..b {
                    for mi in 0..m {
                        out.mu[[pi, bi, mi]] = mu[[bi, mi]];
                        out.se[[pi, bi, mi]] = se[[bi, mi]];
                        out.se_epi[[pi, bi, mi]] = se_epi[[bi, mi]];
                        out.se_ale[[pi, bi, mi]] = se_ale[[bi, mi]];
                    }
                }
            }
        }
        Ok(())
    }

    pub(crate) fn naturalize_draws_3d(&self, draws: &mut Array3<f64>) {
        if is_identity_bounds(self.y_bounds()) {
            return;
        }
        let bounds = self.y_bounds();
        for mut sample in draws.axis_iter_mut(ndarray::Axis(0)) {
            let inv = inv_y(sample.view(), bounds);
            sample.assign(&inv);
        }
    }

    pub(crate) fn warp_draws_3d(&self, draws: &mut Array3<f64>) -> Result<(), ENNError> {
        if is_identity_bounds(self.y_bounds()) {
            return Ok(());
        }
        let bounds = self.y_bounds();
        for mut sample in draws.axis_iter_mut(ndarray::Axis(0)) {
            let warped = warp_y(sample.view(), bounds)?;
            sample.assign(&warped);
        }
        Ok(())
    }

    pub fn batch_posterior(
        &self,
        x: &ArrayView2<f64>,
        paramss: &[ENNParams],
        flags: &PosteriorFlags,
    ) -> Result<ENNNormal, ENNError> {
        crate::util::check_finite_query(x)?;
        let mut out = <Self as PosteriorComputation>::batch_posterior(self, x, paramss, flags)?;
        self.naturalize_enn_normal(&mut out)?;
        self.calibrate_batch(&mut out, paramss)?;
        Ok(out)
    }
}
