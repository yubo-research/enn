//! Dual warped/natural posterior API (kiss statements_per_file split).

use ndarray::{Array2, Array3, ArrayView2};

use super::draw_compute::draw_from_internals;
use super::light::{compute_posterior_light, idx_nested_to_array2};
use super::{compute_conditional_posterior_internals, compute_posterior_internals};
use crate::error::ENNError;
use crate::model::EpistemicNearestNeighbors;
use crate::params::{ENNNormal, ENNParams, PosteriorFlags};
use crate::traits::PosteriorComputation;
use crate::y_bounds::{inv_y, is_identity_bounds, naturalize_mu_se};

impl EpistemicNearestNeighbors {
    /// Crate-internal posterior in warped `z` space (storage / fit / acq).
    pub(crate) fn posterior_warped(
        &self,
        x: &ArrayView2<f64>,
        params: &ENNParams,
        flags: &PosteriorFlags,
    ) -> Result<ENNNormal, ENNError> {
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
        Ok(ENNNormal::new(
            mu.into_dyn(),
            se.into_dyn(),
            se_epi.into_dyn(),
            se_ale.into_dyn(),
            Some(idx),
        ))
    }

    /// Public posterior in natural y units.
    pub fn posterior(
        &self,
        x: &ArrayView2<f64>,
        params: &ENNParams,
        flags: &PosteriorFlags,
    ) -> Result<ENNNormal, ENNError> {
        let mut out = self.posterior_warped(x, params, flags)?;
        self.naturalize_enn_normal(&mut out)?;
        Ok(out)
    }

    pub(crate) fn posterior_function_draw_warped(
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

    pub fn posterior_function_draw(
        &self,
        x: &ArrayView2<f64>,
        params: &ENNParams,
        function_seeds: &[i64],
        flags: &PosteriorFlags,
    ) -> Result<(Array3<f64>, Vec<Vec<usize>>), ENNError> {
        let (mut draws, idx) =
            self.posterior_function_draw_warped(x, params, function_seeds, flags)?;
        self.naturalize_draws_3d(&mut draws);
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
        let internals = compute_conditional_posterior_internals(
            self,
            x,
            x_whatif,
            &y_z.view(),
            params,
            flags,
        )?;
        Ok(ENNNormal::new(
            internals.mu.into_dyn(),
            internals.se.into_dyn(),
            internals.se_epi.into_dyn(),
            internals.se_ale.into_dyn(),
            Some(idx_nested_to_array2(&internals.idx)),
        ))
    }

    pub fn conditional_posterior(
        &self,
        x_whatif: &ArrayView2<f64>,
        y_whatif: &ArrayView2<f64>,
        x: &ArrayView2<f64>,
        params: &ENNParams,
        flags: &PosteriorFlags,
    ) -> Result<ENNNormal, ENNError> {
        let mut out =
            self.conditional_posterior_warped(x_whatif, y_whatif, x, params, flags)?;
        self.naturalize_enn_normal(&mut out)?;
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

    /// Public batch posterior in natural units.
    pub fn batch_posterior(
        &self,
        x: &ArrayView2<f64>,
        paramss: &[ENNParams],
        flags: &PosteriorFlags,
    ) -> Result<ENNNormal, ENNError> {
        let mut out = <Self as PosteriorComputation>::batch_posterior(self, x, paramss, flags)?;
        self.naturalize_enn_normal(&mut out)?;
        Ok(out)
    }
}
