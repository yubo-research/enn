//! Traits for ENN algorithms.

use ndarray::{Array3, ArrayView2};

use crate::error::ENNError;
use crate::params::{ENNNormal, ENNParams, PosteriorFlags};

/// Posterior computation extension for ENN model.
pub trait PosteriorComputation {
    /// Compute posterior predictive distribution.
    fn posterior(
        &self,
        x: &ArrayView2<f64>,
        params: &ENNParams,
        flags: &PosteriorFlags,
    ) -> Result<ENNNormal, ENNError>;

    /// Compute batch posterior with multiple parameter sets.
    fn batch_posterior(
        &self,
        x: &ArrayView2<f64>,
        paramss: &[ENNParams],
        flags: &PosteriorFlags,
    ) -> Result<ENNNormal, ENNError>;

    /// Draw function samples from posterior.
    fn posterior_function_draw(
        &self,
        x: &ArrayView2<f64>,
        params: &ENNParams,
        function_seeds: &[i64],
        flags: &PosteriorFlags,
    ) -> Result<(Array3<f64>, Vec<Vec<usize>>), ENNError>;

    /// Compute conditional posterior with what-if scenarios.
    fn conditional_posterior(
        &self,
        x_whatif: &ArrayView2<f64>,
        y_whatif: &ArrayView2<f64>,
        x: &ArrayView2<f64>,
        params: &ENNParams,
        flags: &PosteriorFlags,
    ) -> Result<ENNNormal, ENNError>;

    /// Draw function samples from conditional posterior.
    fn conditional_posterior_function_draw(
        &self,
        x_whatif: &ArrayView2<f64>,
        y_whatif: &ArrayView2<f64>,
        x: &ArrayView2<f64>,
        params: &ENNParams,
        function_seeds: &[i64],
        flags: &PosteriorFlags,
    ) -> Result<(Array3<f64>, Vec<Vec<usize>>), ENNError>;
}
