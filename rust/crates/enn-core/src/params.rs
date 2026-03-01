//! ENN parameter and configuration data structures.

use ndarray::ArrayD;
use thiserror::Error;

/// Errors that can occur when creating ENN parameters.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ParamsError {
    /// Invalid number of neighbors.
    #[error("k_num_neighbors must be > 0, got {0}")]
    InvalidK(i32),
    /// Invalid epistemic variance scale.
    #[error("epistemic_variance_scale must be >= 0 and finite, got {0}")]
    InvalidEpistemicVariance(f64),
    /// Invalid aleatoric variance scale.
    #[error("aleatoric_variance_scale must be >= 0 and finite, got {0}")]
    InvalidAleatoricVariance(f64),
}

/// Parameters for Epistemic Nearest Neighbors computation.
///
/// These parameters control the behavior of the ENN posterior computation,
/// including the number of neighbors and variance scaling factors.
///
/// # Fields
///
/// * `k_num_neighbors` - Number of nearest neighbors to use (must be > 0)
/// * `epistemic_variance_scale` - Scale factor for epistemic (model) uncertainty
/// * `aleatoric_variance_scale` - Scale factor for aleatoric (observation) noise
///
/// # Example
///
/// ```
/// use enn_core::params::ENNParams;
///
/// let params = ENNParams::new(5, 1.0, 0.1).unwrap();
/// assert_eq!(params.k_num_neighbors, 5);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ENNParams {
    /// Number of nearest neighbors to use.
    pub k_num_neighbors: i32,
    /// Scale factor for epistemic uncertainty.
    pub epistemic_variance_scale: f64,
    /// Scale factor for aleatoric noise.
    pub aleatoric_variance_scale: f64,
}

impl ENNParams {
    /// Create a new ENNParams instance with validation.
    ///
    /// # Arguments
    ///
    /// * `k_num_neighbors` - Number of neighbors (must be > 0)
    /// * `epistemic_variance_scale` - Epistemic variance scale (must be >= 0 and finite)
    /// * `aleatoric_variance_scale` - Aleatoric variance scale (must be >= 0 and finite)
    ///
    /// # Errors
    ///
    /// Returns `ParamsError` if any parameter is invalid.
    pub fn new(
        k_num_neighbors: i32,
        epistemic_variance_scale: f64,
        aleatoric_variance_scale: f64,
    ) -> Result<Self, ParamsError> {
        // Validate k
        if k_num_neighbors <= 0 {
            return Err(ParamsError::InvalidK(k_num_neighbors));
        }

        // Validate epistemic variance
        if !epistemic_variance_scale.is_finite() || epistemic_variance_scale < 0.0 {
            return Err(ParamsError::InvalidEpistemicVariance(
                epistemic_variance_scale,
            ));
        }

        // Validate aleatoric variance
        if !aleatoric_variance_scale.is_finite() || aleatoric_variance_scale < 0.0 {
            return Err(ParamsError::InvalidAleatoricVariance(
                aleatoric_variance_scale,
            ));
        }

        Ok(Self {
            k_num_neighbors,
            epistemic_variance_scale,
            aleatoric_variance_scale,
        })
    }
}

/// Flags controlling posterior computation behavior.
///
/// These flags allow fine-grained control over how the ENN posterior
/// is computed.
///
/// # Fields
///
/// * `exclude_nearest` - Whether to exclude the nearest neighbor from computation
/// * `observation_noise` - Whether to include observation noise in the uncertainty
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PosteriorFlags {
    /// Exclude the nearest neighbor from computation.
    pub exclude_nearest: bool,
    /// Include observation noise in uncertainty computation.
    pub observation_noise: bool,
}

impl PosteriorFlags {
    /// Create new PosteriorFlags with default values (all false).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set exclude_nearest flag.
    pub fn with_exclude_nearest(mut self, value: bool) -> Self {
        self.exclude_nearest = value;
        self
    }

    /// Set observation_noise flag.
    pub fn with_observation_noise(mut self, value: bool) -> Self {
        self.observation_noise = value;
        self
    }
}

/// Normal distribution result from ENN posterior computation.
///
/// Represents the posterior predictive distribution as a Gaussian
/// with mean and standard error.
///
/// # Fields
///
/// * `mu` - Predictive mean for each query point
/// * `se` - Predictive standard error for each query point
/// * `idx` - Optional neighbor indices used for each prediction
#[derive(Debug, Clone, PartialEq)]
pub struct ENNNormal {
    /// Predictive means.
    pub mu: ArrayD<f64>,
    /// Predictive standard errors.
    pub se: ArrayD<f64>,
    /// Optional neighbor indices.
    pub idx: Option<Vec<Vec<usize>>>,
}

impl ENNNormal {
    /// Create a new ENNNormal instance.
    pub fn new(mu: ArrayD<f64>, se: ArrayD<f64>, idx: Option<Vec<Vec<usize>>>) -> Self {
        Self { mu, se, idx }
    }

    /// Get the number of query points.
    pub fn len(&self) -> usize {
        self.mu.len()
    }

    /// Check if there are no query points.
    pub fn is_empty(&self) -> bool {
        self.mu.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, ArrayD, IxDyn};

    #[test]
    fn test_enn_params_valid() {
        let params = ENNParams::new(5, 1.0, 0.1).unwrap();
        assert_eq!(params.k_num_neighbors, 5);
        assert_eq!(params.epistemic_variance_scale, 1.0);
        assert_eq!(params.aleatoric_variance_scale, 0.1);
    }

    #[test]
    fn test_enn_params_invalid_k() {
        assert!(matches!(
            ENNParams::new(0, 1.0, 0.1),
            Err(ParamsError::InvalidK(0))
        ));
        assert!(matches!(
            ENNParams::new(-1, 1.0, 0.1),
            Err(ParamsError::InvalidK(-1))
        ));
    }

    #[test]
    fn test_enn_params_invalid_epistemic() {
        assert!(matches!(
            ENNParams::new(5, -1.0, 0.1),
            Err(ParamsError::InvalidEpistemicVariance(-1.0))
        ));
        assert!(matches!(
            ENNParams::new(5, f64::NAN, 0.1),
            Err(ParamsError::InvalidEpistemicVariance(_))
        ));
        assert!(matches!(
            ENNParams::new(5, f64::INFINITY, 0.1),
            Err(ParamsError::InvalidEpistemicVariance(_))
        ));
    }

    #[test]
    fn test_enn_params_invalid_aleatoric() {
        assert!(matches!(
            ENNParams::new(5, 1.0, -0.1),
            Err(ParamsError::InvalidAleatoricVariance(-0.1))
        ));
    }

    #[test]
    fn test_posterior_flags_default() {
        let flags = PosteriorFlags::new();
        assert!(!flags.exclude_nearest);
        assert!(!flags.observation_noise);
    }

    #[test]
    fn test_posterior_flags_builder() {
        let flags = PosteriorFlags::new()
            .with_exclude_nearest(true)
            .with_observation_noise(true);
        assert!(flags.exclude_nearest);
        assert!(flags.observation_noise);
    }

    #[test]
    fn test_enn_normal() {
        let mu = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();
        let se = array![[0.1, 0.2], [0.3, 0.4]].into_dyn();
        let normal = ENNNormal::new(mu.clone(), se.clone(), None);

        assert_eq!(normal.len(), 4);
        assert!(!normal.is_empty());
        assert_eq!(normal.mu, mu);
        assert_eq!(normal.se, se);
    }

    #[test]
    fn test_enn_normal_with_idx() {
        let mu = array![[1.0], [2.0]].into_dyn();
        let se = array![[0.1], [0.2]].into_dyn();
        let idx = Some(vec![vec![0, 1], vec![1, 2]]);
        let normal = ENNNormal::new(mu, se, idx.clone());

        assert_eq!(normal.idx, idx);
    }

    #[test]
    fn test_enn_normal_empty() {
        let mu = ArrayD::<f64>::zeros(IxDyn(&[0, 0]));
        let se = ArrayD::<f64>::zeros(IxDyn(&[0, 0]));
        let normal = ENNNormal::new(mu, se, None);

        assert!(normal.is_empty());
        assert_eq!(normal.len(), 0);
    }
}
