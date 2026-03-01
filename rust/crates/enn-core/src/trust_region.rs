//! Trust region implementations for TuRBO optimizer.

use ndarray::{Array1, ArrayView1};
use thiserror::Error;

/// Errors that can occur in trust region operations.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum TrustRegionError {
    /// Invalid parameter value.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    /// Invalid state for operation.
    #[error("Invalid state: {0}")]
    InvalidState(String),
}

/// Configuration for trust region length parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TRLengthConfig {
    /// Initial trust region length.
    pub length_init: f64,
    /// Minimum trust region length.
    pub length_min: f64,
    /// Maximum trust region length.
    pub length_max: f64,
}

impl Default for TRLengthConfig {
    fn default() -> Self {
        Self {
            length_init: 0.8,
            length_min: 0.5f64.powi(7), // 0.5^7 ≈ 0.0078
            length_max: 1.6,
        }
    }
}

impl TRLengthConfig {
    /// Create new TRLengthConfig with custom values.
    pub fn new(length_init: f64, length_min: f64, length_max: f64) -> Self {
        Self {
            length_init,
            length_min,
            length_max,
        }
    }
}

/// Trust region for single-objective optimization (TuRBO).
///
/// Implements success/failure-based length adaptation.
#[derive(Debug, Clone, PartialEq)]
pub struct TurboTrustRegion {
    /// Current trust region length.
    length: f64,
    /// Consecutive failure count.
    failure_counter: i32,
    /// Consecutive success count.
    success_counter: i32,
    /// Best observed value.
    best_value: f64,
    /// Number of observations at last update.
    prev_num_obs: usize,
    /// Previous observed values for scale computation.
    prev_values: Vec<f64>,
    /// Success tolerance (consecutive successes before expansion).
    success_tolerance: i32,
    /// Failure tolerance (consecutive failures before contraction).
    failure_tolerance: Option<i32>,
    /// Number of dimensions.
    num_dim: usize,
    /// Batch size (number of arms).
    num_arms: Option<usize>,
    /// Length configuration.
    config: TRLengthConfig,
}

impl TurboTrustRegion {
    /// Create a new TurboTrustRegion.
    pub fn new(num_dim: usize, config: TRLengthConfig) -> Self {
        Self {
            length: config.length_init,
            failure_counter: 0,
            success_counter: 0,
            best_value: f64::NEG_INFINITY,
            prev_num_obs: 0,
            prev_values: Vec::new(),
            success_tolerance: 3,
            failure_tolerance: None,
            num_dim,
            num_arms: None,
            config,
        }
    }

    /// Initialize or update batch size.
    pub fn set_num_arms(&mut self, num_arms: usize) {
        self.num_arms = Some(num_arms);
        self.compute_failure_tolerance();
    }

    /// Compute failure tolerance based on num_arms and num_dim.
    fn compute_failure_tolerance(&mut self) {
        if let Some(num_arms) = self.num_arms {
            let tolerance = ((4.0 / num_arms as f64)
                .max(self.num_dim as f64 / num_arms as f64))
            .ceil() as i32;
            self.failure_tolerance = Some(tolerance.max(1));
        }
    }

    /// Get current trust region length.
    pub fn length(&self) -> f64 {
        self.length
    }

    /// Check if restart is needed (length below minimum).
    pub fn needs_restart(&self) -> bool {
        self.length < self.config.length_min
    }

    /// Update trust region based on new observations.
    ///
    /// # Arguments
    ///
    /// * `y_new` - New objective values
    /// * `num_obs` - Total number of observations so far
    pub fn update(
        &mut self,
        y_new: &ArrayView1<f64>,
        num_obs: usize,
    ) -> Result<(), TrustRegionError> {
        if y_new.is_empty() {
            return Ok(());
        }

        // Get new best value
        let new_best = y_new.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        // Compute scale from previous values
        let scale = if self.prev_values.len() >= 2 {
            let min_val = self.prev_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = self.prev_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            (max_val - min_val).max(1e-6)
        } else {
            1.0
        };

        // Check for improvement
        let improvement_threshold = 1e-3 * scale;
        let improved = new_best > self.best_value + improvement_threshold;

        if improved {
            self.success_counter += 1;
            self.failure_counter = 0;
            self.best_value = new_best;
        } else {
            self.failure_counter += 1;
            self.success_counter = 0;
        }

        // Adapt length
        let success_tol = self.success_tolerance;
        let failure_tol = self.failure_tolerance.unwrap_or(4);

        if self.success_counter >= success_tol {
            // Expand: double length
            self.length = (self.length * 2.0).min(self.config.length_max);
            self.success_counter = 0;
        } else if self.failure_counter >= failure_tol {
            // Contract: halve length
            self.length *= 0.5;
            self.failure_counter = 0;
        }

        // Update previous values (B3: scale computed from single batch only;
        // Python may use a larger window — verify parity if TR length diverges)
        self.prev_values.clear();
        self.prev_values.extend(y_new.iter().copied());
        self.prev_num_obs = num_obs;

        Ok(())
    }

    /// Compute trust region bounds in 1D.
    ///
    /// Returns (lower_bounds, upper_bounds) for each dimension.
    pub fn compute_bounds_1d(
        &self,
        x_center: &ArrayView1<f64>,
        lengthscales: Option<&ArrayView1<f64>>,
    ) -> (Array1<f64>, Array1<f64>) {
        let num_dim = x_center.len();
        let half_length = if let Some(ls) = lengthscales {
            ls * self.length / 2.0
        } else {
            Array1::from_elem(num_dim, self.length / 2.0)
        };

        let lb = x_center - &half_length;
        let ub = x_center + &half_length;

        // Clip to [0, 1]
        let lb = lb.mapv(|v| v.clamp(0.0, 1.0));
        let ub = ub.mapv(|v| v.clamp(0.0, 1.0));

        (lb, ub)
    }

    /// Restart the trust region (reset to initial state).
    pub fn restart(&mut self) {
        self.length = self.config.length_init;
        self.failure_counter = 0;
        self.success_counter = 0;
        self.best_value = f64::NEG_INFINITY;
        self.prev_num_obs = 0;
        self.prev_values.clear();
    }
}

/// Null trust region (no trust region management).
///
/// Always returns full bounds [0, 1]^d and never needs restart.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoTrustRegion {
    num_dim: usize,
}

impl NoTrustRegion {
    /// Create a new NoTrustRegion.
    pub fn new(num_dim: usize) -> Self {
        Self { num_dim }
    }

    /// Get current length (always 1.0).
    pub fn length(&self) -> f64 {
        1.0
    }

    /// Check if restart is needed (always false).
    pub fn needs_restart(&self) -> bool {
        false
    }

    /// Update (no-op).
    pub fn update(&mut self, _y_new: &ArrayView1<f64>, _num_obs: usize) {}

    /// Compute trust region bounds (always returns [0, 1]^d).
    pub fn compute_bounds_1d(
        &self,
        _x_center: &ArrayView1<f64>,
        _lengthscales: Option<&ArrayView1<f64>>,
    ) -> (Array1<f64>, Array1<f64>) {
        let lb = Array1::zeros(self.num_dim);
        let ub = Array1::ones(self.num_dim);
        (lb, ub)
    }

    /// Restart (no-op).
    pub fn restart(&mut self) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_turbo_trust_region_creation() {
        let config = TRLengthConfig::default();
        let tr = TurboTrustRegion::new(5, config);

        assert_eq!(tr.length(), 0.8);
        assert!(!tr.needs_restart());
    }

    #[test]
    fn test_turbo_update_success() {
        let config = TRLengthConfig::default();
        let mut tr = TurboTrustRegion::new(5, config);
        tr.set_num_arms(1);

        // First update - establish best value
        tr.update(&array![1.0].view(), 1).unwrap();
        assert_eq!(tr.length(), 0.8);

        // Three consecutive improvements should expand
        tr.update(&array![2.0].view(), 2).unwrap();
        tr.update(&array![3.0].view(), 3).unwrap();
        tr.update(&array![4.0].view(), 4).unwrap();

        assert!(tr.length() > 0.8); // Should have expanded
    }

    #[test]
    fn test_turbo_update_failure() {
        let config = TRLengthConfig::default();
        let mut tr = TurboTrustRegion::new(5, config);
        tr.set_num_arms(1);

        // Establish best value
        tr.update(&array![1.0].view(), 1).unwrap();

        // Multiple failures should contract
        let failure_tol = tr.failure_tolerance.unwrap();
        for i in 0..failure_tol {
            tr.update(&array![0.5].view(), (2 + i) as usize).unwrap();
        }

        assert!(tr.length() < 0.8); // Should have contracted
    }

    #[test]
    fn test_turbo_restart() {
        let config = TRLengthConfig::default();
        let mut tr = TurboTrustRegion::new(5, config);
        tr.set_num_arms(1);

        // Contract length
        tr.update(&array![1.0].view(), 1).unwrap();
        for _ in 0..10 {
            tr.update(&array![0.5].view(), 2).unwrap();
        }

        assert!(tr.length() < 0.8);

        // Restart
        tr.restart();
        assert_eq!(tr.length(), 0.8);
        assert!(!tr.needs_restart());
    }

    #[test]
    fn test_turbo_bounds() {
        let config = TRLengthConfig::default();
        let tr = TurboTrustRegion::new(5, config);

        let center = array![0.5, 0.5, 0.5, 0.5, 0.5];
        let (lb, ub) = tr.compute_bounds_1d(&center.view(), None);

        assert!(lb.iter().all(|&v| (0.0..=1.0).contains(&v)));
        assert!(ub.iter().all(|&v| (0.0..=1.0).contains(&v)));

        for i in 0..5 {
            assert!(lb[i] < ub[i]);
        }
    }

    #[test]
    fn test_no_trust_region() {
        let tr = NoTrustRegion::new(5);

        assert_eq!(tr.length(), 1.0);
        assert!(!tr.needs_restart());

        let center = array![0.5, 0.5, 0.5, 0.5, 0.5];
        let (lb, ub) = tr.compute_bounds_1d(&center.view(), None);

        assert_eq!(lb, array![0.0, 0.0, 0.0, 0.0, 0.0]);
        assert_eq!(ub, array![1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn test_trust_region_error_display() {
        let e1 = TrustRegionError::InvalidParameter("bad param".to_string());
        let e2 = TrustRegionError::InvalidState("bad state".to_string());
        assert!(e1.to_string().contains("Invalid parameter"));
        assert!(e2.to_string().contains("Invalid state"));
    }
}
