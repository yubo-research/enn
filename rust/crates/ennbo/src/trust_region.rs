
use ndarray::{s, Array1, ArrayView1};
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum TrustRegionError {
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    #[error("Invalid state: {0}")]
    InvalidState(String),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TRLengthConfig {
    pub length_init: f64,
    pub length_min: f64,
    pub length_max: f64,
}

impl Default for TRLengthConfig {
    fn default() -> Self {
        Self {
            length_init: 0.8,
            length_min: 0.5f64.powi(7), 
            length_max: 1.6,
        }
    }
}

fn reject_length(message: String) -> crate::error::ENNError {
    crate::error::ENNError::InvalidParameter(message)
}

impl TRLengthConfig {
    pub fn new(length_init: f64, length_min: f64, length_max: f64) -> Self {
        Self {
            length_init,
            length_min,
            length_max,
        }
    }

    fn check(self) -> Result<Self, crate::error::ENNError> {
        if self.length_init <= 0.0 {
            return Err(reject_length(format!(
                "length_init must be > 0, got {}",
                self.length_init
            )));
        }
        if self.length_min <= 0.0 {
            return Err(reject_length(format!(
                "length_min must be > 0, got {}",
                self.length_min
            )));
        }
        if self.length_max <= 0.0 {
            return Err(reject_length(format!(
                "length_max must be > 0, got {}",
                self.length_max
            )));
        }
        if self.length_min >= self.length_max {
            return Err(reject_length(format!(
                "length_min must be < length_max, got {} >= {}",
                self.length_min, self.length_max
            )));
        }
        if self.length_init > self.length_max || self.length_min > self.length_init {
            return Err(reject_length(format!(
                "lengths out of order: min {} init {} max {}",
                self.length_min, self.length_init, self.length_max
            )));
        }
        Ok(self)
    }

    pub fn resolve(
        length_init: Option<f64>,
        length_min: Option<f64>,
        length_max: Option<f64>,
    ) -> Result<Self, crate::error::ENNError> {
        let defaults = Self::default();
        Self {
            length_init: length_init.unwrap_or(defaults.length_init),
            length_min: length_min.unwrap_or(defaults.length_min),
            length_max: length_max.unwrap_or(defaults.length_max),
        }
        .check()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TurboTrustRegion {
    length: f64,
    failure_counter: i32,
    success_counter: i32,
    best_value: f64,
    prev_num_obs: usize,
    success_tolerance: i32,
    failure_tolerance: Option<i32>,
    num_dim: usize,
    num_arms: Option<usize>,
    config: TRLengthConfig,
    hist_ymin: f64,
    hist_ymax: f64,
}

impl TurboTrustRegion {
    pub fn new(num_dim: usize, config: TRLengthConfig) -> Self {
        Self {
            length: config.length_init,
            failure_counter: 0,
            success_counter: 0,
            best_value: f64::NEG_INFINITY,
            prev_num_obs: 0,
            success_tolerance: 3,
            failure_tolerance: None,
            num_dim,
            num_arms: None,
            config,
            hist_ymin: f64::INFINITY,
            hist_ymax: f64::NEG_INFINITY,
        }
    }

    fn incorporate_hist(&mut self, y_new: &ArrayView1<f64>) {
        for &v in y_new.iter() {
            self.hist_ymin = self.hist_ymin.min(v);
            self.hist_ymax = self.hist_ymax.max(v);
        }
    }

    fn improvement_scale(&self) -> f64 {
        if self.best_value.is_finite() {
            self.best_value.abs()
        } else {
            0.0
        }
    }

    pub fn set_num_arms(&mut self, num_arms: usize) {
        self.num_arms = Some(num_arms);
        self.compute_failure_tolerance();
    }

    fn compute_failure_tolerance(&mut self) {
        if let Some(num_arms) = self.num_arms {
            let tolerance =
                ((4.0 / num_arms as f64).max(self.num_dim as f64 / num_arms as f64)).ceil() as i32;
            self.failure_tolerance = Some(tolerance.max(1));
        }
    }

    pub fn length(&self) -> f64 {
        self.length
    }

    pub fn prev_num_obs(&self) -> usize {
        self.prev_num_obs
    }

    pub fn set_best_value(&mut self, value: f64) {
        self.best_value = value;
    }

    pub fn update_scalar_incumbent(
        &mut self,
        num_obs: usize,
        y_incumbent_value: f64,
        scale: f64,
    ) -> Result<(), TrustRegionError> {
        if num_obs == 0 || num_obs == self.prev_num_obs {
            return Ok(());
        }
        if num_obs < self.prev_num_obs {
            return Err(TrustRegionError::InvalidState(format!(
                "num_obs went backwards: {} < {}",
                num_obs, self.prev_num_obs
            )));
        }
        if !self.best_value.is_finite() {
            self.best_value = y_incumbent_value;
            self.prev_num_obs = num_obs;
            return Ok(());
        }
        let improved = y_incumbent_value > self.best_value + 1e-3 * scale;
        if improved {
            self.success_counter += 1;
            self.failure_counter = 0;
        } else {
            self.failure_counter += 1;
            self.success_counter = 0;
        }
        let success_tol = self.success_tolerance;
        let failure_tol = self.failure_tolerance.unwrap_or(4);
        if self.success_counter >= success_tol {
            self.length = (self.length * 2.0).min(self.config.length_max);
            self.success_counter = 0;
        } else if self.failure_counter >= failure_tol {
            self.length *= 0.5;
            self.failure_counter = 0;
        }
        self.best_value = self.best_value.max(y_incumbent_value);
        self.prev_num_obs = num_obs;
        Ok(())
    }

    pub fn needs_restart(&self) -> bool {
        self.length < self.config.length_min
    }

    pub fn update(
        &mut self,
        y_all: &ArrayView1<f64>,
        num_obs: usize,
    ) -> Result<(), TrustRegionError> {
        let n = y_all.len();
        if n == 0 || n == self.prev_num_obs {
            return Ok(());
        }
        if n < self.prev_num_obs {
            return Err(TrustRegionError::InvalidState(format!(
                "num_obs went backwards: {} < {}",
                n, self.prev_num_obs
            )));
        }
        if num_obs != n {
            return Err(TrustRegionError::InvalidParameter(format!(
                "num_obs {} must equal y_all.len() {}",
                num_obs, n
            )));
        }

        
        if !self.best_value.is_finite() {
            let new_best = y_all.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            self.best_value = new_best;
            self.prev_num_obs = n;
            return Ok(());
        }

        
        let new_batch = y_all.slice(s![self.prev_num_obs..]);

        let new_best = new_batch.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

        let improvement_threshold = 1e-3 * self.improvement_scale();
        let improved = new_best > self.best_value + improvement_threshold;

        if improved {
            self.success_counter += 1;
            self.failure_counter = 0;
            self.best_value = self.best_value.max(new_best);
        } else {
            self.failure_counter += 1;
            self.success_counter = 0;
        }

        
        let success_tol = self.success_tolerance;
        let failure_tol = self.failure_tolerance.unwrap_or(4);

        if self.success_counter >= success_tol {
            
            self.length = (self.length * 2.0).min(self.config.length_max);
            self.success_counter = 0;
        } else if self.failure_counter >= failure_tol {
            
            self.length *= 0.5;
            self.failure_counter = 0;
        }

        self.prev_num_obs = n;
        Ok(())
    }

    pub fn update_with_incumbent(
        &mut self,
        y_all: &ArrayView1<f64>,
        num_obs: usize,
        y_incumbent_value: f64,
    ) -> Result<(), TrustRegionError> {
        let n = y_all.len();
        if n == 0 || n == self.prev_num_obs {
            return Ok(());
        }
        if n < self.prev_num_obs {
            return Err(TrustRegionError::InvalidState(format!(
                "num_obs went backwards: {} < {}",
                n, self.prev_num_obs
            )));
        }
        if num_obs != n {
            return Err(TrustRegionError::InvalidParameter(format!(
                "num_obs {} must equal y_all.len() {}",
                num_obs, n
            )));
        }

        let y_new = y_all.slice(s![self.prev_num_obs..]);
        self.update_with_incumbent_new_batch(&y_new, num_obs, y_incumbent_value)
    }

    pub fn update_with_incumbent_new_batch(
        &mut self,
        y_new: &ArrayView1<f64>,
        num_obs: usize,
        y_incumbent_value: f64,
    ) -> Result<(), TrustRegionError> {
        let n_new = y_new.len();
        if n_new == 0 {
            return Ok(());
        }
        let n = self.prev_num_obs + n_new;
        if num_obs != n {
            return Err(TrustRegionError::InvalidParameter(format!(
                "num_obs {} must equal prev_num_obs {} + y_new.len() {}",
                num_obs, self.prev_num_obs, n_new
            )));
        }

        if !self.best_value.is_finite() {
            self.best_value = y_incumbent_value;
            self.incorporate_hist(y_new);
            self.prev_num_obs = n;
            return Ok(());
        }

        let scale = self.improvement_scale();
        let improved = y_incumbent_value > self.best_value + 1e-3 * scale;
        if improved {
            self.success_counter += 1;
            self.failure_counter = 0;
        } else {
            self.failure_counter += 1;
            self.success_counter = 0;
        }

        let success_tol = self.success_tolerance;
        let failure_tol = self.failure_tolerance.unwrap_or(4);
        if self.success_counter >= success_tol {
            self.length = (self.length * 2.0).min(self.config.length_max);
            self.success_counter = 0;
        } else if self.failure_counter >= failure_tol {
            self.length *= 0.5;
            self.failure_counter = 0;
        }
        self.best_value = self.best_value.max(y_incumbent_value);
        self.incorporate_hist(y_new);
        self.prev_num_obs = n;
        Ok(())
    }

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

        
        let lb = lb.mapv(|v| v.clamp(0.0, 1.0));
        let ub = ub.mapv(|v| v.clamp(0.0, 1.0));

        (lb, ub)
    }

    pub fn restart(&mut self) {
        self.length = self.config.length_init;
        self.failure_counter = 0;
        self.success_counter = 0;
        self.best_value = f64::NEG_INFINITY;
    }

    pub fn set_prev_num_obs(&mut self, prev_num_obs: usize) {
        self.prev_num_obs = prev_num_obs;
    }

    pub fn seed_scale_history(&mut self, y_prefix: &ArrayView1<f64>) {
        self.incorporate_hist(y_prefix);
        if !self.best_value.is_finite() {
            if let Some(best) = y_prefix.iter().copied().reduce(f64::max) {
                self.best_value = best;
            }
        }
    }

    pub fn restart_local(&mut self) {
        self.restart();
        self.prev_num_obs = 0;
        self.hist_ymin = f64::INFINITY;
        self.hist_ymax = f64::NEG_INFINITY;
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NoTrustRegion {
    num_dim: usize,
}

impl NoTrustRegion {
    pub fn new(num_dim: usize) -> Self {
        Self { num_dim }
    }

    pub fn length(&self) -> f64 {
        1.0
    }

    pub fn needs_restart(&self) -> bool {
        false
    }

    pub fn update(&mut self, _y_new: &ArrayView1<f64>, _num_obs: usize) {}

    pub fn compute_bounds_1d(
        &self,
        _x_center: &ArrayView1<f64>,
        _lengthscales: Option<&ArrayView1<f64>>,
    ) -> (Array1<f64>, Array1<f64>) {
        let lb = Array1::zeros(self.num_dim);
        let ub = Array1::ones(self.num_dim);
        (lb, ub)
    }

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

        
        tr.update(&array![1.0].view(), 1).unwrap();
        assert_eq!(tr.length(), 0.8);

        
        tr.update(&array![1.0, 2.0].view(), 2).unwrap();
        tr.update(&array![1.0, 2.0, 3.0].view(), 3).unwrap();
        tr.update(&array![1.0, 2.0, 3.0, 4.0].view(), 4).unwrap();

        assert!(tr.length() > 0.8); 
    }

    #[test]
    fn test_turbo_update_failure() {
        let config = TRLengthConfig::default();
        let mut tr = TurboTrustRegion::new(5, config);
        tr.set_num_arms(1);

        
        tr.update(&array![1.0].view(), 1).unwrap();

        
        let failure_tol = tr.failure_tolerance.unwrap();
        let mut y_all = vec![1.0];
        for _ in 0..failure_tol {
            y_all.push(0.5);
            let y_arr = ndarray::Array1::from_vec(y_all.clone());
            tr.update(&y_arr.view(), y_all.len()).unwrap();
        }

        assert!(tr.length() < 0.8); 
    }

    #[test]
    fn test_turbo_restart() {
        let config = TRLengthConfig::default();
        let mut tr = TurboTrustRegion::new(5, config);
        tr.set_num_arms(1);

        
        tr.update(&array![1.0].view(), 1).unwrap();
        let mut y_all = vec![1.0, 0.5];
        for _ in 0..10 {
            let y_arr = ndarray::Array1::from_vec(y_all.clone());
            tr.update(&y_arr.view(), y_all.len()).unwrap();
            y_all.push(0.5);
        }

        assert!(tr.length() < 0.8);

        
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

    #[test]
    fn improvement_bar_uses_abs_best_not_y_range() {
        let mut tr = TurboTrustRegion::new(2, TRLengthConfig::default());
        tr.set_num_arms(1);
        tr.seed_scale_history(&array![-1000.0, 10.0].view());
        tr.set_prev_num_obs(2);
        tr.update_with_incumbent_new_batch(&array![10.02].view(), 3, 10.02)
            .unwrap();
        assert_eq!(
            tr.success_counter, 1,
            "a gain of 0.02 beats 0.001 * |10| and would fail a range bar near 1010"
        );
    }

    #[test]
    fn seed_scale_history_uses_init_range_as_threshold() {
        let mut tr = TurboTrustRegion::new(2, TRLengthConfig::default());
        tr.set_num_arms(1);
        let prefix = array![-20.0, -5.0];
        tr.seed_scale_history(&prefix.view());
        tr.set_prev_num_obs(2);
        tr.update_with_incumbent_new_batch(&array![-4.0].view(), 3, -4.0)
            .unwrap();
        let length_before = tr.length();
        tr.update_with_incumbent_new_batch(&array![-3.999].view(), 4, -3.999)
            .unwrap();
        assert_eq!(tr.length(), length_before);
        assert_eq!(tr.failure_counter, 1);
    }

    #[test]
    fn restart_preserves_hist_scale_for_improvement() {
        
        
        
        let config = TRLengthConfig::default();
        let mut tr = TurboTrustRegion::new(2, config);
        tr.set_num_arms(1);

        let y0 = array![0.0, 100.0];
        tr.update_with_incumbent_new_batch(&y0.view(), 2, 100.0)
            .unwrap();
        assert_eq!(tr.hist_ymin, 0.0);
        assert_eq!(tr.hist_ymax, 100.0);

        let length_before = tr.length();
        tr.restart();
        assert_eq!(tr.prev_num_obs(), 2);
        assert_eq!(
            tr.hist_ymin, 0.0,
            "restart must preserve historical y min for scale"
        );
        assert_eq!(
            tr.hist_ymax, 100.0,
            "restart must preserve historical y max for scale"
        );

        
        let y1 = array![100.0];
        tr.update_with_incumbent_new_batch(&y1.view(), 3, 100.0)
            .unwrap();

        
        
        
        for (i, inc) in [100.05_f64, 100.10, 100.15].into_iter().enumerate() {
            let y = array![inc];
            tr.update_with_incumbent_new_batch(&y.view(), 4 + i, inc)
                .unwrap();
        }
        assert_eq!(
            tr.length(),
            length_before,
            "tiny post-restart gains must not expand length when hist scale is preserved"
        );
        assert_eq!(tr.success_counter, 0);
        assert!(tr.failure_counter > 0);
    }

    #[test]
    fn update_with_incumbent_new_batch_matches_full_history() {
        let config = TRLengthConfig::default();
        let mut full = TurboTrustRegion::new(2, config);
        let mut batch = TurboTrustRegion::new(2, config);
        full.set_num_arms(1);
        batch.set_num_arms(1);

        let batches: [&[f64]; 4] = [&[1.0], &[0.5, 0.4], &[2.0], &[1.9, 1.8, 2.1]];
        let incumbents = [1.0, 1.0, 2.0, 2.1];
        let mut y_all: Vec<f64> = Vec::new();
        for (b, &inc) in batches.iter().zip(incumbents.iter()) {
            y_all.extend_from_slice(b);
            let y_arr = ndarray::Array1::from_vec(y_all.clone());
            full
                .update_with_incumbent(&y_arr.view(), y_all.len(), inc)
                .unwrap();
            let y_new = ndarray::Array1::from_vec(b.to_vec());
            batch
                .update_with_incumbent_new_batch(&y_new.view(), y_all.len(), inc)
                .unwrap();
            assert_eq!(full.length(), batch.length());
            assert_eq!(full.prev_num_obs, batch.prev_num_obs);
            assert_eq!(full.success_counter, batch.success_counter);
            assert_eq!(full.failure_counter, batch.failure_counter);
        }
    }

    #[test]
    fn update_with_incumbent_restart_preserves_watermark_then_batch_matches_full() {
        
        let config = TRLengthConfig::default();
        let mut full = TurboTrustRegion::new(2, config);
        let mut mixed = TurboTrustRegion::new(2, config);
        full.set_num_arms(1);
        mixed.set_num_arms(1);

        let pre: [&[f64]; 2] = [&[1.0, 0.5], &[0.8]];
        let mut y_all: Vec<f64> = Vec::new();
        for (b, &inc) in pre.iter().zip([1.0_f64, 1.0].iter()) {
            y_all.extend_from_slice(b);
            let y_arr = ndarray::Array1::from_vec(y_all.clone());
            full
                .update_with_incumbent(&y_arr.view(), y_all.len(), inc)
                .unwrap();
            mixed
                .update_with_incumbent(&y_arr.view(), y_all.len(), inc)
                .unwrap();
        }

        let watermark = mixed.prev_num_obs();
        assert!(watermark > 0);
        full.restart();
        mixed.restart();
        assert_eq!(mixed.prev_num_obs(), watermark);
        assert_eq!(full.prev_num_obs(), watermark);

        let post: [&[f64]; 3] = [&[2.0], &[1.5, 2.2], &[2.3]];
        let post_inc = [2.0_f64, 2.2, 2.3];
        for (b, &inc) in post.iter().zip(post_inc.iter()) {
            y_all.extend_from_slice(b);
            let y_arr = ndarray::Array1::from_vec(y_all.clone());
            full
                .update_with_incumbent(&y_arr.view(), y_all.len(), inc)
                .unwrap();
            let y_new = ndarray::Array1::from_vec(b.to_vec());
            mixed
                .update_with_incumbent_new_batch(&y_new.view(), y_all.len(), inc)
                .unwrap();
            assert_eq!(full.length(), mixed.length());
            assert_eq!(full.prev_num_obs, mixed.prev_num_obs);
            assert_eq!(full.success_counter, mixed.success_counter);
            assert_eq!(full.failure_counter, mixed.failure_counter);
            assert_eq!(full.hist_ymin, mixed.hist_ymin);
            assert_eq!(full.hist_ymax, mixed.hist_ymax);
        }
    }

    #[test]
    fn tr_length_config_new_smoke() {
        let cfg = TRLengthConfig::new(0.8, 0.1, 1.0);
        assert!(cfg.length_init > 0.0);
    }
}
