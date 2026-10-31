
use ndarray::{Array2, ArrayD};
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum ParamsError {
    #[error("k_num_neighbors must be > 0, got {0}")]
    InvalidK(i32),
    #[error("epistemic_variance_scale must be >= 0 and finite, got {0}")]
    InvalidEpistemicVariance(f64),
    #[error("aleatoric_variance_scale must be >= 0 and finite, got {0}")]
    InvalidAleatoricVariance(f64),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ENNParams {
    pub k_num_neighbors: i32,
    pub epistemic_variance_scale: f64,
    pub aleatoric_variance_scale: f64,
}

impl ENNParams {
    pub fn new(
        k_num_neighbors: i32,
        epistemic_variance_scale: f64,
        aleatoric_variance_scale: f64,
    ) -> Result<Self, ParamsError> {

        if k_num_neighbors <= 0 {
            return Err(ParamsError::InvalidK(k_num_neighbors));
        }


        if !epistemic_variance_scale.is_finite() || epistemic_variance_scale < 0.0 {
            return Err(ParamsError::InvalidEpistemicVariance(
                epistemic_variance_scale,
            ));
        }


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

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct PosteriorFlags {
    pub exclude_nearest: bool,
    pub observation_noise: bool,
}

impl PosteriorFlags {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_exclude_nearest(mut self, value: bool) -> Self {
        self.exclude_nearest = value;
        self
    }

    pub fn with_observation_noise(mut self, value: bool) -> Self {
        self.observation_noise = value;
        self
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ENNNormal {
    pub mu: ArrayD<f64>,
    pub se: ArrayD<f64>,
    pub se_epi: ArrayD<f64>,
    pub se_ale: ArrayD<f64>,
    pub idx: Option<Array2<i64>>,
    pub y_bounds: Option<Array2<f64>>,
}

impl ENNNormal {
    pub fn new(
        mu: ArrayD<f64>,
        se: ArrayD<f64>,
        se_epi: ArrayD<f64>,
        se_ale: ArrayD<f64>,
        idx: Option<Array2<i64>>,
    ) -> Self {
        Self {
            mu,
            se,
            se_epi,
            se_ale,
            idx,
            y_bounds: None,
        }
    }

    pub fn with_y_bounds(mut self, y_bounds: Array2<f64>) -> Self {
        self.y_bounds = Some(y_bounds);
        self
    }

    pub fn sample(
        &self,
        num_samples: usize,
        seed: u64,
        clip: Option<f64>,
    ) -> Result<ArrayD<f64>, crate::error::ENNError> {
        crate::normal_sample::sample_normal(
            &self.mu,
            &self.se,
            self.y_bounds.as_ref(),
            num_samples,
            seed,
            clip,
        )
    }

    pub fn confidence_interval(
        &self,
        level: f64,
    ) -> Result<(ArrayD<f64>, ArrayD<f64>), crate::error::ENNError> {
        crate::normal_sample::confidence_interval(
            &self.mu,
            &self.se,
            self.y_bounds.as_ref(),
            level,
        )
    }

    pub fn len(&self) -> usize {
        self.mu.len()
    }

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
        let se_epi = se.clone();
        let se_ale = array![[0.0, 0.0], [0.0, 0.0]].into_dyn();
        let normal = ENNNormal::new(mu.clone(), se.clone(), se_epi, se_ale, None);

        assert_eq!(normal.len(), 4);
        assert!(!normal.is_empty());
        assert_eq!(normal.mu, mu);
        assert_eq!(normal.se, se);
    }

    #[test]
    fn test_enn_normal_with_idx() {
        let mu = array![[1.0], [2.0]].into_dyn();
        let se = array![[0.1], [0.2]].into_dyn();
        let se_epi = se.clone();
        let se_ale = array![[0.0], [0.0]].into_dyn();
        let idx = Some(array![[0, 1], [1, 2]]);
        let normal = ENNNormal::new(mu, se, se_epi, se_ale, idx.clone());

        assert_eq!(normal.idx, idx);
    }

    #[test]
    fn test_enn_normal_empty() {
        let mu = ArrayD::<f64>::zeros(IxDyn(&[0, 0]));
        let se = ArrayD::<f64>::zeros(IxDyn(&[0, 0]));
        let se_epi = se.clone();
        let se_ale = se.clone();
        let normal = ENNNormal::new(mu, se, se_epi, se_ale, None);

        assert!(normal.is_empty());
        assert_eq!(normal.len(), 0);
    }
}
