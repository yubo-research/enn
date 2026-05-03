//! Surrogate models for optimization.

use ndarray::{s, Array1, Array2, Array3, ArrayView2};
use rand::RngCore;
use rand::SeedableRng;

use crate::error::ENNError;
use crate::fit::enn_fit;
use crate::index::IndexDriver;
use crate::model::EpistemicNearestNeighbors;
use crate::params::ENNParams;
use crate::traits::PosteriorComputation;

#[derive(Debug, Clone)]
pub struct SurrogatePrediction {
    pub mu: Array2<f64>,
    pub se: Array2<f64>,
}

pub trait Surrogate: Send + Sync {
    fn fit(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<(), ENNError>;

    fn predict(&self, x: &ArrayView2<f64>) -> Result<SurrogatePrediction, ENNError>;

    fn sample(
        &self,
        x: &ArrayView2<f64>,
        num_samples: usize,
        rng: &mut dyn RngCore,
    ) -> Result<Array3<f64>, ENNError>;

    fn lengthscales(&self) -> Option<Array1<f64>>;

    fn get_incumbent_indices(&self, y_obs: &ArrayView2<f64>) -> Vec<usize>;
}

pub type BoxedSurrogate = Box<dyn Surrogate + Send + Sync>;

#[derive(Debug, Clone)]
pub struct ENNSurrogateConfig {
    pub k: i32,
    pub scale_x: bool,
    pub num_fit_candidates: usize,
    pub num_fit_samples: usize,
    pub infer_aleatoric_variance: bool,
    pub index_driver: IndexDriver,
}

impl Default for ENNSurrogateConfig {
    fn default() -> Self {
        Self {
            k: 10,
            scale_x: false,
            num_fit_candidates: 30,
            num_fit_samples: 10,
            infer_aleatoric_variance: true,
            index_driver: IndexDriver::Exact,
        }
    }
}

pub struct ENNSurrogate {
    config: ENNSurrogateConfig,
    model: Option<EpistemicNearestNeighbors>,
    params: Option<ENNParams>,
}

impl ENNSurrogate {
    pub fn new(config: ENNSurrogateConfig) -> Self {
        Self {
            config,
            model: None,
            params: None,
        }
    }

    pub fn model(&self) -> Option<&EpistemicNearestNeighbors> {
        self.model.as_ref()
    }

    pub fn params(&self) -> Option<&ENNParams> {
        self.params.as_ref()
    }

    fn try_incremental_fit(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut rand::rngs::StdRng,
    ) -> Result<bool, ENNError> {
        let Some(ref mut model) = self.model else {
            return Ok(false);
        };
        let n_old = model.len();
        let n_new = x.nrows();
        if n_new <= n_old {
            return Ok(false);
        }

        let prefix_x = x.slice(s![..n_old, ..]);
        let prefix_y = y.slice(s![..n_old, ..]);
        let same_prefix = prefix_x
            .iter()
            .zip(model.train_x.iter())
            .all(|(a, b)| (a - b).abs() < 1e-12)
            && prefix_y
                .iter()
                .zip(model.train_y.iter())
                .all(|(a, b)| (a - b).abs() < 1e-12);

        let same_yvar_prefix = match (model.train_yvar(), yvar) {
            (None, None) => true,
            (Some(_), None) | (None, Some(_)) => false,
            (Some(m_yv), Some(in_yv)) => {
                if in_yv.nrows() < n_old {
                    false
                } else {
                    let p = in_yv.slice(s![..n_old, ..]);
                    p.shape() == m_yv.shape()
                        && p.iter().zip(m_yv.iter()).all(|(a, b)| (a - b).abs() < 1e-12)
                }
            }
        };

        if !same_prefix || !same_yvar_prefix {
            return Ok(false);
        }

        let x_add = x.slice(s![n_old.., ..]);
        let y_add = y.slice(s![n_old.., ..]);
        let yvar_add = yvar.map(|v| v.slice(s![n_old.., ..]));

        model.add(&x_add, &y_add, yvar_add.as_ref())?;

        let params = enn_fit(
            model,
            self.config.k,
            self.config.num_fit_candidates,
            self.config.num_fit_samples,
            rng,
            self.params.as_ref(),
            self.config.infer_aleatoric_variance,
        )?;

        self.params = Some(params);
        Ok(true)
    }
}

impl Surrogate for ENNSurrogate {
    fn fit(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<(), ENNError> {
        let mut seed_bytes = [0u8; 32];
        rng.fill_bytes(&mut seed_bytes);
        let mut local_rng = rand::rngs::StdRng::from_seed(seed_bytes);

        let use_incremental = self.try_incremental_fit(x, y, yvar, &mut local_rng)?;
        if use_incremental {
            return Ok(());
        }

        let model = EpistemicNearestNeighbors::new(
            x.to_owned(),
            y.to_owned(),
            yvar.map(|v| v.to_owned()),
            self.config.scale_x,
            self.config.index_driver,
        )?;

        let params = enn_fit(
            &model,
            self.config.k,
            self.config.num_fit_candidates,
            self.config.num_fit_samples,
            &mut local_rng,
            self.params.as_ref(),
            self.config.infer_aleatoric_variance,
        )?;

        self.model = Some(model);
        self.params = Some(params);

        Ok(())
    }

    fn predict(&self, x: &ArrayView2<f64>) -> Result<SurrogatePrediction, ENNError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| ENNError::InvalidParameter("Surrogate not fitted".to_string()))?;
        let params = self
            .params
            .as_ref()
            .ok_or_else(|| ENNError::InvalidParameter("Surrogate not fitted".to_string()))?;

        let posterior = model.posterior(x, params, &Default::default())?;

        // Convert from dynamic dimension to fixed 2D
        let mu = posterior.mu.into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| ENNError::InvalidParameter(format!("Shape error: {}", e)))?;
        let se = posterior.se.into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| ENNError::InvalidParameter(format!("Shape error: {}", e)))?;

        Ok(SurrogatePrediction { mu, se })
    }

    fn sample(
        &self,
        x: &ArrayView2<f64>,
        num_samples: usize,
        rng: &mut dyn RngCore,
    ) -> Result<Array3<f64>, ENNError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| ENNError::InvalidParameter("Surrogate not fitted".to_string()))?;
        let params = self
            .params
            .as_ref()
            .ok_or_else(|| ENNError::InvalidParameter("Surrogate not fitted".to_string()))?;

        // Use deterministic seeds based on current state
        let mut seed_bytes = [0u8; 8];
        rng.fill_bytes(&mut seed_bytes);
        let base_seed = u64::from_le_bytes(seed_bytes) as i64;
        let function_seeds: Vec<i64> = (0..num_samples as i64).map(|i| base_seed + i).collect();

        let (draws, _) = model.posterior_function_draw(x, params, &function_seeds, &Default::default())?;

        Ok(draws)
    }

    fn lengthscales(&self) -> Option<Array1<f64>> {
        None
    }

    fn get_incumbent_indices(&self, y_obs: &ArrayView2<f64>) -> Vec<usize> {
        let k = self.config.k as usize;
        let n = y_obs.nrows();

        if y_obs.ncols() > 1 {
            let num_top = k.min(n);
            let mut union_indices: std::collections::HashSet<usize> = std::collections::HashSet::new();

            for m in 0..y_obs.ncols() {
                let col: Vec<f64> = y_obs.column(m).to_vec();
                let mut indexed: Vec<(usize, f64)> = col.into_iter().enumerate().collect();
                indexed.sort_by(|a, b| b.1.total_cmp(&a.1));

                for item in indexed.iter().take(num_top) {
                    union_indices.insert(item.0);
                }
            }

            let mut result: Vec<usize> = union_indices.into_iter().collect();
            result.sort();
            result
        } else {
            let y_flat: Vec<f64> = if y_obs.ndim() == 2 {
                y_obs.column(0).to_vec()
            } else {
                y_obs.iter().copied().collect()
            };

            let num_top = k.min(y_flat.len());
            let mut indexed: Vec<(usize, f64)> = y_flat.into_iter().enumerate().collect();
            indexed.sort_by(|a, b| b.1.total_cmp(&a.1));

            indexed.iter().take(num_top).map(|(i, _)| *i).collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn test_enn_surrogate_fit_predict() {
        let config = ENNSurrogateConfig {
            k: 2,
            num_fit_candidates: 5,
            num_fit_samples: 3,
            ..Default::default()
        };
        let mut surrogate = ENNSurrogate::new(config);

        let x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let y = array![[0.0], [1.0], [1.0], [2.0]];

        let mut rng = StdRng::seed_from_u64(42);
        surrogate.fit(&x.view(), &y.view(), None, &mut rng).unwrap();

        // Check model is fitted
        assert!(surrogate.model().is_some());
        assert!(surrogate.params().is_some());

        // Predict
        let x_query = array![[0.5, 0.5]];
        let pred = surrogate.predict(&x_query.view()).unwrap();
        assert_eq!(pred.mu.shape(), &[1, 1]);
        assert!(pred.mu[[0, 0]].is_finite());
    }

    /// Regression: incremental `fit` must not reuse stale `train_yvar` when the caller
    /// updates observation noise on prefix rows (same `x`/`y`, new `yvar` on old rows).
    #[test]
    fn regression_incremental_fit_refreshes_prefix_yvar_to_match_full_refit() {
        let config = ENNSurrogateConfig {
            k: 2,
            num_fit_candidates: 4,
            num_fit_samples: 2,
            ..Default::default()
        };
        let x0 = array![[0.0, 0.0], [1.0, 0.0]];
        let y0 = array![[0.0], [1.0]];
        let yvar0 = array![[1.0], [2.0]];
        let x1 = array![[0.0, 0.0], [1.0, 0.0], [3.0, 0.0]];
        let y1 = array![[0.0], [1.0], [5.0]];
        let yvar1 = array![[1.0e6], [2.0], [1.0]];

        let mut rng_a = StdRng::seed_from_u64(11);
        let mut sur_inc = ENNSurrogate::new(config.clone());
        sur_inc
            .fit(&x0.view(), &y0.view(), Some(&yvar0.view()), &mut rng_a)
            .unwrap();
        sur_inc
            .fit(&x1.view(), &y1.view(), Some(&yvar1.view()), &mut rng_a)
            .unwrap();
        let v_inc = sur_inc.model().unwrap().train_yvar().unwrap()[[0, 0]];

        let mut rng_b = StdRng::seed_from_u64(11);
        let mut sur_full = ENNSurrogate::new(config);
        sur_full
            .fit(&x1.view(), &y1.view(), Some(&yvar1.view()), &mut rng_b)
            .unwrap();
        let v_full = sur_full.model().unwrap().train_yvar().unwrap()[[0, 0]];

        assert!(
            (v_inc - v_full).abs() < 1e-9,
            "train_yvar row0 incremental={v_inc} full_refit={v_full} (prefix yvar must refresh)"
        );
    }

    #[test]
    fn test_get_incumbent_indices_single() {
        let config = ENNSurrogateConfig {
            k: 2,
            ..Default::default()
        };
        let surrogate = ENNSurrogate::new(config);

        let y = array![[0.0], [3.0], [1.0], [2.0]];
        let indices = surrogate.get_incumbent_indices(&y.view());

        // Should return top 2 indices by value
        assert_eq!(indices.len(), 2);
        assert!(indices.contains(&1)); // 3.0
    }

    #[test]
    fn test_get_incumbent_indices_multi() {
        let config = ENNSurrogateConfig {
            k: 3,
            ..Default::default()
        };
        let surrogate = ENNSurrogate::new(config);

        // Multi-objective
        let y = array![[0.0, 3.0], [3.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let indices = surrogate.get_incumbent_indices(&y.view());

        // Should return union of top-k per objective
        assert!(indices.len() >= 2);
        assert!(indices.contains(&0) || indices.contains(&1)); // top of each objective
    }

    #[test]
    fn test_surrogate_prediction_clone() {
        let pred = SurrogatePrediction {
            mu: array![[1.0], [2.0]],
            se: array![[0.1], [0.2]],
        };
        let cloned = pred.clone();
        assert_eq!(cloned.mu.shape(), &[2, 1]);
        assert_eq!(cloned.se.shape(), &[2, 1]);
        assert_eq!(cloned.mu[[1, 0]], 2.0);
    }
}
