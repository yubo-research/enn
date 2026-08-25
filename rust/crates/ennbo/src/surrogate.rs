//! Surrogate models for optimization.

use ndarray::{Array1, Array2, Array3, ArrayView2};
use rand::RngCore;
use rand::SeedableRng;

use std::path::PathBuf;

use crate::backend::EnnStorage;
use crate::error::ENNError;
use crate::fitter::ENNFitter;
use crate::index::IndexDriver;
use crate::model::EpistemicNearestNeighbors;
use crate::params::{ENNParams, PosteriorFlags};

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

    fn fit_append(
        &mut self,
        x_new: &ArrayView2<f64>,
        y_new: &ArrayView2<f64>,
        yvar_new: Option<&ArrayView2<f64>>,
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

    fn fitted_num_metrics(&self) -> Option<usize>;
    fn observation_count(&self) -> Option<usize>;
    fn observation_row_x(&self, idx: usize) -> Result<Array1<f64>, ENNError>;
    fn observation_row_y(&self, idx: usize) -> Result<Array1<f64>, ENNError>;
    fn observations_y(&self) -> Result<Option<Array2<f64>>, ENNError>;
    fn naturalize_observations_y(&self, y_warped: Array2<f64>) -> Array2<f64>;
    fn naturalize_prediction(&self, pred: SurrogatePrediction) -> SurrogatePrediction;
    fn warp_observations_y(&self, y: &ArrayView2<f64>) -> Result<Array2<f64>, ENNError>;
    fn observations_x(&self) -> Result<Option<Array2<f64>>, ENNError>;
    fn schedule_background_flush(&self) -> Result<(), ENNError>;
    fn wait_for_background_flush(&self) -> Result<(), ENNError>;
    fn release_observation_pages(&self) -> Result<(), ENNError>;
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
    pub storage: EnnStorage,
    pub work_dir: Option<PathBuf>,
    /// Optional per-metric natural-unit y bounds, shape `(num_metrics, 2)`.
    pub y_bounds: Option<Array2<f64>>,
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
            storage: EnnStorage::InMemory,
            work_dir: None,
            y_bounds: None,
        }
    }
}

pub struct ENNSurrogate {
    config: ENNSurrogateConfig,
    model: Option<EpistemicNearestNeighbors>,
    params: Option<ENNParams>,
    fitter: Option<ENNFitter>,
}

impl ENNSurrogate {
    pub fn new(config: ENNSurrogateConfig) -> Self {
        Self {
            config,
            model: None,
            params: None,
            fitter: None,
        }
    }

    pub fn model(&self) -> Option<&EpistemicNearestNeighbors> {
        self.model.as_ref()
    }

    pub fn params(&self) -> Option<&ENNParams> {
        self.params.as_ref()
    }

    fn construct_model(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Result<EpistemicNearestNeighbors, ENNError> {
        EpistemicNearestNeighbors::new_with_storage(
            x.to_owned(),
            y.to_owned(),
            yvar.map(|v| v.to_owned()),
            self.config.scale_x,
            self.config.index_driver,
            self.config.storage,
            self.config.work_dir.clone(),
            self.config.y_bounds.clone(),
        )
    }

    fn run_fitter(&mut self, rng: &mut rand::rngs::StdRng) -> Result<(), ENNError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| ENNError::InvalidParameter("Surrogate not fitted".to_string()))?;
        if self.fitter.is_none() {
            let n = model.len();
            let indices: Vec<usize> = (0..n).collect();
            
            let (train_x, train_y, train_yvar) = model.train_rows_at(&indices)?;
            let mut fitter = ENNFitter::new(self.config.k, self.config.infer_aleatoric_variance);
            let yvar_view = train_yvar.as_ref().map(|v| v.view());
            fitter.tell(
                &train_x.view(),
                &train_y.view(),
                yvar_view.as_ref(),
            )?;
            if let Some(p) = self.params {
                fitter.set_params(p);
            }
            self.fitter = Some(fitter);
        }
        let fitter = self.fitter.as_mut().expect("fitter");
        let p = fitter.ask(
            model,
            self.config.num_fit_candidates,
            self.config.num_fit_samples,
            self.params.as_ref(),
            rng,
        )?;
        self.params = Some(p);
        Ok(())
    }

    fn fit_append_internal(
        &mut self,
        x_new: &ArrayView2<f64>,
        y_new: &ArrayView2<f64>,
        yvar_new: Option<&ArrayView2<f64>>,
        rng: &mut rand::rngs::StdRng,
    ) -> Result<(), ENNError> {
        
        
        
        
        
        const BULK_DISK_TELL_SKIP_FIT_ROWS: usize = 4_096;
        if let Some(model) = &mut self.model {
            model.add(x_new, y_new, yvar_new)?;
            if let Some(fitter) = self.fitter.as_mut() {
                
                fitter.tell(x_new, y_new, yvar_new)?;
            }
            let is_disk = model.backend_driver() == IndexDriver::BpAnnDisk;
            let bulk_disk = is_disk && x_new.nrows() >= BULK_DISK_TELL_SKIP_FIT_ROWS;
            
            
            let skip_fit = bulk_disk || (is_disk && self.params.is_some());
            
            
            if !skip_fit || bulk_disk {
                model.ensure_index_sync()?;
            }
            if !skip_fit {
                self.run_fitter(rng)?;
            }
            if !skip_fit || bulk_disk {
                if let Some(model) = &self.model {
                    model.index_access().release_observation_pages()?;
                }
            }
            return Ok(());
        }
        let mut fitter = ENNFitter::new(self.config.k, self.config.infer_aleatoric_variance);
        
        let model = self.construct_model(x_new, y_new, yvar_new)?;
        fitter.tell(x_new, y_new, yvar_new)?;
        self.model = Some(model);
        self.fitter = Some(fitter);
        let skip_fit = self.config.index_driver == IndexDriver::BpAnnDisk
            && x_new.nrows() >= BULK_DISK_TELL_SKIP_FIT_ROWS;
        if !skip_fit {
            
            if self.config.index_driver == IndexDriver::BpAnnDisk {
                if let Some(model) = &self.model {
                    model.ensure_index_sync()?;
                }
            }
            self.run_fitter(rng)?;
            if self.config.index_driver == IndexDriver::BpAnnDisk {
                if let Some(model) = &self.model {
                    model.index_access().release_observation_pages()?;
                }
            }
        }
        Ok(())
    }
}

impl Surrogate for ENNSurrogate {
    fn fitted_num_metrics(&self) -> Option<usize> {
        self.model.as_ref().map(|m| m.num_metrics())
    }

    fn observation_count(&self) -> Option<usize> {
        self.model.as_ref().map(|m| m.len())
    }

    fn observation_row_x(&self, idx: usize) -> Result<Array1<f64>, ENNError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| ENNError::InvalidParameter("Surrogate not fitted".to_string()))?;
        model.rows().row_x(idx)
    }

    fn observation_row_y(&self, idx: usize) -> Result<Array1<f64>, ENNError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| ENNError::InvalidParameter("Surrogate not fitted".to_string()))?;
        
        model.row_y_natural(idx)
    }

    fn observations_y(&self) -> Result<Option<Array2<f64>>, ENNError> {
        let model = match self.model.as_ref() {
            Some(m) => m,
            None => return Ok(None),
        };
        let n = model.len();
        if n == 0 {
            return Ok(None);
        }
        
        
        let mut y = Array2::zeros((n, model.num_metrics()));
        for i in 0..n {
            y.row_mut(i).assign(&model.rows().row_y(i)?);
        }
        Ok(Some(y))
    }

    fn naturalize_observations_y(&self, y_warped: Array2<f64>) -> Array2<f64> {
        let Some(model) = self.model.as_ref() else {
            return y_warped;
        };
        if crate::y_bounds::is_identity_bounds(model.y_bounds()) {
            return y_warped;
        }
        crate::y_bounds::inv_y(y_warped.view(), model.y_bounds())
    }

    fn naturalize_prediction(&self, mut pred: SurrogatePrediction) -> SurrogatePrediction {
        let Some(model) = self.model.as_ref() else {
            return pred;
        };
        let bounds = model.y_bounds();
        if crate::y_bounds::is_identity_bounds(bounds) {
            return pred;
        }
        let mut se_epi = pred.se.clone();
        let mut se_ale = pred.se.clone();
        crate::y_bounds::naturalize_mu_se(
            &mut pred.mu,
            &mut pred.se,
            &mut se_epi,
            &mut se_ale,
            bounds,
        );
        pred
    }

    fn warp_observations_y(&self, y: &ArrayView2<f64>) -> Result<Array2<f64>, ENNError> {
        if let Some(model) = self.model.as_ref() {
            let (yz, _) = model.warp_observations(y, None)?;
            return Ok(yz);
        }
        let bounds = match &self.config.y_bounds {
            Some(b) => b.clone(),
            None => crate::y_bounds::unbounded_bounds(y.ncols()),
        };
        crate::y_bounds::validate_bounds(&bounds, y.ncols())?;
        crate::y_bounds::warp_y(*y, &bounds)
    }

    fn observations_x(&self) -> Result<Option<Array2<f64>>, ENNError> {
        let model = match self.model.as_ref() {
            Some(m) => m,
            None => return Ok(None),
        };
        let n = model.len();
        if n == 0 {
            return Ok(None);
        }
        let mut x = Array2::zeros((n, model.num_dim()));
        for i in 0..n {
            x.row_mut(i).assign(&model.rows().row_x(i)?);
        }
        Ok(Some(x))
    }

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

        let model = self.construct_model(x, y, yvar)?;

        let mut fitter = ENNFitter::new(self.config.k, self.config.infer_aleatoric_variance);
        fitter.tell(x, y, yvar)?;
        if let Some(p) = self.params {
            fitter.set_params(p);
        }
        let p = fitter.ask(
            &model,
            self.config.num_fit_candidates,
            self.config.num_fit_samples,
            self.params.as_ref(),
            &mut local_rng,
        )?;
        self.params = Some(p);
        self.model = Some(model);
        self.fitter = Some(fitter);

        Ok(())
    }

    fn fit_append(
        &mut self,
        x_new: &ArrayView2<f64>,
        y_new: &ArrayView2<f64>,
        yvar_new: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<(), ENNError> {
        let mut seed_bytes = [0u8; 32];
        rng.fill_bytes(&mut seed_bytes);
        let mut local_rng = rand::rngs::StdRng::from_seed(seed_bytes);
        self.fit_append_internal(x_new, y_new, yvar_new, &mut local_rng)
    }

    fn schedule_background_flush(&self) -> Result<(), ENNError> {
        if let Some(model) = &self.model {
            model.backend.schedule_background_flush()
        } else {
            Ok(())
        }
    }

    fn wait_for_background_flush(&self) -> Result<(), ENNError> {
        if let Some(model) = &self.model {
            
            
            
            model.backend.wait_for_flush()
        } else {
            Ok(())
        }
    }

    fn release_observation_pages(&self) -> Result<(), ENNError> {
        if let Some(model) = &self.model {
            model.index_access().release_observation_pages()
        } else {
            Ok(())
        }
    }

    fn predict(&self, x: &ArrayView2<f64>) -> Result<SurrogatePrediction, ENNError> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| ENNError::InvalidParameter("Surrogate not fitted".to_string()))?;
        let params = match self.params.as_ref() {
            Some(p) => *p,
            None => {
                
                
                ENNParams::new(self.config.k, 1.0, 0.0).map_err(|e| {
                    ENNError::InvalidParameter(format!("Failed to create default params: {e}"))
                })?
            }
        };

        let flags = PosteriorFlags::new();
        let posterior = model.posterior_warped(x, &params, &flags)?;

        
        let mu = posterior
            .mu
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| ENNError::InvalidParameter(format!("Shape error: {}", e)))?;
        let se = posterior
            .se
            .into_dimensionality::<ndarray::Ix2>()
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
        let params = match self.params.as_ref() {
            Some(p) => *p,
            None => {
                
                
                ENNParams::new(self.config.k, 1.0, 0.0).map_err(|e| {
                    ENNError::InvalidParameter(format!("Failed to create default params: {e}"))
                })?
            }
        };

        
        let mut seed_bytes = [0u8; 8];
        rng.fill_bytes(&mut seed_bytes);
        let base_seed = u64::from_le_bytes(seed_bytes) as i64;
        let function_seeds: Vec<i64> = (0..num_samples as i64).map(|i| base_seed + i).collect();

        let (draws, _) =
            model.posterior_function_draw_warped(x, &params, &function_seeds, &Default::default())?;

        Ok(draws)
    }

    fn lengthscales(&self) -> Option<Array1<f64>> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

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

        
        assert!(surrogate.model().is_some());
        assert!(surrogate.params().is_some());

        
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
        let v_inc = sur_inc.model().unwrap().rows().row_yvar(0).unwrap().unwrap()[[0]];

        let mut rng_b = StdRng::seed_from_u64(11);
        let mut sur_full = ENNSurrogate::new(config);
        sur_full
            .fit(&x1.view(), &y1.view(), Some(&yvar1.view()), &mut rng_b)
            .unwrap();
        let v_full = sur_full.model().unwrap().rows().row_yvar(0).unwrap().unwrap()[[0]];

        assert!(
            (v_inc - v_full).abs() < 1e-9,
            "train_yvar row0 incremental={v_inc} full_refit={v_full} (prefix yvar must refresh)"
        );
    }

    #[test]
    fn regression_incremental_fit_rejects_nan_y_on_append() {
        let config = ENNSurrogateConfig {
            k: 2,
            num_fit_candidates: 4,
            num_fit_samples: 2,
            ..Default::default()
        };
        let x0 = array![[0.0, 0.0], [1.0, 0.0]];
        let y0 = array![[0.0], [1.0]];
        let x1 = array![[0.0, 0.0], [1.0, 0.0], [0.5, 0.5]];
        let y1 = array![[0.0], [1.0], [f64::NAN]];

        let mut sur = ENNSurrogate::new(config);
        let mut rng = StdRng::seed_from_u64(42);
        sur.fit(&x0.view(), &y0.view(), None, &mut rng).unwrap();
        let result = sur.fit(&x1.view(), &y1.view(), None, &mut rng);
        assert!(
            result.is_err(),
            "non-finite y on incremental append must be rejected (use tell)"
        );
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

    #[test]
    fn kiss_surrogate_config_default() {
        let cfg = ENNSurrogateConfig::default();
        assert!(cfg.k >= 1);
    }

    #[test]
    fn fit_tells_fitter_natural_y_under_y_bounds() {
        
        let x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]];
        let y = array![[0.1], [0.2], [0.8], [0.9]];
        let bounds = array![[0.0, 1.0]];
        let mut fit_nat = crate::fitter::ENNFitter::new(2, true);
        fit_nat.tell(&x.view(), &y.view(), None).unwrap();
        let nat_std = fit_nat.y_std()[0];
        let y_z = crate::y_bounds::warp_y(y.view(), &bounds).unwrap();
        let mut fit_z = crate::fitter::ENNFitter::new(2, true);
        fit_z.tell(&x.view(), &y_z.view(), None).unwrap();
        let z_std = fit_z.y_std()[0];
        
        assert!(
            (nat_std - 0.35355).abs() < 0.01,
            "natural y_std={nat_std}"
        );
        assert!(z_std > 1.5, "warped y_std={z_std}");

        let config = ENNSurrogateConfig {
            k: 2,
            num_fit_candidates: 4,
            num_fit_samples: 4,
            y_bounds: Some(bounds),
            ..Default::default()
        };
        let mut sur = ENNSurrogate::new(config);
        let mut rng = StdRng::seed_from_u64(7);
        sur.fit(&x.view(), &y.view(), None, &mut rng).unwrap();
        let fitter_std = sur.fitter.as_ref().expect("fitter").y_std()[0];
        assert!(
            (fitter_std - nat_std).abs() < 1e-9,
            "surrogate fitter must be told natural y: got {fitter_std} want {nat_std} (warped would be {z_std})"
        );
    }

    #[test]
    fn fit_append_disk_drains_pending_before_search() {
        use crate::backend::EnnStorage;
        use crate::index::IndexDriver;
        use ndarray::Array2;
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        let config = ENNSurrogateConfig {
            k: 2,
            num_fit_candidates: 2,
            num_fit_samples: 2,
            index_driver: IndexDriver::BpAnnDisk,
            storage: EnnStorage::Disk,
            work_dir: Some(dir.path().to_path_buf()),
            scale_x: false,
            ..Default::default()
        };
        let mut sur = ENNSurrogate::new(config);
        let mut rng = StdRng::seed_from_u64(7);
        let x0 = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let y0 = array![[0.0], [1.0], [0.5]];
        sur.fit(&x0.view(), &y0.view(), None, &mut rng).unwrap();

        let x1 = Array2::from_shape_fn((4_096, 2), |(i, j)| (i + j) as f64 * 0.001);
        let y1 = Array2::from_shape_fn((4_096, 1), |(i, _)| (i as f64) * 0.01);
        sur.fit_append(&x1.view(), &y1.view(), None, &mut rng)
            .unwrap();

        let model = sur.model().expect("model");
        
        
        let indexed = std::fs::read(dir.path().join("indexed_rows.bin"))
            .ok()
            .and_then(|b| {
                if b.len() >= 8 {
                    Some(u64::from_le_bytes(b[..8].try_into().ok()?) as usize)
                } else {
                    None
                }
            })
            .unwrap_or(0);
        assert_eq!(
            indexed, model.len(),
            "indexed_rows.bin must match num_obs after fit_append sync-before-fit"
        );
    }

    #[test]
    fn fit_append_disk_streaming_skips_sync_when_params_exist() {
        use crate::backend::EnnStorage;
        use crate::index::IndexDriver;
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        let config = ENNSurrogateConfig {
            k: 2,
            num_fit_candidates: 2,
            num_fit_samples: 2,
            index_driver: IndexDriver::BpAnnDisk,
            storage: EnnStorage::Disk,
            work_dir: Some(dir.path().to_path_buf()),
            scale_x: false,
            ..Default::default()
        };
        let mut sur = ENNSurrogate::new(config);
        let mut rng = StdRng::seed_from_u64(11);
        let x0 = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let y0 = array![[0.0], [1.0], [0.5]];
        sur.fit(&x0.view(), &y0.view(), None, &mut rng).unwrap();
        assert!(sur.params().is_some(), "initial fit must set params");

        
        let before = std::fs::metadata(dir.path().join("indexed_rows.bin"))
            .ok()
            .map(|m| m.len());
        for i in 0..20 {
            let x = array![[i as f64 * 0.01, 0.1]];
            let y = array![[i as f64 * 0.1]];
            sur.fit_append(&x.view(), &y.view(), None, &mut rng)
                .unwrap();
        }
        let after = std::fs::metadata(dir.path().join("indexed_rows.bin"))
            .ok()
            .map(|m| m.len());
        assert_eq!(
            before, after,
            "streaming one-row disk fit_append must not rewrite indexed_rows.bin"
        );
        assert_eq!(sur.model().expect("model").len(), 23);

        
        sur.wait_for_background_flush().unwrap();
        let after_wait = std::fs::metadata(dir.path().join("indexed_rows.bin"))
            .ok()
            .map(|m| m.len());
        assert_eq!(
            before, after_wait,
            "wait_for_background_flush must not force ensure_index_sync"
        );
    }
}
