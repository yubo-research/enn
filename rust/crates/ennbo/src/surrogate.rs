//! Surrogate models for optimization.

use ndarray::{Array1, Array2, Array3, ArrayView2};
use rand::RngCore;
use rand::SeedableRng;

use crate::error::ENNError;
use crate::fit_samples::FitSamples;
use crate::fitter::ENNFitter;
use crate::index::IndexDriver;
use crate::layout::EnnLayout;
use crate::metric_auto::MetricLearning;
use crate::model::EpistemicNearestNeighbors;
use crate::neighbor_count::NeighborCount;
use crate::params::{ENNParams, PosteriorFlags};
use crate::surrogate_state::{AppendInput, AppendPlan, FitState, ScaleFit};

fn enable_auto_if_configured(
    model: &mut EpistemicNearestNeighbors,
    config: &ENNSurrogateConfig,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
) -> Result<(), ENNError> {
    if config.layout.metric_learning() == MetricLearning::Auto {
        model.enable_auto_metric(config.tied_dims.clone(), x, y)?;
    }
    Ok(())
}

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

    /// Posterior mean and standard error in natural `y` units.
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
    fn warp_observations_y(&self, y: &ArrayView2<f64>) -> Result<Array2<f64>, ENNError>;
    fn observations_x(&self) -> Result<Option<Array2<f64>>, ENNError>;
    fn schedule_background_flush(&self) -> Result<(), ENNError>;
    fn wait_for_background_flush(&self) -> Result<(), ENNError>;
    fn release_observation_pages(&self) -> Result<(), ENNError>;
    /// Drop fitted rows so the next `fit_append` starts a new local dataset.
    fn clear_observations(&mut self) -> Result<(), ENNError>;
}

pub type BoxedSurrogate = Box<dyn Surrogate + Send + Sync>;

#[derive(Debug, Clone)]
pub struct ENNSurrogateConfig {
    pub k: NeighborCount,
    /// Scale-search settings, or frozen scales.
    pub fit_samples: FitSamples,
    /// Index, storage, and metric. Illegal pairs are not a variant.
    pub layout: EnnLayout,
    /// Optional per-metric natural-unit y bounds, shape `(num_metrics, 2)`.
    pub y_bounds: Option<Array2<f64>>,
    /// Groups of tied input dimensions. Only `MetricLearning::Auto` reads them.
    pub tied_dims: Vec<Vec<usize>>,
}

impl Default for ENNSurrogateConfig {
    fn default() -> Self {
        Self {
            k: NeighborCount::default(),
            fit_samples: FitSamples::default(),
            layout: EnnLayout::memory(IndexDriver::Flat, false),
            y_bounds: None,
            tied_dims: Vec::new(),
        }
    }
}

impl ENNSurrogateConfig {
    /// Reject `tied_dims` that would be ignored or that do not fit `num_dim` inputs.
    pub fn validate(&self, num_dim: usize) -> Result<(), ENNError> {
        if self.tied_dims.is_empty() {
            return Ok(());
        }
        if self.layout.metric_learning() != MetricLearning::Auto {
            return Err(ENNError::InvalidParameter(
                "tied_dims require metric_learning=Auto".into(),
            ));
        }
        crate::metric_weights::validate_tied_dims(&self.tied_dims, num_dim)
    }

    fn default_params(&self) -> Result<ENNParams, ENNError> {
        ENNParams::new(self.k.get(), 1.0, 0.0).map_err(|e| {
            ENNError::InvalidParameter(format!("Failed to create default params: {e}"))
        })
    }

}

pub struct ENNSurrogate {
    config: ENNSurrogateConfig,
    state: FitState,
}

impl ENNSurrogate {
    pub fn new(config: ENNSurrogateConfig) -> Self {
        Self {
            config,
            state: FitState::Unfitted,
        }
    }

    pub fn model(&self) -> Option<&EpistemicNearestNeighbors> {
        self.state.model()
    }

    pub fn params(&self) -> Option<&ENNParams> {
        self.state.params()
    }

    fn fitted_model(&self) -> Result<&EpistemicNearestNeighbors, ENNError> {
        self.state
            .model()
            .ok_or_else(|| ENNError::InvalidParameter("Surrogate not fitted".to_string()))
    }

    fn params_or_default(&self) -> Result<ENNParams, ENNError> {
        match self.state.params() {
            Some(p) => Ok(*p),
            None => self.config.default_params(),
        }
    }

    fn on_disk(&self) -> bool {
        self.config.layout.index_driver() == IndexDriver::BpAnnDisk
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
            self.config.layout.clone(),
            self.config.y_bounds.clone(),
        )
    }

    fn search_scales(&mut self, rng: &mut rand::rngs::StdRng) -> Result<(), ENNError> {
        if let FitState::Searched { model, fit } = &mut self.state {
            let ScaleFit {
                search,
                fitter,
                params,
                calibrator,
            } = fit.as_mut();
            *params = Some(fitter.ask(model, search, params.as_ref(), rng)?);
            *calibrator = fitter.calibrator().cloned();
        }
        Ok(())
    }

    fn apply_plan(&mut self, plan: AppendPlan, rng: &mut rand::rngs::StdRng) -> Result<(), ENNError> {
        if plan.sync {
            self.fitted_model()?.ensure_index_sync()?;
        }
        if plan.search {
            self.search_scales(rng)?;
        }
        if plan.release {
            self.fitted_model()?.index_access().release_observation_pages()?;
        }
        Ok(())
    }

    fn fit_append_internal(
        &mut self,
        x_new: &ArrayView2<f64>,
        y_new: &ArrayView2<f64>,
        yvar_new: Option<&ArrayView2<f64>>,
        rng: &mut rand::rngs::StdRng,
    ) -> Result<(), ENNError> {
        match &mut self.state {
            FitState::Unfitted => return self.start_model(x_new, y_new, yvar_new, None, rng),
            FitState::Frozen { model, .. } => model.add(x_new, y_new, yvar_new)?,
            FitState::Searched { model, fit } => {
                model.add(x_new, y_new, yvar_new)?;
                fit.fitter.tell(x_new, y_new, yvar_new, self.config.y_bounds.as_ref())?;
            }
        }
        let has_params = self.state.params().is_some();
        let frozen = matches!(self.state, FitState::Frozen { .. });
        self.apply_plan(
            AppendPlan::new(AppendInput {
                on_disk: self.on_disk(),
                rows: x_new.nrows(),
                frozen,
                has_params,
                initial: false,
            }),
            rng,
        )
    }

    /// Replace the state with a model of `x`, `y`, `yvar` alone and fit its scales, starting
    /// the search from `warm`.
    fn start_model(
        &mut self,
        x_new: &ArrayView2<f64>,
        y_new: &ArrayView2<f64>,
        yvar_new: Option<&ArrayView2<f64>>,
        warm: Option<ENNParams>,
        rng: &mut rand::rngs::StdRng,
    ) -> Result<(), ENNError> {
        let mut model = self.construct_model(x_new, y_new, yvar_new)?;
        enable_auto_if_configured(&mut model, &self.config, x_new, y_new)?;
        self.state = match self.config.fit_samples {
            FitSamples::Frozen => FitState::Frozen {
                model,
                params: self.config.default_params()?,
            },
            FitSamples::Draw(search) => {
                let mut fitter = ENNFitter::new(self.config.k.get());
                fitter.tell(x_new, y_new, yvar_new, self.config.y_bounds.as_ref())?;
                FitState::Searched {
                    model,
                    fit: Box::new(ScaleFit {
                        search,
                        fitter,
                        params: warm,
                        calibrator: None,
                    }),
                }
            }
        };
        let frozen = self.config.fit_samples.is_frozen();
        self.apply_plan(
            AppendPlan::new(AppendInput {
                on_disk: self.on_disk(),
                rows: x_new.nrows(),
                frozen,
                has_params: false,
                initial: true,
            }),
            rng,
        )
    }
}

impl Surrogate for ENNSurrogate {
    fn fitted_num_metrics(&self) -> Option<usize> {
        self.state.model().map(|m| m.num_metrics())
    }

    fn observation_count(&self) -> Option<usize> {
        self.state.model().map(|m| m.len())
    }

    fn observation_row_x(&self, idx: usize) -> Result<Array1<f64>, ENNError> {
        let model = self.fitted_model()?;
        model.rows().row_x(idx)
    }

    fn observation_row_y(&self, idx: usize) -> Result<Array1<f64>, ENNError> {
        let model = self.fitted_model()?;
        
        model.row_y_natural(idx)
    }

    fn observations_y(&self) -> Result<Option<Array2<f64>>, ENNError> {
        let model = match self.state.model() {
            Some(m) => m,
            None => return Ok(None),
        };
        let n = model.len();
        if n == 0 {
            return Ok(None);
        }
        let mut y = Array2::zeros((n, model.num_metrics()));
        for i in 0..n {
            y.row_mut(i).assign(&model.row_y_natural(i)?);
        }
        Ok(Some(y))
    }

    fn naturalize_observations_y(&self, y_warped: Array2<f64>) -> Array2<f64> {
        let Some(model) = self.state.model() else {
            return y_warped;
        };
        if crate::y_bounds::is_identity_bounds(model.y_bounds()) {
            return y_warped;
        }
        crate::y_bounds::inv_y(y_warped.view(), model.y_bounds())
    }

    fn warp_observations_y(&self, y: &ArrayView2<f64>) -> Result<Array2<f64>, ENNError> {
        if let Some(model) = self.state.model() {
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
        let model = match self.state.model() {
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
        let warm = self.state.params().copied();
        self.state = FitState::Unfitted;
        self.start_model(x, y, yvar, warm, &mut local_rng)
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
        if let Some(model) = self.state.model() {
            model.backend.schedule_background_flush()
        } else {
            Ok(())
        }
    }

    fn wait_for_background_flush(&self) -> Result<(), ENNError> {
        if let Some(model) = self.state.model() {
            
            
            
            model.backend.wait_for_flush()
        } else {
            Ok(())
        }
    }

    fn release_observation_pages(&self) -> Result<(), ENNError> {
        if let Some(model) = self.state.model() {
            model.index_access().release_observation_pages()
        } else {
            Ok(())
        }
    }

    fn clear_observations(&mut self) -> Result<(), ENNError> {
        let owned_store = match self.state.model() {
            Some(model) => {
                model.backend.wait_for_flush()?;
                self.config.layout.work_dir().map(std::path::Path::to_path_buf)
            }
            None => None,
        };
        self.state = FitState::Unfitted;
        if let Some(work_dir) = owned_store {
            ennbo_bpann::bpann_remove_store(&work_dir)
                .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
        }
        Ok(())
    }

    fn predict(&self, x: &ArrayView2<f64>) -> Result<SurrogatePrediction, ENNError> {
        let model = self.fitted_model()?;
        let params = self.params_or_default()?;

        let flags = PosteriorFlags::new();
        let posterior = model.posterior(x, &params, &flags)?;

        let mu = posterior
            .mu
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| ENNError::InvalidParameter(format!("Shape error: {}", e)))?;
        let se = posterior
            .se
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(|e| ENNError::InvalidParameter(format!("Shape error: {}", e)))?;

        if let Some(cal) = self.state.calibrator() {
            let (mu, se) = crate::surrogate_affine::apply_prediction(cal, mu, se, model.y_bounds())?;
            return Ok(SurrogatePrediction { mu, se });
        }
        Ok(SurrogatePrediction { mu, se })
    }

    fn sample(
        &self,
        x: &ArrayView2<f64>,
        num_samples: usize,
        rng: &mut dyn RngCore,
    ) -> Result<Array3<f64>, ENNError> {
        let model = self.fitted_model()?;
        let params = self.params_or_default()?;

        
        let mut seed_bytes = [0u8; 8];
        rng.fill_bytes(&mut seed_bytes);
        let base_seed = u64::from_le_bytes(seed_bytes) as i64;
        let function_seeds: Vec<i64> = (0..num_samples as i64).map(|i| base_seed + i).collect();

        if let Some(cal) = self.state.calibrator() {
            return crate::surrogate_affine::calibrated_sample(model, &params, cal, x, num_samples, rng);
        }
        let (draws, _) =
            model.posterior_function_draw_warped(x, &params, &function_seeds, &Default::default())?;

        Ok(draws)
    }

    fn lengthscales(&self) -> Option<Array1<f64>> {
        if !matches!(self.config.layout, EnnLayout::DiskAuto { .. }) {
            return None;
        }
        let weights = self.state.model()?.metric_weights()?;
        Some(Array1::from(crate::metric_weights::trust_region_sides(weights)))
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
            k: crate::NeighborCount::new(2).unwrap(),
            fit_samples: crate::FitSamples::draw(3, 5).unwrap(),
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

    #[test]
    fn freeze_params_keeps_unit_epistemic_and_zero_aleatoric() {
        let config = ENNSurrogateConfig {
            k: crate::NeighborCount::new(2).unwrap(),
            fit_samples: FitSamples::Frozen,
            ..Default::default()
        };
        let mut surrogate = ENNSurrogate::new(config);
        let mut rng = StdRng::seed_from_u64(1);
        let x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];
        let y = array![[0.0], [1.0], [1.0], [2.0], [0.4]];
        surrogate
            .fit_append(&x.view(), &y.view(), None, &mut rng)
            .unwrap();
        let params = surrogate.params().unwrap();
        assert_eq!(params.k_num_neighbors, 2);
        assert_eq!(params.epistemic_variance_scale, 1.0);
        assert_eq!(params.aleatoric_variance_scale, 0.0);
        surrogate
            .fit_append(&array![[0.2, 0.2]].view(), &array![[0.3]].view(), None, &mut rng)
            .unwrap();
        let params = surrogate.params().unwrap();
        assert_eq!(params.epistemic_variance_scale, 1.0);
        assert_eq!(params.aleatoric_variance_scale, 0.0);
    }

    #[test]
    fn affine_flag_off_matches_itself_and_on_changes_prediction() {
        let x = array![[0.0], [0.25], [0.5], [0.75], [1.0], [1.25]];
        let y = array![[1.0], [1.5], [2.0], [2.5], [3.0], [3.5]];
        let query = array![[0.6]];
        let base = ENNSurrogateConfig {
            k: crate::NeighborCount::new(3).unwrap(),
            fit_samples: FitSamples::Draw(crate::fit_samples::test_search(6, 4, false)),
            ..Default::default()
        };
        let mut off_a = ENNSurrogate::new(base.clone());
        let mut off_b = ENNSurrogate::new(base.clone());
        let mut on_cfg = base;
        if let FitSamples::Draw(search) = &mut on_cfg.fit_samples {
            search.affine_calibrate = true;
        }
        let mut on = ENNSurrogate::new(on_cfg);
        off_a.fit(&x.view(), &y.view(), None, &mut StdRng::seed_from_u64(1)).unwrap();
        off_b.fit(&x.view(), &y.view(), None, &mut StdRng::seed_from_u64(1)).unwrap();
        on.fit(&x.view(), &y.view(), None, &mut StdRng::seed_from_u64(1)).unwrap();
        let a = off_a.predict(&query.view()).unwrap();
        let b = off_b.predict(&query.view()).unwrap();
        let c = on.predict(&query.view()).unwrap();
        assert_eq!(a.mu[[0, 0]], b.mu[[0, 0]]);
        assert_eq!(a.se[[0, 0]], b.se[[0, 0]]);
        assert!(off_a.state.calibrator().is_none());
        let cal = on.state.calibrator().expect("flag on stores a calibrator");
        let mu_gap = (c.mu[[0, 0]] - a.mu[[0, 0]]).abs();
        let se_gap = (c.se[[0, 0]] - a.se[[0, 0]]).abs();
        let moved = mu_gap > 1e-8 || se_gap > 1e-8;
        assert!(moved, "cal a={} b={} c={}", cal.a[0], cal.b[0], cal.c[0]);
    }

    /// Regression: incremental `fit` must not reuse stale `train_yvar` when the caller
    /// updates observation noise on prefix rows (same `x`/`y`, new `yvar` on old rows).
    #[test]
    fn regression_incremental_fit_refreshes_prefix_yvar_to_match_full_refit() {
        let config = ENNSurrogateConfig {
            k: crate::NeighborCount::new(2).unwrap(),
            fit_samples: crate::FitSamples::draw(2, 4).unwrap(),
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
            k: crate::NeighborCount::new(2).unwrap(),
            fit_samples: crate::FitSamples::draw(2, 4).unwrap(),
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
        assert!(cfg.k.get() >= 1);
    }

    #[test]
    fn fit_tells_fitter_warped_y_under_y_bounds() {
        let x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 2.0]];
        let y = array![[0.1], [0.2], [0.8], [0.9]];
        let bounds = array![[0.0, 1.0]];
        let mut fit_nat = crate::fitter::ENNFitter::new(2);
        fit_nat
            .tell(&x.view(), &y.view(), None, None)
            .unwrap();
        let nat_std = fit_nat.y_std()[0];
        let y_z = crate::y_bounds::warp_y(y.view(), &bounds).unwrap();
        let mut fit_z = crate::fitter::ENNFitter::new(2);
        fit_z
            .tell(&x.view(), &y_z.view(), None, None)
            .unwrap();
        let z_std = fit_z.y_std()[0];

        assert!(
            (nat_std - 0.35355).abs() < 0.01,
            "natural y_std={nat_std}"
        );
        assert!(z_std > 1.5, "warped y_std={z_std}");

        let mut fit_bounded = crate::fitter::ENNFitter::new(2);
        fit_bounded
            .tell(&x.view(), &y.view(), None, Some(&bounds))
            .unwrap();
        let bounded_std = fit_bounded.y_std()[0];
        assert!(
            (bounded_std - z_std).abs() < 1e-9,
            "tell with y_bounds must warp: got {bounded_std} want {z_std} (natural would be {nat_std})"
        );

        let config = ENNSurrogateConfig {
            k: crate::NeighborCount::new(2).unwrap(),
            fit_samples: crate::FitSamples::draw(4, 4).unwrap(),
            y_bounds: Some(bounds),
            ..Default::default()
        };
        let mut sur = ENNSurrogate::new(config);
        let mut rng = StdRng::seed_from_u64(7);
        sur.fit(&x.view(), &y.view(), None, &mut rng).unwrap();
        let FitState::Searched { fit, .. } = &sur.state else {
            panic!("Draw config must keep a fitter");
        };
        let fitter_std = fit.fitter.y_std()[0];
        assert!(
            (fitter_std - z_std).abs() < 1e-9,
            "surrogate fitter must track warped y: got {fitter_std} want {z_std} (natural would be {nat_std})"
        );
    }

    #[test]
    fn fit_append_disk_drains_pending_before_search() {
        use ndarray::Array2;
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        let config = ENNSurrogateConfig {
            k: crate::NeighborCount::new(2).unwrap(),
            fit_samples: crate::FitSamples::draw(2, 2).unwrap(),
            layout: EnnLayout::disk(dir.path().to_path_buf(), false),
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
        assert_eq!(
            indexed_rows(dir.path()),
            model.len(),
            "indexed_rows.bin must match num_obs after fit_append sync-before-fit"
        );
    }

    fn indexed_rows(dir: &std::path::Path) -> usize {
        std::fs::read(dir.join("indexed_rows.bin"))
            .ok()
            .and_then(|b| Some(u64::from_le_bytes(b.get(..8)?.try_into().ok()?) as usize))
            .unwrap_or(0)
    }

    #[test]
    fn first_bulk_disk_tell_defers_index_until_later_bulk_tell() {
        use ndarray::Array2;
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        let config = ENNSurrogateConfig {
            k: crate::NeighborCount::new(2).unwrap(),
            fit_samples: crate::FitSamples::draw(2, 2).unwrap(),
            layout: EnnLayout::disk(dir.path().to_path_buf(), false),
            ..Default::default()
        };
        let mut sur = ENNSurrogate::new(config);
        let mut rng = StdRng::seed_from_u64(7);
        let x = Array2::from_shape_fn((4_096, 2), |(i, j)| (i + j) as f64 * 0.001);
        let y = Array2::from_shape_fn((4_096, 1), |(i, _)| (i as f64) * 0.01);
        sur.fit_append(&x.view(), &y.view(), None, &mut rng).unwrap();
        sur.wait_for_background_flush().unwrap();
        let n = sur.model().expect("model").len();
        assert!(
            indexed_rows(dir.path()) < n,
            "first bulk disk tell must leave the index short of {n} rows"
        );
        sur.fit_append(&x.view(), &y.view(), None, &mut rng).unwrap();
        sur.wait_for_background_flush().unwrap();
        assert_eq!(indexed_rows(dir.path()), sur.model().expect("model").len());
    }

    #[test]
    fn fit_append_disk_streaming_skips_sync_when_params_exist() {
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        let config = ENNSurrogateConfig {
            k: crate::NeighborCount::new(2).unwrap(),
            fit_samples: crate::FitSamples::draw(2, 2).unwrap(),
            layout: EnnLayout::disk(dir.path().to_path_buf(), false),
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

    /// Regression: a TuRBO restart clears observations and refits in the same work_dir.
    #[test]
    fn clear_observations_disk_restarts_in_same_work_dir() {
        use tempfile::TempDir;

        let dir = TempDir::new().unwrap();
        let config = ENNSurrogateConfig {
            k: crate::NeighborCount::new(2).unwrap(),
            fit_samples: crate::FitSamples::draw(2, 2).unwrap(),
            layout: EnnLayout::disk(dir.path().to_path_buf(), false),
            ..Default::default()
        };
        let mut sur = ENNSurrogate::new(config);
        let mut rng = StdRng::seed_from_u64(5);
        let x0 = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let y0 = array![[0.0], [1.0], [0.5], [2.0]];
        sur.fit_append(&x0.view(), &y0.view(), None, &mut rng).unwrap();
        sur.fit_append(&array![[0.5, 0.5]].view(), &array![[0.7]].view(), None, &mut rng)
            .unwrap();
        let other = dir.path().join("not_bpann.txt");
        std::fs::write(&other, "keep").unwrap();

        sur.clear_observations().unwrap();
        assert!(sur.observation_count().is_none());

        let x1 = array![[0.2, 0.2], [0.8, 0.3], [0.4, 0.9]];
        let y1 = array![[3.0], [4.0], [5.0]];
        sur.fit_append(&x1.view(), &y1.view(), None, &mut rng).unwrap();
        assert_eq!(sur.observation_count(), Some(3));
        let ys = sur.observations_y().unwrap().unwrap();
        assert_eq!(ys, y1);
        assert!(sur.predict(&array![[0.3, 0.3]].view()).unwrap().mu[[0, 0]].is_finite());
        assert_eq!(std::fs::read_to_string(&other).unwrap(), "keep");
    }

    /// Under non-identity `y_bounds`, public `Surrogate::predict` returns natural-unit μ.
    #[test]
    fn regression_surrogate_predict_natural_under_y_bounds() {
        let bounds = array![[0.0, 1.0]];
        let config = ENNSurrogateConfig {
            k: crate::NeighborCount::new(2).unwrap(),
            fit_samples: crate::FitSamples::draw(3, 4).unwrap(),
            y_bounds: Some(bounds),
            ..Default::default()
        };
        let x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let y = array![[0.1], [0.9], [0.3], [0.7]];
        let mut sur = ENNSurrogate::new(config);
        let mut rng = StdRng::seed_from_u64(3);
        sur.fit(&x.view(), &y.view(), None, &mut rng).unwrap();

        let x_query = array![[0.5, 0.5]];
        let mu = sur.predict(&x_query.view()).unwrap().mu[[0, 0]];
        assert!(mu > 0.0 && mu < 1.0, "natural mu must lie in open (0,1); got {mu}");
    }

    #[test]
    fn surrogate_observation_row_and_batch_agree_natural_under_y_bounds() {
        let bounds = array![[0.0, 1.0]];
        let config = ENNSurrogateConfig {
            k: crate::NeighborCount::new(2).unwrap(),
            fit_samples: crate::FitSamples::draw(3, 4).unwrap(),
            y_bounds: Some(bounds),
            ..Default::default()
        };
        let x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let y = array![[0.1], [0.9], [0.3], [0.7]];
        let mut sur = ENNSurrogate::new(config);
        let mut rng = StdRng::seed_from_u64(3);
        sur.fit(&x.view(), &y.view(), None, &mut rng).unwrap();

        let batch = Surrogate::observations_y(&sur).unwrap().expect("observations");
        assert_eq!(batch.shape(), &[4, 1]);
        for i in 0..4 {
            let row = Surrogate::observation_row_y(&sur, i).unwrap();
            assert!(
                (row[[0]] - batch[[i, 0]]).abs() < 1e-12,
                "row {i}: row={} batch={}",
                row[[0]],
                batch[[i, 0]]
            );
            assert!(
                batch[[i, 0]] > 0.0 && batch[[i, 0]] < 1.0,
                "public observation y must be natural in (0,1), got {}",
                batch[[i, 0]]
            );
            assert!((batch[[i, 0]] - y[[i, 0]]).abs() < 1e-12);
        }

        let y_z = Surrogate::warp_observations_y(&sur, &batch.view()).unwrap();
        let max_diff = batch
            .iter()
            .zip(y_z.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_diff > 0.1,
            "warped storage must differ from natural under logit bounds"
        );
        let back = Surrogate::naturalize_observations_y(&sur, y_z);
        for (a, b) in batch.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    fn fitted_sides(layout: EnnLayout, weights: Option<&[f64]>) -> Option<Array1<f64>> {
        let config = ENNSurrogateConfig {
            k: crate::NeighborCount::new(2).unwrap(),
            fit_samples: crate::FitSamples::draw(2, 2).unwrap(),
            layout,
            ..Default::default()
        };
        let mut sur = ENNSurrogate::new(config);
        let mut rng = StdRng::seed_from_u64(5);
        let x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let y = array![[0.0], [1.0], [0.5], [2.0]];
        sur.fit_append(&x.view(), &y.view(), None, &mut rng).unwrap();
        if let Some(w) = weights {
            let FitState::Searched { model, .. } = &mut sur.state else {
                panic!("Draw config must reach the Searched state");
            };
            model.metric_set_weights(w, None).unwrap();
        }
        Surrogate::lengthscales(&sur)
    }

    #[test]
    fn fit_enables_auto_metric_like_fit_append() {
        let dir = tempfile::TempDir::new().unwrap();
        let x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let y = array![[0.0], [1.0], [0.5], [2.0]];
        let seen = |name: &str, use_fit: bool| {
            let mut sur = ENNSurrogate::new(ENNSurrogateConfig {
                k: NeighborCount::new(2).unwrap(),
                fit_samples: FitSamples::draw(2, 2).unwrap(),
                layout: EnnLayout::DiskAuto {
                    work_dir: dir.path().join(name),
                },
                ..Default::default()
            });
            let mut rng = StdRng::seed_from_u64(5);
            if use_fit {
                sur.fit(&x.view(), &y.view(), None, &mut rng).unwrap();
            } else {
                sur.fit_append(&x.view(), &y.view(), None, &mut rng).unwrap();
            }
            sur.model().unwrap().metric_snapshot().map(|s| s.num_seen)
        };
        let from_fit = seen("fit", true);
        assert!(from_fit.is_some(), "fit must enable the Auto metric");
        assert_eq!(from_fit, seen("append", false));
    }

    #[test]
    fn validate_rejects_ignored_or_out_of_range_tied_dims() {
        assert!(ENNSurrogateConfig::default().validate(2).is_ok());
        let mut cfg = ENNSurrogateConfig {
            tied_dims: vec![vec![0, 1]],
            ..Default::default()
        };
        let err = cfg.validate(2).unwrap_err();
        assert!(err.to_string().contains("metric_learning=Auto"), "{err}");
        cfg.layout = EnnLayout::DiskAuto {
            work_dir: std::path::PathBuf::from("/tmp/unused"),
        };
        cfg.validate(2).unwrap();
        assert!(cfg.validate(1).is_err());
        cfg.tied_dims = vec![vec![0], vec![0]];
        assert!(cfg.validate(2).is_err());
    }

    #[test]
    fn lengthscales_follow_auto_weights_only_under_disk_auto() {
        let dir = tempfile::TempDir::new().unwrap();
        let auto = |name: &str| EnnLayout::DiskAuto {
            work_dir: dir.path().join(name),
        };
        assert_eq!(fitted_sides(auto("a"), None).unwrap().to_vec(), vec![1.0, 1.0]);
        let s = fitted_sides(auto("b"), Some(&[1.0, 4.0])).unwrap();
        assert!((s[0] - 2.0).abs() < 1e-12 && (s[1] - 0.5).abs() < 1e-12);
        assert!(fitted_sides(EnnLayout::memory(IndexDriver::Flat, false), None).is_none());
        let disk = EnnLayout::disk(dir.path().join("disk"), false);
        assert!(fitted_sides(disk, None).is_none());
        let unfitted = ENNSurrogate::new(ENNSurrogateConfig {
            layout: auto("c"),
            ..Default::default()
        });
        assert!(Surrogate::lengthscales(&unfitted).is_none());
    }
}
