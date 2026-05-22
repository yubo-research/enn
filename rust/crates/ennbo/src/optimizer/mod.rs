//! Optimizer state machine for ask/tell pattern.

mod incumbent;
mod observation_delta;
mod observation_store;
mod tr_state;

pub use observation_delta::ObservationDelta;

use ndarray::{Array1, Array2, ArrayView2};
use rand::RngCore;

use crate::candidates::SobolEngine;
use crate::config::{InitStrategy, OptimizerConfig, SurrogateConfig};
use crate::error::ENNError;
use crate::incumbent_tracker::{
    tracker_m_from_enn_k, tracker_m_no_surrogate, IncrementalIncumbentTracker,
};
use crate::strategy::Strategy;
use crate::surrogate::{BoxedSurrogate, ENNSurrogate, Surrogate};
use tr_state::TrustRegionState;

use observation_store::ObservationStore;

/// Telemetry for timing.
#[derive(Debug, Clone, Default)]
pub struct Telemetry {
    pub dt_fit: f64,
    pub dt_gen: f64,
    pub dt_sel: f64,
    pub dt_tell: f64,
    pub num_candidates: usize,
}

/// Optimizer state machine.
pub struct Optimizer {
    bounds: Array2<f64>,
    num_dim: usize,
    config: OptimizerConfig,
    tr_state: TrustRegionState,
    surrogate: Option<BoxedSurrogate>,
    strategy: Strategy,
    obs_store: ObservationStore,
    incumbent_idx: Option<usize>,
    incumbent_x_unit: Option<Array1<f64>>,
    incumbent_y_scalar: Option<Array1<f64>>,
    restart_generation: usize,
    sobol_engine: Option<SobolEngine>,
    sobol_seed_base: u64,
    telemetry: Telemetry,
    incumbent_tracker: IncrementalIncumbentTracker,
}

impl Optimizer {
    /// Create a new optimizer.
    pub fn new(
        bounds: Array2<f64>,
        config: OptimizerConfig,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        Self::new_with_strategy(bounds, config, Strategy::hybrid(InitStrategy::LHD, 10), rng)
    }

    /// Create a new optimizer with an explicit strategy.
    pub fn new_with_strategy(
        bounds: Array2<f64>,
        config: OptimizerConfig,
        strategy: Strategy,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        let num_dim = bounds.nrows();
        if bounds.ncols() != 2 {
            return Err(ENNError::InvalidShape {
                expected: vec![num_dim, 2],
                got: vec![num_dim, bounds.ncols()],
            });
        }

        let tr_state = TrustRegionState::from_config(num_dim, &config.trust_region, rng)
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;

        // Initialize surrogate (None means no surrogate; NoSurrogate was wasteful/unclear)
        let surrogate: Option<BoxedSurrogate> = match &config.surrogate {
            SurrogateConfig::ENN(enn_config) => {
                Some(Box::new(ENNSurrogate::new(enn_config.clone())))
            }
            SurrogateConfig::None => None,
        };

        // Initialize Sobol engine if needed (scramble for randomized quasi-random)
        let sobol_engine =
            if config.candidates.candidate_rv == crate::candidates::CandidateRV::Sobol {
                let mut eng = SobolEngine::new(num_dim)?;
                eng.scramble(rng);
                Some(eng)
            } else {
                None
            };

        // Generate seed from RngCore
        let mut seed_bytes = [0u8; 8];
        rng.fill_bytes(&mut seed_bytes);
        let sobol_seed_base = u64::from_le_bytes(seed_bytes) % (1u64 << 31);
        let num_metrics = tr_state.num_metrics();
        let tracker_m = match &config.surrogate {
            SurrogateConfig::ENN(enn_config) => tracker_m_from_enn_k(enn_config.k),
            SurrogateConfig::None => tracker_m_no_surrogate(),
        };
        let noise_aware = config.noise_aware
            || tr_state
                .morbo()
                .map(|m| m.noise_aware())
                .unwrap_or(false);
        let incumbent_tracker =
            IncrementalIncumbentTracker::new(tracker_m, noise_aware, num_metrics);

        Ok(Self {
            bounds,
            num_dim,
            config,
            tr_state,
            surrogate,
            strategy,
            obs_store: ObservationStore::new(),
            incumbent_idx: None,
            incumbent_x_unit: None,
            incumbent_y_scalar: None,
            restart_generation: 0,
            sobol_engine,
            sobol_seed_base,
            telemetry: Telemetry::default(),
            incumbent_tracker,
        })
    }

    /// Ask for candidates.
    pub fn ask(&mut self, num_arms: usize, rng: &mut dyn RngCore) -> Result<Array2<f64>, ENNError> {
        let start = std::time::Instant::now();

        // Take strategy and telemetry out temporarily to avoid borrow issues
        let strategy = std::mem::replace(&mut self.strategy, Strategy::turbo());
        let mut telemetry = std::mem::take(&mut self.telemetry);
        let result = strategy.ask(self, num_arms, &mut telemetry, rng);
        self.strategy = strategy;
        self.telemetry = telemetry;

        self.telemetry.dt_gen = start.elapsed().as_secs_f64();
        result
    }

    /// Tell observations.
    pub fn tell(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        rng: &mut dyn RngCore,
    ) -> Result<(), ENNError> {
        let start = std::time::Instant::now();

        // Take strategy and telemetry out temporarily to avoid borrow issues
        let mut strategy = std::mem::replace(&mut self.strategy, Strategy::turbo());
        let mut telemetry = std::mem::take(&mut self.telemetry);
        let result = strategy.tell(self, x, y, &mut telemetry, rng);
        self.strategy = strategy;
        self.telemetry = telemetry;

        self.telemetry.dt_tell = start.elapsed().as_secs_f64();
        result
    }

    /// Get current telemetry.
    pub fn telemetry(&self) -> &Telemetry {
        &self.telemetry
    }

    /// Get bounds.
    pub fn bounds(&self) -> &Array2<f64> {
        &self.bounds
    }

    /// Get number of dimensions.
    pub fn num_dim(&self) -> usize {
        self.num_dim
    }

    /// Get configuration.
    pub fn config(&self) -> &OptimizerConfig {
        &self.config
    }

    /// Get trust region state.
    pub fn trust_region(&self) -> &TrustRegionState {
        &self.tr_state
    }

    /// Get mutable trust region state.
    pub fn trust_region_mut(&mut self) -> &mut TrustRegionState {
        &mut self.tr_state
    }

    /// Trust region length (TuRBO or Morbo inner).
    pub fn tr_length(&self) -> f64 {
        self.tr_state.length()
    }

    /// Get surrogate.
    pub fn surrogate(&self) -> Option<&(dyn Surrogate + Send + Sync)> {
        self.surrogate.as_ref().map(|s| s.as_ref())
    }

    /// Get mutable surrogate.
    pub fn surrogate_mut(&mut self) -> Option<&mut (dyn Surrogate + Send + Sync)> {
        match self.surrogate.as_mut() {
            Some(s) => Some(s.as_mut()),
            None => None,
        }
    }

    /// Get observations (uses cache; rebuilds only when invalidated).
    pub fn x_obs(&self) -> Option<Array2<f64>> {
        self.obs_store.x_obs_array()
    }

    /// Get observation values (uses cache; rebuilds only when invalidated).
    pub fn y_obs(&self) -> Option<Array2<f64>> {
        self.obs_store.y_obs_array()
    }

    /// Add observations (internal).
    pub fn add_observations(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
    ) -> Result<ObservationDelta, ENNError> {
        if x.nrows() != y.nrows() {
            return Err(ENNError::InvalidShape {
                expected: vec![x.nrows(), y.ncols()],
                got: vec![y.nrows(), y.ncols()],
            });
        }
        let old_n = self.obs_store.len();
        for i in 0..x.nrows() {
            let x_row: Array1<f64> = x.row(i).to_owned();
            let y_row: Array1<f64> = y.row(i).to_owned();
            self.incumbent_tracker.tell(old_n + i, &y_row);
            self.obs_store.push(x_row, y_row);
        }
        observation_delta::observation_delta_from_store(&self.obs_store, old_n)
    }

    /// Get incumbent x in unit space.
    pub fn incumbent_x_unit(&self) -> Option<&Array1<f64>> {
        self.incumbent_x_unit.as_ref()
    }

    /// Get incumbent y scalar.
    pub fn incumbent_y_scalar(&self) -> Option<&Array1<f64>> {
        self.incumbent_y_scalar.as_ref()
    }

    /// Increment restart generation.
    pub fn increment_restart_generation(&mut self) {
        self.restart_generation += 1;
    }

    /// Get restart generation.
    pub fn restart_generation(&self) -> usize {
        self.restart_generation
    }

    /// Get sobol engine.
    pub fn sobol_engine_mut(&mut self) -> Option<&mut SobolEngine> {
        self.sobol_engine.as_mut()
    }

    /// Get sobol seed base.
    pub fn sobol_seed_base(&self) -> u64 {
        self.sobol_seed_base
    }

    /// Get init progress from strategy.
    pub fn init_progress(&self) -> Option<(usize, usize)> {
        self.strategy.init_progress()
    }

    /// Current number of stored observations.
    pub fn obs_count(&self) -> usize {
        self.obs_store.len()
    }
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_incremental;
#[cfg(test)]
mod tests_morbo_incumbent;
#[cfg(test)]
mod tests_morbo_noise_aware_incumbent;
