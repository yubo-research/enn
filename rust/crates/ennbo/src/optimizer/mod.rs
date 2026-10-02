//! Optimizer state machine for ask/tell pattern.

mod incumbent;
pub mod obs_access;
mod observation_delta;
mod restart;
mod tr_state;

pub use observation_delta::ObservationDelta;

use ndarray::{Array1, Array2, ArrayView2};
use rand::rngs::StdRng;
use rand::{RngCore, SeedableRng};

use crate::candidates::SobolEngine;
use crate::config::{InitStrategy, OptimizerConfig, SurrogateConfig};
use crate::error::ENNError;
use crate::incumbent_tracker::{
    tracker_m_from_enn_k, tracker_m_no_surrogate, IncrementalIncumbentTracker,
};
use crate::strategy::Strategy;
use crate::surrogate::{BoxedSurrogate, ENNSurrogate, Surrogate};
use tr_state::TrustRegionState;

fn sobol_seed_for_state(
    seed_base: u64,
    restart_generation: usize,
    n_obs: usize,
    num_arms: usize,
) -> u64 {
    let mut x = seed_base;
    x ^= (restart_generation.wrapping_add(1) as u64).wrapping_mul(0xD1342543DE82EF95);
    x ^= (n_obs as u64)
        .wrapping_add(1)
        .wrapping_mul(0x9E3779B97F4A7C15);
    x ^= (num_arms as u64)
        .wrapping_add(1)
        .wrapping_mul(0xBF58476D1CE4E5B9);
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^= z >> 31;
    z & 0xFFFF_FFFF
}

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
    pub(crate) fallback_x: Vec<Array1<f64>>,
    pub(crate) fallback_y: Vec<Array1<f64>>,
    incumbent_idx: Option<usize>,
    incumbent_x_unit: Option<Array1<f64>>,
    incumbent_y_scalar: Option<Array1<f64>>,
    restart_generation: usize,
    sobol_engine: Option<SobolEngine>,
    sobol_seed_base: u64,
    telemetry: Telemetry,
    incumbent_tracker: IncrementalIncumbentTracker,
    /// RNG owned for the life of the optimizer. `ask` and `tell` draw from it.
    rng: StdRng,
    /// `Some(true)` after the first non-empty tell that included `yvar`.
    yvar_on_tell: Option<bool>,
    /// Points in the fresh local sample drawn after a collapsed trust region.
    local_init_budget: usize,
    local_init_kind: InitStrategy,
    /// `Some(remaining)` while that fresh sample is still being collected.
    reinit_left: Option<usize>,
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

        let surrogate: Option<BoxedSurrogate> = match &config.surrogate {
            SurrogateConfig::ENN(enn_config) => {
                Some(Box::new(ENNSurrogate::new(enn_config.clone())))
            }
            SurrogateConfig::None => None,
        };

        let sobol_engine =
            if config.candidates.candidate_rv == crate::candidates::CandidateRV::Sobol {
                let mut eng = SobolEngine::new(num_dim)?;
                eng.scramble(rng);
                Some(eng)
            } else {
                None
            };

        let (local_init_budget, local_init_kind) = strategy.init_plan();
        let mut seed_bytes = [0u8; 8];
        rng.fill_bytes(&mut seed_bytes);
        let sobol_seed_base = u64::from_le_bytes(seed_bytes) % (1u64 << 31);
        let owned_rng = StdRng::from_rng(rng).map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
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
            fallback_x: Vec::new(),
            fallback_y: Vec::new(),
            incumbent_idx: None,
            incumbent_x_unit: None,
            incumbent_y_scalar: None,
            restart_generation: 0,
            sobol_engine,
            sobol_seed_base,
            telemetry: Telemetry::default(),
            incumbent_tracker,
            rng: owned_rng,
            yvar_on_tell: None,
            local_init_budget,
            local_init_kind,
            reinit_left: None,
        })
    }

    /// Ask for `num_arms` points in natural units (inside `bounds`).
    ///
    /// Draws from the `StdRng` stored at construction. Rejects `num_arms == 0`.
    /// A trust-region restart happens here, not in the `tell` that collapsed the
    /// box, because that `tell` still owes the caller a posterior.
    pub fn ask(&mut self, num_arms: usize) -> Result<Array2<f64>, ENNError> {
        if num_arms == 0 {
            return Err(ENNError::InvalidParameter(format!(
                "num_arms must be > 0, got {num_arms}"
            )));
        }
        if !self.tr_state.is_morbo() && self.tr_state.needs_restart() {
            self.begin_local_restart()?;
        }
        let start = std::time::Instant::now();
        let mut rng = self.rng.clone();

        let strategy = std::mem::replace(&mut self.strategy, Strategy::turbo());
        let mut telemetry = std::mem::take(&mut self.telemetry);
        let result = strategy.ask(self, num_arms, &mut telemetry, &mut rng);
        self.strategy = strategy;
        self.telemetry = telemetry;
        self.rng = rng;

        self.telemetry.dt_gen = start.elapsed().as_secs_f64();
        if result.is_ok() {
            if let Some(surrogate) = self.surrogate.as_ref() {
                surrogate.schedule_background_flush()?;
            }
        }
        let unit = result?;
        Ok(crate::candidates::from_unit(&unit.view(), &self.bounds.view()))
    }

    /// Record observations in natural units. `x` is stored in the unit cube.
    ///
    /// The first non-empty tell fixes whether `yvar` is required. Later tells
    /// must match. Zero rows return `Ok(())` and do not change that flag.
    /// Draws from the owned `StdRng`.
    pub fn tell(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Result<(), ENNError> {
        if x.ncols() != self.num_dim {
            return Err(ENNError::InvalidShape {
                expected: vec![x.nrows(), self.num_dim],
                got: vec![x.nrows(), x.ncols()],
            });
        }
        if y.nrows() != x.nrows() {
            return Err(ENNError::InvalidShape {
                expected: vec![x.nrows(), y.ncols()],
                got: vec![y.nrows(), y.ncols()],
            });
        }
        if x.nrows() == 0 {
            return Ok(());
        }
        let has_yvar = yvar.is_some();
        match self.yvar_on_tell {
            None => self.yvar_on_tell = Some(has_yvar),
            Some(expected) if expected != has_yvar => {
                return Err(ENNError::InvalidParameter(format!(
                    "y_var must be {} on every tell()",
                    if expected { "provided" } else { "omitted" }
                )));
            }
            Some(_) => {}
        }
        let x_unit = crate::candidates::to_unit(x, &self.bounds.view());
        let start = std::time::Instant::now();
        let mut rng = self.rng.clone();

        if let Some(surrogate) = self.surrogate.as_ref() {
            surrogate.wait_for_background_flush()?;
        }

        let mut strategy = std::mem::replace(&mut self.strategy, Strategy::turbo());
        let mut telemetry = std::mem::take(&mut self.telemetry);
        let result = strategy.tell(self, &x_unit.view(), y, yvar, &mut telemetry, &mut rng);
        self.strategy = strategy;
        self.telemetry = telemetry;
        self.rng = rng;

        self.telemetry.dt_tell = start.elapsed().as_secs_f64();
        if result.is_ok() && x.nrows() < 64 {
            if let Some(surrogate) = self.surrogate.as_ref() {
                surrogate.schedule_background_flush()?;
            }
        }
        result
    }

    /// Posterior mean at `x` in natural units, in natural `y` units.
    ///
    /// `x` is converted to the unit cube, matching `tell`. There is no
    /// surrogate on TuRBO-ZERO, and this returns an error in that case.
    pub fn posterior_mu(&self, x_natural: &ArrayView2<f64>) -> Result<Array2<f64>, ENNError> {
        if x_natural.ncols() != self.num_dim {
            return Err(ENNError::InvalidShape {
                expected: vec![x_natural.nrows(), self.num_dim],
                got: vec![x_natural.nrows(), x_natural.ncols()],
            });
        }
        let surrogate = self.surrogate.as_ref().ok_or_else(|| {
            ENNError::InvalidParameter("No surrogate".to_string())
        })?;
        let x_unit = crate::candidates::to_unit(x_natural, &self.bounds.view());
        let pred = surrogate.naturalize_prediction(surrogate.predict(&x_unit.view())?);
        Ok(pred.mu)
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

    /// Row-level observation access (ENN surrogate or fallback store).
    pub fn obs_access(&self) -> obs_access::ObsAccess<'_> {
        obs_access::ObsAccess::new(self)
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

    /// Stored `x` in the unit cube (internal).
    pub(crate) fn x_obs_unit(&self) -> Option<Array2<f64>> {
        if let Some(surrogate) = self.surrogate.as_ref() {
            return surrogate.observations_x().ok().flatten();
        }
        if self.fallback_x.is_empty() {
            return None;
        }
        Some(obs_access::build_obs_array2(&self.fallback_x))
    }

    /// Observations in natural units.
    pub fn x_obs(&self) -> Option<Array2<f64>> {
        let unit = self.x_obs_unit()?;
        Some(crate::candidates::from_unit(&unit.view(), &self.bounds.view()))
    }

    /// Get observation values in natural units (ENN model or fallback store).
    pub fn y_obs(&self) -> Option<Array2<f64>> {
        let y_z = self.obs_access().y_obs_warped()?;
        if let Some(surrogate) = self.surrogate.as_ref() {
            return Some(surrogate.naturalize_observations_y(y_z));
        }
        Some(y_z)
    }

    /// Add observations (internal). `y` is natural-unit; incumbent stores natural units
    /// so trust-region updates compare batch `y` and incumbent in the same space.
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
        let old_n = self.obs_count();
        for i in 0..x.nrows() {
            let y_row: Array1<f64> = y.row(i).to_owned();
            self.incumbent_tracker.tell(old_n + i, &y_row);
            if self.surrogate.is_none() {
                self.fallback_x.push(x.row(i).to_owned());
                self.fallback_y.push(y.row(i).to_owned());
            }
        }
        observation_delta::observation_delta_from_batch(old_n, x, y)
    }

    /// Incumbent `x` in the unit cube. Internal; callers use [`Self::incumbent_x`].
    pub(crate) fn incumbent_x_unit(&self) -> Option<&Array1<f64>> {
        self.incumbent_x_unit.as_ref()
    }

    /// Incumbent `x` in natural units.
    pub fn incumbent_x(&self) -> Option<Array1<f64>> {
        let unit = self.incumbent_x_unit.as_ref()?;
        let row = unit.clone().insert_axis(ndarray::Axis(0));
        let natural = crate::candidates::from_unit(&row.view(), &self.bounds.view());
        Some(natural.row(0).to_owned())
    }

    /// Get incumbent y scalar.
    pub fn incumbent_y_scalar(&self) -> Option<&Array1<f64>> {
        self.incumbent_y_scalar.as_ref()
    }

    /// Get sobol engine.
    pub fn sobol_engine_mut(&mut self) -> Option<&mut SobolEngine> {
        self.sobol_engine.as_mut()
    }

    /// Fresh scrambled Sobol draw for this ask. Matches Python's per-ask engine.
    pub fn reseed_sobol(&mut self, num_arms: usize) -> Result<(), ENNError> {
        let Some(engine) = self.sobol_engine.as_ref() else {
            return Ok(());
        };
        let dim = engine.dimension();
        let seed = sobol_seed_for_state(
            self.sobol_seed_base,
            self.restart_generation,
            self.obs_count(),
            num_arms,
        );
        let mut eng = SobolEngine::new(dim)?;
        let mut rng = StdRng::seed_from_u64(seed);
        eng.scramble(&mut rng);
        self.sobol_engine = Some(eng);
        Ok(())
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
        if let Some(surrogate) = self.surrogate.as_ref() {
            return surrogate.observation_count().unwrap_or(0);
        }
        self.fallback_x.len()
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
