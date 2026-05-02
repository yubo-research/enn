//! Optimizer state machine for ask/tell pattern.

use std::cell::RefCell;
use ndarray::{Array1, Array2, ArrayView2};
use rand::RngCore;

use crate::candidates::SobolEngine;
use crate::config::{OptimizerConfig, SurrogateConfig, InitStrategy};
use crate::error::ENNError;
use crate::strategy::Strategy;
use crate::surrogate::{BoxedSurrogate, ENNSurrogate, Surrogate};
use crate::trust_region::TurboTrustRegion;

/// Telemetry for timing.
#[derive(Debug, Clone, Default)]
pub struct Telemetry {
    pub dt_fit: f64,
    pub dt_gen: f64,
    pub dt_sel: f64,
    pub dt_tell: f64,
}

/// Observation store with auto-invalidating cache.
struct ObservationStore {
    x_obs: Vec<Array1<f64>>,
    y_obs: Vec<Array1<f64>>,
    cached_x: RefCell<Option<Array2<f64>>>,
    cached_y: RefCell<Option<Array2<f64>>>,
}

impl ObservationStore {
    fn new() -> Self {
        Self {
            x_obs: Vec::new(),
            y_obs: Vec::new(),
            cached_x: RefCell::new(None),
            cached_y: RefCell::new(None),
        }
    }

    fn invalidate_cache(&self) {
        *self.cached_x.borrow_mut() = None;
        *self.cached_y.borrow_mut() = None;
    }

    fn push(&mut self, x: Array1<f64>, y: Array1<f64>) {
        self.invalidate_cache();
        self.x_obs.push(x);
        self.y_obs.push(y);
    }

    fn len(&self) -> usize {
        self.x_obs.len()
    }

    fn is_empty(&self) -> bool {
        self.x_obs.is_empty()
    }

    fn x_obs_array(&self) -> Option<Array2<f64>> {
        if self.x_obs.is_empty() {
            return None;
        }
        let mut cache = self.cached_x.borrow_mut();
        if let Some(ref cached) = *cache {
            return Some(cached.clone());
        }
        let arr = Self::build_array2(&self.x_obs);
        *cache = Some(arr.clone());
        Some(arr)
    }

    fn y_obs_array(&self) -> Option<Array2<f64>> {
        if self.y_obs.is_empty() {
            return None;
        }
        let mut cache = self.cached_y.borrow_mut();
        if let Some(ref cached) = *cache {
            return Some(cached.clone());
        }
        let arr = Self::build_array2(&self.y_obs);
        *cache = Some(arr.clone());
        Some(arr)
    }

    fn build_array2(vecs: &[Array1<f64>]) -> Array2<f64> {
        let n = vecs.len();
        let d = vecs[0].len();
        let mut result = Array2::zeros((n, d));
        for (i, v) in vecs.iter().enumerate() {
            for j in 0..d {
                result[[i, j]] = v[j];
            }
        }
        result
    }

    fn replace(&mut self, new_x: Vec<Array1<f64>>, new_y: Vec<Array1<f64>>) {
        self.invalidate_cache();
        self.x_obs = new_x;
        self.y_obs = new_y;
    }

    fn x_at(&self, idx: usize) -> &Array1<f64> {
        &self.x_obs[idx]
    }

    fn y_at(&self, idx: usize) -> &Array1<f64> {
        &self.y_obs[idx]
    }

    fn iter_indices(&self) -> impl Iterator<Item = usize> {
        0..self.x_obs.len()
    }
}

/// Optimizer state machine.
pub struct Optimizer {
    bounds: Array2<f64>,
    num_dim: usize,
    config: OptimizerConfig,
    tr_state: TurboTrustRegion,
    surrogate: Option<BoxedSurrogate>,
    strategy: Strategy,
    obs_store: ObservationStore,
    trailing_obs: Option<usize>,
    incumbent_idx: Option<usize>,
    incumbent_x_unit: Option<Array1<f64>>,
    incumbent_y_scalar: Option<Array1<f64>>,
    restart_generation: usize,
    sobol_engine: Option<SobolEngine>,
    sobol_seed_base: u64,
    telemetry: Telemetry,
}

impl Optimizer {
    /// Create a new optimizer.
    pub fn new(
        bounds: Array2<f64>,
        config: OptimizerConfig,
        rng: &mut dyn RngCore,
    ) -> Result<Self, ENNError> {
        Self::new_with_strategy(
            bounds,
            config,
            Strategy::hybrid(InitStrategy::LHD, 10),
            rng,
        )
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

        // Initialize trust region
        let tr_state = TurboTrustRegion::new(
            num_dim,
            config.trust_region,
        );

        // Initialize surrogate (None means no surrogate; NoSurrogate was wasteful/unclear)
        let surrogate: Option<BoxedSurrogate> = match &config.surrogate {
            SurrogateConfig::ENN(enn_config) => {
                Some(Box::new(ENNSurrogate::new(enn_config.clone())))
            }
            SurrogateConfig::None => None,
        };

        // Initialize Sobol engine if needed (scramble for randomized quasi-random)
        let sobol_engine = if config.candidates.candidate_rv == crate::candidates::CandidateRV::Sobol {
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
        let trailing_obs = config.trailing_obs;

        Ok(Self {
            bounds,
            num_dim,
            config,
            tr_state,
            surrogate,
            strategy,
            obs_store: ObservationStore::new(),
            trailing_obs,
            incumbent_idx: None,
            incumbent_x_unit: None,
            incumbent_y_scalar: None,
            restart_generation: 0,
            sobol_engine,
            sobol_seed_base,
            telemetry: Telemetry::default(),
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

    /// Get trust region.
    pub fn trust_region(&self) -> &TurboTrustRegion {
        &self.tr_state
    }

    /// Get mutable trust region.
    pub fn trust_region_mut(&mut self) -> &mut TurboTrustRegion {
        &mut self.tr_state
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
    ) -> Result<(), ENNError> {
        if x.nrows() != y.nrows() {
            return Err(ENNError::InvalidShape {
                expected: vec![x.nrows(), y.ncols()],
                got: vec![y.nrows(), y.ncols()],
            });
        }
        for i in 0..x.nrows() {
            let x_row: Array1<f64> = x.row(i).to_owned();
            let y_row: Array1<f64> = y.row(i).to_owned();
            self.obs_store.push(x_row, y_row);
        }
        Ok(())
    }

    /// Trim observations to trailing_obs limit, preserving incumbent + recent.
    /// Must be called after update_incumbent so incumbent_idx is current.
    pub fn trim_trailing_obs(&mut self) -> Result<(), ENNError> {
        let Some(limit) = self.trailing_obs else {
            return Ok(());
        };
        let n = self.obs_store.len();
        if n <= limit {
            return Ok(());
        }

        let start = n.saturating_sub(limit);
        let recent: std::collections::HashSet<usize> = (start..n).collect();

        let mut keep: std::collections::HashSet<usize> = recent;
        if let Some(idx) = self.incumbent_idx {
            keep.insert(idx);
        }

        let keep = if keep.len() > limit {
            let mut k: std::collections::HashSet<usize> =
                self.incumbent_idx.iter().copied().collect();
            let mut remaining = limit.saturating_sub(k.len());
            for i in (0..n).rev() {
                if remaining == 0 {
                    break;
                }
                if !k.contains(&i) {
                    k.insert(i);
                    remaining -= 1;
                }
            }
            k
        } else {
            keep
        };

        let mut indices: Vec<usize> = keep.into_iter().collect();
        indices.sort_unstable();

        let new_x: Vec<Array1<f64>> = indices
            .iter()
            .map(|&i| self.obs_store.x_at(i).clone())
            .collect();
        let new_y: Vec<Array1<f64>> = indices
            .iter()
            .map(|&i| self.obs_store.y_at(i).clone())
            .collect();

        self.obs_store.replace(new_x, new_y);

        let new_incumbent_idx = self
            .incumbent_idx
            .and_then(|old_idx| indices.iter().position(|&i| i == old_idx));
        self.incumbent_idx = new_incumbent_idx;
        if let Some(idx) = self.incumbent_idx {
            self.incumbent_x_unit = Some(self.obs_store.x_at(idx).clone());
            self.incumbent_y_scalar = Some(self.obs_store.y_at(idx).clone());
        }

        Ok(())
    }

    /// Update incumbent.
    pub fn update_incumbent(&mut self, _rng: &mut dyn RngCore) -> Result<(), ENNError> {
        if self.obs_store.is_empty() {
            self.incumbent_idx = None;
            self.incumbent_x_unit = None;
            self.incumbent_y_scalar = None;
            return Ok(());
        }

        let candidate_indices = if let Some(surrogate) = &self.surrogate {
            let y_obs = self.y_obs().unwrap();
            surrogate.get_incumbent_indices(&y_obs.view())
        } else {
            self.obs_store.iter_indices().collect()
        };

        if candidate_indices.is_empty() {
            self.incumbent_idx = None;
            self.incumbent_x_unit = None;
            self.incumbent_y_scalar = None;
            return Ok(());
        }

        let best_idx = candidate_indices
            .into_iter()
            .max_by(|&a, &b| {
                let a_y = self.obs_store.y_at(a)[0];
                let b_y = self.obs_store.y_at(b)[0];
                a_y.total_cmp(&b_y)
            })
            .ok_or_else(|| ENNError::InvalidParameter("No incumbent candidates".to_string()))?;
        self.incumbent_idx = Some(best_idx);
        self.incumbent_x_unit = Some(self.obs_store.x_at(best_idx).clone());
        self.incumbent_y_scalar = Some(self.obs_store.y_at(best_idx).clone());

        Ok(())
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
