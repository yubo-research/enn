//! Optimization strategies for ask/tell pattern.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::seq::SliceRandom;
use rand::RngCore;

use crate::util::argmax_random_tie;

use crate::acquisition::{ParetoAcquisition, RandomAcquisition, UCBAcquisition};
use crate::candidates::{generate_candidates, generate_lhd, generate_uniform};
use crate::config::{AcquisitionConfig, InitStrategy};
use crate::error::ENNError;
use crate::optimizer::{Optimizer, Telemetry};

/// Strategy state for initialization phase.
#[derive(Debug, Clone)]
pub struct InitStrategyState {
    pub strategy_type: InitStrategy,
    pub num_init: usize,
    pub completed: usize,
}

impl InitStrategyState {
    pub fn new(strategy_type: InitStrategy, num_init: usize) -> Self {
        Self {
            strategy_type,
            num_init,
            completed: 0,
        }
    }
}

/// Strategy state for TuRBO normal phase.
#[derive(Debug, Clone, Default)]
pub struct TurboStrategyState;

/// Strategy enum - uses concrete types instead of trait objects.
#[derive(Debug, Clone)]
pub enum Strategy {
    /// Initialization-only strategy.
    Init(InitStrategyState),
    /// TuRBO normal strategy.
    Turbo(TurboStrategyState),
    /// Hybrid: initialization then TuRBO.
    Hybrid {
        init: InitStrategyState,
        turbo: TurboStrategyState,
        in_init: bool,
    },
}

impl Strategy {
    /// Create a new initialization-only strategy.
    pub fn init(strategy_type: InitStrategy, num_init: usize) -> Self {
        Strategy::Init(InitStrategyState::new(strategy_type, num_init))
    }

    /// Create a new TuRBO strategy.
    pub fn turbo() -> Self {
        Strategy::Turbo(TurboStrategyState)
    }

    /// Create a new hybrid strategy.
    pub fn hybrid(init_strategy: InitStrategy, num_init: usize) -> Self {
        Strategy::Hybrid {
            init: InitStrategyState::new(init_strategy, num_init),
            turbo: TurboStrategyState,
            in_init: true,
        }
    }

    /// Generate candidates (ask).
    pub fn ask(
        &self,
        optimizer: &mut Optimizer,
        num_arms: usize,
        telemetry: &mut Telemetry,
        rng: &mut dyn RngCore,
    ) -> Result<Array2<f64>, ENNError> {
        match self {
            Strategy::Init(state) => ask_init(state, optimizer, num_arms, rng),
            Strategy::Turbo(_) => ask_turbo(optimizer, num_arms, telemetry, rng),
            Strategy::Hybrid {
                init,
                in_init: true,
                ..
            } => ask_init_hybrid(init, optimizer, num_arms, rng),
            Strategy::Hybrid { .. } => ask_turbo(optimizer, num_arms, telemetry, rng),
        }
    }

    /// Process observations (tell).
    pub fn tell(
        &mut self,
        optimizer: &mut Optimizer,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        telemetry: &mut Telemetry,
        rng: &mut dyn RngCore,
    ) -> Result<(), ENNError> {
        match self {
            Strategy::Init(state) => tell_init(state, optimizer, x, y, rng),
            Strategy::Turbo(_) => tell_turbo(optimizer, x, y, telemetry, rng),
            Strategy::Hybrid {
                init,
                turbo: _,
                in_init,
            } => {
                if *in_init {
                    tell_init(init, optimizer, x, y, rng)?;
                    // Check if init is complete
                    if init.completed >= init.num_init {
                        *in_init = false;
                    }
                    Ok(())
                } else {
                    tell_turbo(optimizer, x, y, telemetry, rng)
                }
            }
        }
    }

    /// Get initialization progress if applicable.
    pub fn init_progress(&self) -> Option<(usize, usize)> {
        match self {
            Strategy::Init(state) => Some((state.completed, state.num_init)),
            Strategy::Hybrid {
                init,
                in_init: true,
                ..
            } => Some((init.completed, init.num_init)),
            _ => None,
        }
    }
}

/// Ask for initialization phase.
fn ask_init(
    state: &InitStrategyState,
    optimizer: &mut Optimizer,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    let num_dim = optimizer.num_dim();
    let lower = Array1::zeros(num_dim);
    let upper = Array1::ones(num_dim);

    let candidates = match state.strategy_type {
        InitStrategy::LHD => {
            let mut unit_bounds = Array2::zeros((num_dim, 2));
            for j in 0..num_dim {
                unit_bounds[[j, 1]] = 1.0;
            }
            generate_lhd(num_arms, num_dim, &unit_bounds.view(), rng)
        }
        InitStrategy::Random => generate_uniform(&lower, &upper, num_arms, rng)?,
    };

    Ok(candidates)
}

/// Ask for initialization phase in hybrid mode.
fn ask_init_hybrid(
    state: &InitStrategyState,
    optimizer: &mut Optimizer,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    ask_init(state, optimizer, num_arms, rng)
}

fn morbo_sync_ranges_from_obs(optimizer: &mut Optimizer) -> Result<(), ENNError> {
    if !optimizer.trust_region().is_morbo() {
        return Ok(());
    }
    let Some(y_all) = optimizer.y_obs() else {
        return Ok(());
    };
    if y_all.nrows() == 0 {
        return Ok(());
    }
    optimizer
        .trust_region_mut()
        .morbo_update_ranges_only(&y_all.view())
}

/// Common tell logic: add observations, fit surrogate, update incumbent.
fn tell_common(
    optimizer: &mut Optimizer,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    telemetry: Option<&mut Telemetry>,
    rng: &mut dyn RngCore,
) -> Result<(), ENNError> {
    let delta = optimizer.add_observations(x, y)?;

    let refit_all = optimizer
        .surrogate()
        .and_then(|s| s.fitted_num_metrics())
        .is_some_and(|nm| nm != y.ncols());
    let obs_snapshot = if refit_all {
        (optimizer.x_obs(), optimizer.y_obs())
    } else {
        (None, None)
    };
    if let Some(surrogate) = optimizer.surrogate_mut() {
        let start = std::time::Instant::now();
        if let (Some(x_all), Some(y_all)) = obs_snapshot {
            surrogate.fit(&x_all.view(), &y_all.view(), None, rng)?;
        } else {
            surrogate.fit_append(
                &delta.x_new_view(),
                &delta.y_new_view(),
                None,
                rng,
            )?;
        }
        if let Some(tel) = telemetry {
            tel.dt_fit = start.elapsed().as_secs_f64();
        }
    }

    if optimizer.trust_region().is_morbo() && delta.new_n > delta.old_n {
        optimizer
            .trust_region_mut()
            .morbo_update_ranges_only(&delta.y_new_view())?;
    }

    optimizer.update_incumbent(rng)?;

    if optimizer.trust_region().is_morbo() {
        let num_obs = delta.new_n;
        if num_obs > 0 {
            let y_inc = optimizer
                .incumbent_y_scalar()
                .ok_or_else(|| ENNError::InvalidParameter("Missing incumbent y".to_string()))?
                .to_owned();
            optimizer.trust_region_mut().morbo_update_incumbent_only(
                &y_inc.view(),
                num_obs,
            )?;
        }
    }

    Ok(())
}

/// Tell for initialization phase.
fn tell_init(
    state: &mut InitStrategyState,
    optimizer: &mut Optimizer,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    rng: &mut dyn RngCore,
) -> Result<(), ENNError> {
    state.completed += x.nrows();
    tell_common(optimizer, x, y, None, rng)
}

/// Ask for TuRBO phase.
fn ask_turbo(
    optimizer: &mut Optimizer,
    num_arms: usize,
    telemetry: &mut Telemetry,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    optimizer.trust_region_mut().resample_on_propose(rng);
    optimizer.trust_region_mut().set_num_arms(num_arms);

    if optimizer.trust_region().is_morbo() {
        let num_obs = optimizer.obs_count();
        if num_obs > 0 {
            optimizer
                .trust_region_mut()
                .morbo_rescalarize_incumbent(num_obs)?;
        }
    }

    // Fetch incumbent center and lengthscales once (B5: was duplicated)
    let default_center = Array1::from_elem(optimizer.num_dim(), 0.5);
    let x_center = optimizer
        .incumbent_x_unit()
        .map(|x| x.to_owned())
        .unwrap_or(default_center);
    let lengthscales = optimizer.surrogate().and_then(|s| s.lengthscales());
    let ls_ref: Option<ArrayView1<f64>> = lengthscales.as_ref().map(|ls| ls.view());

    let (lower_1d, upper_1d) = optimizer
        .trust_region()
        .compute_bounds_1d(&x_center.view(), ls_ref.as_ref());

    // Generate candidates
    let num_dim = optimizer.num_dim();
    let config = optimizer.config().candidates.clone();
    let num_candidates = config.num_candidates(num_dim, num_arms);
    telemetry.num_candidates = num_candidates;

    let x_cand_unit = generate_candidates(
        || (lower_1d.clone(), upper_1d.clone()),
        &x_center.view(),
        ls_ref.as_ref(),
        num_candidates,
        config.candidate_rv,
        rng,
        optimizer.sobol_engine_mut(),
        20,
    )?;

    let capped_candidates = maybe_cap_selection_candidates(
        &x_cand_unit,
        optimizer.num_dim(),
        optimizer.obs_count(),
        num_arms,
        rng,
    );

    // Select arms using acquisition function (with timing)
    let start = std::time::Instant::now();
    let selected = select_arms(optimizer, &capped_candidates.view(), num_arms, rng)?;
    telemetry.dt_sel = start.elapsed().as_secs_f64();

    Ok(selected)
}

fn selection_candidate_cap(num_dim: usize, num_obs: usize, num_arms: usize) -> usize {
    if let Ok(v) = std::env::var("ENN_DISABLE_SEL_CAP") {
        if v == "1" || v.eq_ignore_ascii_case("true") {
            return usize::MAX;
        }
    }
    let min_cap = num_arms.saturating_mul(16).max(256);
    if num_dim >= 10_000 {
        return min_cap.max(256);
    }
    if num_dim >= 1_000 && num_obs >= 10_000 {
        return min_cap.max(320);
    }
    if num_dim >= 1_000 {
        return min_cap.max(384);
    }
    usize::MAX
}

fn maybe_cap_selection_candidates(
    x_cand: &Array2<f64>,
    num_dim: usize,
    num_obs: usize,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Array2<f64> {
    let cap = selection_candidate_cap(num_dim, num_obs, num_arms);
    if x_cand.nrows() <= cap {
        return x_cand.clone();
    }
    let mut indices: Vec<usize> = (0..x_cand.nrows()).collect();
    use rand::seq::SliceRandom;
    indices.shuffle(rng);
    indices.truncate(cap);
    select_by_indices(&x_cand.view(), &indices)
}

/// Tell for TuRBO phase.
fn tell_turbo(
    optimizer: &mut Optimizer,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    telemetry: &mut Telemetry,
    rng: &mut dyn RngCore,
) -> Result<(), ENNError> {
    tell_common(optimizer, x, y, Some(telemetry), rng)?;

    let y_all = optimizer
        .y_obs()
        .ok_or_else(|| ENNError::InvalidParameter("Missing y observations".to_string()))?;
    let num_obs = y_all.nrows();
    let y_incumbent = optimizer
        .incumbent_y_scalar()
        .ok_or_else(|| ENNError::InvalidParameter("Missing incumbent y".to_string()))?
        .to_owned();
    optimizer.trust_region_mut().set_num_arms(x.nrows());
    if !optimizer.trust_region().is_morbo() {
        optimizer
            .trust_region_mut()
            .tell_update(&y_all.view(), &y_incumbent.view(), num_obs)?;
    }
    if optimizer.trust_region().needs_restart() {
        optimizer.trust_region_mut().restart(Some(rng));
        optimizer.increment_restart_generation();
        optimizer.reset_incumbent_tracker();
        morbo_sync_ranges_from_obs(optimizer)?;
    }

    Ok(())
}

/// Select arms randomly.
fn select_with_random(
    x_cand: &ArrayView2<f64>,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    let random_acq = RandomAcquisition;
    let indices = random_acq
        .select(x_cand.nrows(), num_arms, rng)
        .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
    Ok(select_by_indices(x_cand, &indices))
}

/// Select arms via Thompson sampling (posterior draw).
fn select_with_thompson(
    optimizer: &Optimizer,
    surrogate: &(dyn crate::surrogate::Surrogate + Send + Sync),
    x_cand: &ArrayView2<f64>,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    let samples = surrogate.sample(x_cand, num_arms, rng)?;
    let n_candidates = x_cand.nrows();
    if optimizer.trust_region().is_morbo() {
        let num_metrics = samples.shape()[2];
        let mut flat = ndarray::Array2::zeros((num_arms * n_candidates, num_metrics));
        for arm in 0..num_arms {
            for cand in 0..n_candidates {
                for m in 0..num_metrics {
                    flat[[arm * n_candidates + cand, m]] = samples[[arm, cand, m]];
                }
            }
        }
        let flat_scores = optimizer
            .trust_region()
            .morbo_scalarize(&flat.view(), false)
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
        let mut all_scores = ndarray::Array2::zeros((num_arms, n_candidates));
        for arm in 0..num_arms {
            for cand in 0..n_candidates {
                all_scores[[arm, cand]] = flat_scores[arm * n_candidates + cand];
            }
        }
        let mut indices = Vec::with_capacity(num_arms);
        for arm in 0..num_arms {
            let mut arm_scores = vec![f64::NEG_INFINITY; n_candidates];
            for cand in 0..n_candidates {
                arm_scores[cand] = all_scores[[arm, cand]];
            }
            for &prev in &indices {
                arm_scores[prev] = f64::NEG_INFINITY;
            }
            indices.push(argmax_random_tie(&arm_scores, rng));
        }
        return Ok(select_by_indices(x_cand, &indices));
    }
    let sample_values: Vec<f64> = (0..n_candidates).map(|i| samples[[0, i, 0]]).collect();
    let mut indices: Vec<usize> = (0..n_candidates).collect();
    indices.sort_by(|&a, &b| {
        sample_values[b]
            .partial_cmp(&sample_values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let selected: Vec<usize> = indices.into_iter().take(num_arms).collect();
    Ok(select_by_indices(x_cand, &selected))
}

/// Select arms via UCB (upper confidence bound).
fn select_with_ucb(
    optimizer: &Optimizer,
    surrogate: &(dyn crate::surrogate::Surrogate + Send + Sync),
    x_cand: &ArrayView2<f64>,
    num_arms: usize,
    beta: f64,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    let pred = surrogate.predict(x_cand)?;
    if optimizer.trust_region().is_morbo() {
        let ucb_vals = &pred.mu + &(pred.se * beta);
        let scores = optimizer
            .trust_region()
            .morbo_scalarize(&ucb_vals.view(), false)
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
        let mut indices: Vec<usize> = (0..scores.len()).collect();
        indices.shuffle(rng);
        indices.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
        let selected: Vec<usize> = indices.into_iter().take(num_arms).collect();
        return Ok(select_by_indices(x_cand, &selected));
    }
    let mu = pred.mu.column(0);
    let sigma = pred.se.column(0);
    let ucb = UCBAcquisition::new(beta);
    let indices = ucb
        .select(&mu, &sigma, num_arms, rng)
        .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
    Ok(select_by_indices(x_cand, &indices))
}

/// Select arms via Pareto frontier.
fn select_with_pareto(
    surrogate: &(dyn crate::surrogate::Surrogate + Send + Sync),
    x_cand: &ArrayView2<f64>,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    let pred = surrogate.predict(x_cand)?;
    let pareto = ParetoAcquisition::new();
    let indices = pareto
        .select(&pred.mu.view(), &pred.se.view(), num_arms, rng)
        .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
    Ok(select_by_indices(x_cand, &indices))
}

/// Select arms using acquisition function.
fn select_arms(
    optimizer: &Optimizer,
    x_cand: &ArrayView2<f64>,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    let config = optimizer.config().acquisition;

    match config {
        AcquisitionConfig::Random => select_with_random(x_cand, num_arms, rng),
        AcquisitionConfig::Thompson => match optimizer.surrogate() {
            Some(s) => select_with_thompson(optimizer, s, x_cand, num_arms, rng),
            None => select_with_random(x_cand, num_arms, rng),
        },
        AcquisitionConfig::UCB { beta } => match optimizer.surrogate() {
            Some(s) => select_with_ucb(optimizer, s, x_cand, num_arms, beta, rng),
            None => select_with_random(x_cand, num_arms, rng),
        },
        AcquisitionConfig::Pareto => match optimizer.surrogate() {
            Some(s) => select_with_pareto(s, x_cand, num_arms, rng),
            None => select_with_random(x_cand, num_arms, rng),
        },
    }
}

/// Select rows by indices.
fn select_by_indices(x: &ArrayView2<f64>, indices: &[usize]) -> Array2<f64> {
    use ndarray::Axis;
    let rows: Vec<_> = indices.iter().map(|&i| x.row(i).to_owned()).collect();
    ndarray::stack(Axis(0), &rows.iter().map(|r| r.view()).collect::<Vec<_>>())
        .expect("stack should succeed for same-shaped rows")
}

#[cfg(test)]
mod tests_init;
#[cfg(test)]
mod tests_morbo_acq;
#[cfg(test)]
mod tests_selection;
