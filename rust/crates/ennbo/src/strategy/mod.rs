//! Optimization strategies for ask/tell pattern.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::RngCore;

use crate::candidates::{generate_candidates, generate_lhd, generate_sobol_masked, generate_uniform};
use crate::config::InitStrategy;
use crate::error::ENNError;
use crate::optimizer::{Optimizer, Telemetry};

mod select;
use select::select_arms;
#[cfg(test)]
use select::{
    select_by_indices, select_with_pareto, select_with_random, select_with_thompson,
    select_with_ucb,
};

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
        if optimizer.reinit_left().is_some() {
            return ask_local_reinit(optimizer, num_arms, rng);
        }
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
        yvar: Option<&ArrayView2<f64>>,
        telemetry: &mut Telemetry,
        rng: &mut dyn RngCore,
    ) -> Result<(), ENNError> {
        if optimizer.reinit_left().is_some() {
            return tell_local_reinit(optimizer, x, y, yvar, rng);
        }
        match self {
            Strategy::Init(state) => tell_init(state, optimizer, x, y, yvar, rng),
            Strategy::Turbo(_) => tell_turbo(optimizer, x, y, yvar, telemetry, rng),
            Strategy::Hybrid {
                init,
                turbo: _,
                in_init,
            } => {
                if *in_init {
                    tell_init(init, optimizer, x, y, yvar, rng)?;

                    if init.completed >= init.num_init {
                        *in_init = false;
                    }
                    Ok(())
                } else {
                    tell_turbo(optimizer, x, y, yvar, telemetry, rng)
                }
            }
        }
    }

    /// Budget and design of the initial sample. TuRBO-only has no sample.
    pub fn init_plan(&self) -> (usize, InitStrategy) {
        match self {
            Strategy::Init(state) => (state.num_init, state.strategy_type),
            Strategy::Hybrid { init, .. } => (init.num_init, init.strategy_type),
            Strategy::Turbo(_) => (0, InitStrategy::LHD),
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

/// Fresh Latin hypercube (or uniform draw) after a collapsed trust region.
fn ask_local_reinit(
    optimizer: &mut Optimizer,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    let left = optimizer.reinit_left().unwrap_or(num_arms);
    let n = num_arms.min(left).max(1);
    let state = InitStrategyState::new(optimizer.local_init_kind(), n);
    ask_init(&state, optimizer, n, rng)
}

fn tell_local_reinit(
    optimizer: &mut Optimizer,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
    rng: &mut dyn RngCore,
) -> Result<(), ENNError> {
    tell_common(optimizer, x, y, yvar, None, rng)?;
    optimizer.consume_reinit(x.nrows());
    Ok(())
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
    yvar: Option<&ArrayView2<f64>>,
    telemetry: Option<&mut Telemetry>,
    rng: &mut dyn RngCore,
) -> Result<(), ENNError> {
    if optimizer.trust_region().is_morbo() {
        let nm = optimizer.trust_region().num_metrics();
        if y.ncols() != nm {
            return Err(ENNError::InvalidParameter(format!(
                "y has {} metric columns but Morbo expects {nm}",
                y.ncols()
            )));
        }
    }

    let delta = optimizer.add_observations(x, y)?;

    if let Some(nm) = optimizer.surrogate().and_then(|s| s.fitted_num_metrics()) {
        if nm != y.ncols() {
            return Err(ENNError::InvalidParameter(format!(
                "y has {} metric columns but the fitted model has {nm}; changing output width is unsupported",
                y.ncols()
            )));
        }
    }
    if let Some(surrogate) = optimizer.surrogate_mut() {
        let start = std::time::Instant::now();
        surrogate.fit_append(
            &delta.x_new_view(),
            &delta.y_new_view(),
            yvar,
            rng,
        )?;
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




    if x.nrows() >= 64 {
        if let Some(surrogate) = optimizer.surrogate() {
            surrogate.release_observation_pages()?;
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
    yvar: Option<&ArrayView2<f64>>,
    rng: &mut dyn RngCore,
) -> Result<(), ENNError> {
    state.completed += x.nrows();
    tell_common(optimizer, x, y, yvar, None, rng)
}

fn draw_turbo_candidates(
    optimizer: &mut Optimizer,
    x_center: &ArrayView1<f64>,
    lower_1d: &Array1<f64>,
    upper_1d: &Array1<f64>,
    num_candidates: usize,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    let raasp_fast = optimizer.config().candidates.raasp_fast;
    let candidate_rv = optimizer.config().candidates.candidate_rv;
    if raasp_fast && candidate_rv == crate::candidates::CandidateRV::RAASP {
        return crate::candidates_fast::generate_tr_candidates_fast(
            x_center,
            lower_1d,
            upper_1d,
            num_candidates,
            rng,
            20,
        );
    }
    if optimizer.trust_region().is_morbo()
        && candidate_rv == crate::candidates::CandidateRV::Sobol
    {
        if let Some(engine) = optimizer.sobol_engine_mut() {
            return generate_sobol_masked(
                x_center,
                lower_1d,
                upper_1d,
                num_candidates,
                rng,
                engine,
                20,
            );
        }
    }
    generate_candidates(
        || (lower_1d.clone(), upper_1d.clone()),
        x_center,
        None,
        num_candidates,
        candidate_rv,
        rng,
        optimizer.sobol_engine_mut(),
        20,
    )
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


    let num_dim = optimizer.num_dim();
    let config = optimizer.config().candidates.clone();
    let num_candidates = config.num_candidates(num_dim, num_arms);
    telemetry.num_candidates = num_candidates;
    if optimizer.trust_region().is_morbo()
        && config.candidate_rv == crate::candidates::CandidateRV::Sobol
    {
        optimizer.reseed_sobol(num_arms)?;
    }

    let x_cand_unit = draw_turbo_candidates(
        optimizer,
        &x_center.view(),
        &lower_1d,
        &upper_1d,
        num_candidates,
        rng,
    )?;


    let start = std::time::Instant::now();
    let selected = select_arms(optimizer, &x_cand_unit.view(), num_arms, rng)?;
    telemetry.dt_sel = start.elapsed().as_secs_f64();

    Ok(selected)
}

fn seed_turbo_scale_history(optimizer: &mut Optimizer, prev: usize) {
    if prev == 0 {
        return;
    }
    let Some(y_all) = optimizer.y_obs() else {
        return;
    };
    if y_all.ncols() != 1 || y_all.nrows() < prev {
        return;
    }
    let prefix = y_all.column(0).slice(ndarray::s![..prev]).to_owned();
    optimizer
        .trust_region_mut()
        .turbo_seed_scale_history(&prefix.view());
}

/// Tell for TuRBO phase.
fn tell_turbo(
    optimizer: &mut Optimizer,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
    telemetry: &mut Telemetry,
    rng: &mut dyn RngCore,
) -> Result<(), ENNError> {
    tell_common(optimizer, x, y, yvar, Some(telemetry), rng)?;

    let num_obs = optimizer.obs_count();
    let y_incumbent = optimizer
        .incumbent_y_scalar()
        .ok_or_else(|| ENNError::InvalidParameter("Missing incumbent y".to_string()))?
        .to_owned();
    optimizer.trust_region_mut().set_num_arms(x.nrows());
    if !optimizer.trust_region().is_morbo() {



        if optimizer.trust_region().turbo_prev_num_obs() == 0 {
            let prev = num_obs.saturating_sub(y.nrows());
            seed_turbo_scale_history(optimizer, prev);
            optimizer
                .trust_region_mut()
                .set_turbo_prev_num_obs(prev);
        }
        optimizer.trust_region_mut().tell_update_new_batch(
            y,
            &y_incumbent.view(),
            num_obs,
        )?;
    }
    if optimizer.trust_region().needs_restart() && optimizer.trust_region().is_morbo() {
        optimizer.trust_region_mut().restart(Some(rng));
        optimizer.increment_restart_generation();
        morbo_sync_ranges_from_obs(optimizer)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests_aniso;
#[cfg(test)]
mod tests_init;
#[cfg(test)]
mod tests_morbo_acq;
#[cfg(test)]
mod tests_selection;
