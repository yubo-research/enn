//! Factory functions for creating optimizers with preset configs.

use ndarray::Array2;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::config::{
    lhd_only_config, turbo_enn_config, turbo_zero_config, ConfigOverrides, InitStrategy,
    OptimizerInitKind, SurrogateConfig,
};
use crate::error::ENNError;
use crate::optimizer::Optimizer;
use crate::strategy::Strategy;

/// Default neighbor count when `k` is omitted.
pub const DEFAULT_ENN_K: i32 = 10;
/// Default initialization budget when `num_init` is omitted.
pub const DEFAULT_NUM_INIT: usize = 10;

fn resolve_k(k: Option<i32>) -> i32 {
    k.unwrap_or(DEFAULT_ENN_K)
}

fn resolve_num_init(num_init: Option<usize>) -> usize {
    num_init.unwrap_or(DEFAULT_NUM_INIT)
}

/// Create an optimizer for TuRBO-ENN.
///
/// `k` and `num_init` use [`DEFAULT_ENN_K`] and [`DEFAULT_NUM_INIT`] when omitted.
/// `seed` initializes the optimizer's owned `StdRng` once.
pub fn create_optimizer_enn(
    bounds: Array2<f64>,
    k: Option<i32>,
    num_init: Option<usize>,
    seed: u64,
) -> Result<Optimizer, ENNError> {
    create_optimizer_enn_with_overrides(bounds, k, num_init, seed, None)
}

/// Create TuRBO-ENN with optional config overrides (for future Python pass-through).
pub fn create_optimizer_enn_with_overrides(
    bounds: Array2<f64>,
    k: Option<i32>,
    num_init: Option<usize>,
    seed: u64,
    overrides: Option<&ConfigOverrides>,
) -> Result<Optimizer, ENNError> {
    let mut config = turbo_enn_config();
    if let SurrogateConfig::ENN(enn_cfg) = &mut config.surrogate {
        enn_cfg.k = resolve_k(k);
    }
    if let Some(o) = overrides {
        config = o.apply_to(config)?;
    }
    config.validate_kind(OptimizerInitKind::Hybrid)?;
    let num_init = resolve_num_init(num_init);
    let mut rng = StdRng::seed_from_u64(seed);
    let strategy = Strategy::hybrid(InitStrategy::LHD, num_init);
    Optimizer::new_with_strategy(bounds, config, strategy, &mut rng)
}

/// Create an optimizer for TuRBO-ZERO.
pub fn create_optimizer_zero(
    bounds: Array2<f64>,
    num_init: Option<usize>,
    seed: u64,
) -> Result<Optimizer, ENNError> {
    create_optimizer_zero_with_overrides(bounds, num_init, seed, None)
}

/// Create TuRBO-ZERO with optional config overrides.
pub fn create_optimizer_zero_with_overrides(
    bounds: Array2<f64>,
    num_init: Option<usize>,
    seed: u64,
    overrides: Option<&ConfigOverrides>,
) -> Result<Optimizer, ENNError> {
    let mut config = turbo_zero_config();
    if let Some(o) = overrides {
        config = o.apply_to(config)?;
    }
    config.validate_kind(OptimizerInitKind::Hybrid)?;
    let num_init = resolve_num_init(num_init);
    let mut rng = StdRng::seed_from_u64(seed);
    let strategy = Strategy::hybrid(InitStrategy::LHD, num_init);
    Optimizer::new_with_strategy(bounds, config, strategy, &mut rng)
}

/// Create an optimizer for LHD-only.
pub fn create_optimizer_lhd(
    bounds: Array2<f64>,
    num_init: Option<usize>,
    seed: u64,
) -> Result<Optimizer, ENNError> {
    create_optimizer_lhd_with_overrides(bounds, num_init, seed, None)
}

/// Create LHD-only with optional config overrides.
pub fn create_optimizer_lhd_with_overrides(
    bounds: Array2<f64>,
    num_init: Option<usize>,
    seed: u64,
    overrides: Option<&ConfigOverrides>,
) -> Result<Optimizer, ENNError> {
    let mut config = lhd_only_config();
    if let Some(o) = overrides {
        config = o.apply_to(config)?;
    }
    config.validate_kind(OptimizerInitKind::LhdOnly)?;
    let num_init = resolve_num_init(num_init);
    let mut rng = StdRng::seed_from_u64(seed);
    let strategy = Strategy::init(InitStrategy::LHD, num_init);
    Optimizer::new_with_strategy(bounds, config, strategy, &mut rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn create_optimizer_public_wrappers_smoke() {
        let bounds = array![[0.0, 1.0], [0.0, 1.0]];

        let mut enn = create_optimizer_enn(bounds.clone(), Some(3), Some(2), 101).unwrap();
        let _ = enn.ask(1).unwrap();

        let mut zero = create_optimizer_zero(bounds.clone(), Some(2), 101).unwrap();
        let _ = zero.ask(1).unwrap();

        let mut lhd = create_optimizer_lhd(bounds, Some(2), 101).unwrap();
        let _ = lhd.ask(1).unwrap();
    }
}
