//! Factory functions for creating optimizers with preset configs.

use ndarray::Array2;
use rand::RngCore;

use crate::config::{
    ConfigOverrides, InitStrategy, SurrogateConfig,
    lhd_only_config, turbo_enn_config, turbo_zero_config,
};
use crate::error::ENNError;
use crate::optimizer::Optimizer;
use crate::strategy::Strategy;

/// Create an optimizer for TuRBO-ENN.
pub fn create_optimizer_enn(
    bounds: Array2<f64>,
    k: i32,
    num_init: usize,
    rng: &mut dyn RngCore,
) -> Result<Optimizer, ENNError> {
    create_optimizer_enn_with_overrides(bounds, k, num_init, rng, None)
}

/// Create TuRBO-ENN with optional config overrides (for future Python pass-through).
pub fn create_optimizer_enn_with_overrides(
    bounds: Array2<f64>,
    k: i32,
    num_init: usize,
    rng: &mut dyn RngCore,
    overrides: Option<&ConfigOverrides>,
) -> Result<Optimizer, ENNError> {
    let mut config = turbo_enn_config();
    if let SurrogateConfig::ENN(enn_cfg) = &mut config.surrogate {
        enn_cfg.k = k;
    }
    if let Some(o) = overrides {
        config = o.apply_to(config);
    }
    let strategy = Strategy::hybrid(InitStrategy::LHD, num_init);
    Optimizer::new_with_strategy(bounds, config, strategy, rng)
}

/// Create an optimizer for TuRBO-ZERO.
pub fn create_optimizer_zero(
    bounds: Array2<f64>,
    num_init: usize,
    rng: &mut dyn RngCore,
) -> Result<Optimizer, ENNError> {
    create_optimizer_zero_with_overrides(bounds, num_init, rng, None)
}

/// Create TuRBO-ZERO with optional config overrides.
pub fn create_optimizer_zero_with_overrides(
    bounds: Array2<f64>,
    num_init: usize,
    rng: &mut dyn RngCore,
    overrides: Option<&ConfigOverrides>,
) -> Result<Optimizer, ENNError> {
    let mut config = turbo_zero_config();
    if let Some(o) = overrides {
        config = o.apply_to(config);
    }
    let strategy = Strategy::hybrid(InitStrategy::LHD, num_init);
    Optimizer::new_with_strategy(bounds, config, strategy, rng)
}

/// Create an optimizer for LHD-only.
pub fn create_optimizer_lhd(
    bounds: Array2<f64>,
    num_init: usize,
    rng: &mut dyn RngCore,
) -> Result<Optimizer, ENNError> {
    create_optimizer_lhd_with_overrides(bounds, num_init, rng, None)
}

/// Create LHD-only with optional config overrides.
pub fn create_optimizer_lhd_with_overrides(
    bounds: Array2<f64>,
    num_init: usize,
    rng: &mut dyn RngCore,
    overrides: Option<&ConfigOverrides>,
) -> Result<Optimizer, ENNError> {
    let mut config = lhd_only_config();
    if let Some(o) = overrides {
        config = o.apply_to(config);
    }
    let strategy = Strategy::init(InitStrategy::LHD, num_init);
    Optimizer::new_with_strategy(bounds, config, strategy, rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn create_optimizer_public_wrappers_smoke() {
        let bounds = array![[0.0, 1.0], [0.0, 1.0]];
        let mut rng = StdRng::seed_from_u64(101);

        let mut enn = create_optimizer_enn(bounds.clone(), 3, 2, &mut rng).unwrap();
        let _ = enn.ask(1, &mut rng).unwrap();

        let mut zero = create_optimizer_zero(bounds.clone(), 2, &mut rng).unwrap();
        let _ = zero.ask(1, &mut rng).unwrap();

        let mut lhd = create_optimizer_lhd(bounds, 2, &mut rng).unwrap();
        let _ = lhd.ask(1, &mut rng).unwrap();
    }
}
