//! Morbo incumbent selection should use rng on ties (Python argmax_random_tie).

use crate::config::turbo_enn_config;
use crate::morbo_trust_region::{MorboTRSettings, Rescalarize};
use crate::optimizer::Optimizer;
use crate::strategy::Strategy;
use crate::trust_region::TRLengthConfig;
use crate::trust_region_config::TrustRegionConfig;
use ndarray::array;
use rand::rngs::StdRng;
use rand::SeedableRng;

fn morbo_optimizer_with_tied_obs(seed: u64) -> Optimizer {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(seed);
    let mut cfg = turbo_enn_config();
    cfg.trust_region = TrustRegionConfig::Morbo(MorboTRSettings {
        num_metrics: 2,
        alpha: 0.05,
        length: TRLengthConfig::default(),
        rescalarize: Rescalarize::OnRestart,
        noise_aware: false,
    });
    let mut opt =
        Optimizer::new_with_strategy(bounds, cfg, Strategy::turbo(), &mut rng).unwrap();
    let y_fit = array![[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]];
    let y_inc = array![1.0, 2.0];
    let morbo = opt.trust_region_mut().morbo_mut().expect("morbo tr");
    morbo.set_num_arms(2);
    morbo.update(&y_fit.view(), &y_inc.view()).unwrap();
    let x = array![[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]];
    let y = array![[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]];
    let delta = opt.add_observations(&x.view(), &y.view()).unwrap();
    let surrogate = opt.surrogate_mut().expect("enn surrogate");
    surrogate
        .fit_append(
            &delta.x_new_view(),
            &delta.y_new_view(),
            None,
            &mut rng,
        )
        .unwrap();
    opt
}

#[test]
fn morbo_incumbent_tie_break_should_vary_with_rng() {
    let mut indices = Vec::new();
    for seed in 0..40u64 {
        let mut opt = morbo_optimizer_with_tied_obs(seed);
        let mut rng = StdRng::seed_from_u64(seed + 1000);
        opt.update_incumbent(&mut rng).unwrap();
        let x_inc = opt.incumbent_x_unit().expect("incumbent x");
        indices.push(format!("{:.8},{:.8}", x_inc[0], x_inc[1]));
    }
    let unique: std::collections::HashSet<String> = indices.into_iter().collect();
    assert!(
        unique.len() > 1,
        "Morbo incumbent should break Chebyshev ties with rng"
    );
}
