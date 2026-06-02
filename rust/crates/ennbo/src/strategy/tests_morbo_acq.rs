//! Regression tests for Morbo acquisition / incumbent RNG contracts.

use super::{select_with_thompson, select_with_ucb};
use crate::config::{turbo_enn_config, AcquisitionConfig, InitStrategy};
use approx::relative_eq;
use crate::error::ENNError;
use crate::morbo_trust_region::{MorboTRSettings, Rescalarize};
use crate::optimizer::Optimizer;
use crate::strategy::Strategy;
use crate::surrogate::{Surrogate, SurrogatePrediction};
use crate::trust_region::TRLengthConfig;
use crate::trust_region_config::TrustRegionConfig;
use ndarray::{array, Array1, Array2, Array3, ArrayView2};
use rand::rngs::StdRng;
use rand::RngCore;
use rand::SeedableRng;

struct TieSurrogate {
    sample_value: f64,
    mu: f64,
    se: f64,
}

impl Surrogate for TieSurrogate {
    fn fit(
        &mut self,
        _x: &ArrayView2<f64>,
        _y: &ArrayView2<f64>,
        _yvar: Option<&ArrayView2<f64>>,
        _rng: &mut dyn RngCore,
    ) -> Result<(), ENNError> {
        Ok(())
    }

    fn predict(&self, x: &ArrayView2<f64>) -> Result<SurrogatePrediction, ENNError> {
        let n = x.nrows();
        let m = 2usize;
        Ok(SurrogatePrediction {
            mu: Array2::from_elem((n, m), self.mu),
            se: Array2::from_elem((n, m), self.se),
        })
    }

    fn sample(
        &self,
        x: &ArrayView2<f64>,
        num_samples: usize,
        _rng: &mut dyn RngCore,
    ) -> Result<Array3<f64>, ENNError> {
        let n = x.nrows();
        let m = 2usize;
        Ok(Array3::from_elem((num_samples, n, m), self.sample_value))
    }

    fn lengthscales(&self) -> Option<Array1<f64>> {
        None
    }
}

fn morbo_optimizer_scalarize_ready(seed: u64) -> Optimizer {
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
    let y_inc = array![2.0, 1.0];
    let morbo = opt.trust_region_mut().morbo_mut().expect("morbo tr");
    morbo.set_num_arms(2);
    morbo.update(&y_fit.view(), &y_inc.view()).unwrap();
    opt
}

#[test]
fn morbo_thompson_tie_break_should_vary_with_rng() {
    let opt = morbo_optimizer_scalarize_ready(10);
    let x_cand = array![
        [0.11, 0.21],
        [0.12, 0.22],
        [0.13, 0.23],
        [0.14, 0.24],
    ];
    let sur = TieSurrogate {
        sample_value: 1.0,
        mu: 0.0,
        se: 0.0,
    };
    let mut picks = Vec::new();
    for seed in 0..40u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let out = select_with_thompson(&opt, &sur, &x_cand.view(), 2, &mut rng).unwrap();
        picks.push(format!("{:.8},{:.8}", out[[0, 0]], out[[0, 1]]));
    }
    let unique: std::collections::HashSet<String> = picks.into_iter().collect();
    assert!(
        unique.len() > 1,
        "Morbo Thompson should break ties with rng (match Python argmax_random_tie)"
    );
}

#[test]
fn morbo_ucb_equal_scores_should_vary_with_rng() {
    let opt = morbo_optimizer_scalarize_ready(11);
    let x_cand = array![
        [0.11, 0.21],
        [0.12, 0.22],
        [0.13, 0.23],
        [0.14, 0.24],
        [0.15, 0.25],
        [0.16, 0.26],
    ];
    let sur = TieSurrogate {
        sample_value: 0.0,
        mu: 1.0,
        se: 0.0,
    };
    let mut picks = Vec::new();
    for seed in 0..40u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let out = select_with_ucb(&opt, &sur, &x_cand.view(), 2, 1.0, &mut rng).unwrap();
        picks.push(format!("{:.8},{:.8}", out[[0, 0]], out[[0, 1]]));
    }
    let unique: std::collections::HashSet<String> = picks.into_iter().collect();
    assert!(
        unique.len() > 1,
        "Morbo UCB should shuffle equal scores with rng (match Python permutation+argpartition)"
    );
}

#[test]
fn morbo_pareto_ask_after_multiobjective_tell() {
    let bounds = array![[-1.0, 1.0], [-1.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(14);
    let mut cfg = turbo_enn_config();
    cfg.acquisition = AcquisitionConfig::Pareto;
    cfg.candidates.min_candidates = 32;
    cfg.candidates.num_candidates_factor = 1.0;
    cfg.candidates.num_candidates_per_arm = Some(32);
    cfg.trust_region = TrustRegionConfig::Morbo(MorboTRSettings {
        num_metrics: 2,
        alpha: 0.05,
        length: TRLengthConfig::default(),
        rescalarize: Rescalarize::OnRestart,
        noise_aware: false,
    });
    let mut opt = Optimizer::new_with_strategy(
        bounds,
        cfg,
        Strategy::hybrid(InitStrategy::LHD, 8),
        &mut rng,
    )
    .unwrap();
    let y_fit = array![
        [1.0, 0.5],
        [0.8, 0.9],
        [0.2, 0.3],
        [0.4, 0.7],
        [0.6, 0.2],
        [0.3, 0.8],
        [0.7, 0.4],
        [0.5, 0.6],
    ];
    for i in 0..4 {
        let x = opt.ask(2, &mut rng).unwrap();
        let y = y_fit.slice(ndarray::s![i * 2..i * 2 + 2, ..]);
        opt.tell(&x.view(), &y, &mut rng).unwrap();
    }
    let x_arms = opt.ask(2, &mut rng).unwrap();
    assert_eq!(x_arms.nrows(), 2);
    assert!(opt.trust_region().is_morbo());
}

#[test]
fn morbo_on_restart_rescalarize_via_ask_turbo() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(302);
    let mut cfg = turbo_enn_config();
    cfg.candidates.num_candidates_factor = 1.0;
    cfg.candidates.num_candidates_per_arm = Some(32);
    cfg.trust_region = TrustRegionConfig::Morbo(MorboTRSettings {
        num_metrics: 2,
        alpha: 0.05,
        length: TRLengthConfig::default(),
        rescalarize: Rescalarize::OnRestart,
        noise_aware: false,
    });
    let mut opt = Optimizer::new_with_strategy(
        bounds,
        cfg,
        Strategy::hybrid(InitStrategy::LHD, 8),
        &mut rng,
    )
    .unwrap();
    let y_fit = array![
        [1.0, 0.5],
        [0.8, 0.9],
        [0.2, 0.3],
        [0.4, 0.7],
        [0.6, 0.2],
        [0.3, 0.8],
        [0.7, 0.4],
        [0.5, 0.6],
    ];
    for i in 0..4 {
        let x = opt.ask(2, &mut rng).unwrap();
        let y = y_fit.slice(ndarray::s![i * 2..i * 2 + 2, ..]);
        opt.tell(&x.view(), &y, &mut rng).unwrap();
    }
    let w0 = opt
        .trust_region()
        .morbo()
        .expect("morbo tr")
        .weights()
        .to_owned();
    let _ = opt.ask(2, &mut rng).unwrap();
    let w1 = opt
        .trust_region()
        .morbo()
        .expect("morbo tr")
        .weights()
        .to_owned();
    assert!(relative_eq!(w0, w1, epsilon = 1e-12));
    opt.trust_region_mut().restart(Some(&mut rng));
    let w2 = opt
        .trust_region()
        .morbo()
        .expect("morbo tr")
        .weights()
        .to_owned();
    assert!(!relative_eq!(w1, w2, epsilon = 1e-12));
}

#[test]
fn morbo_on_propose_rescalarize_via_ask_turbo() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(301);
    let mut cfg = turbo_enn_config();
    cfg.candidates.num_candidates_factor = 1.0;
    cfg.candidates.num_candidates_per_arm = Some(32);
    cfg.trust_region = TrustRegionConfig::Morbo(MorboTRSettings {
        num_metrics: 2,
        alpha: 0.05,
        length: TRLengthConfig::default(),
        rescalarize: Rescalarize::OnPropose,
        noise_aware: false,
    });
    let mut opt = Optimizer::new_with_strategy(
        bounds,
        cfg,
        Strategy::hybrid(InitStrategy::LHD, 8),
        &mut rng,
    )
    .unwrap();
    let y_fit = array![
        [1.0, 0.5],
        [0.8, 0.9],
        [0.2, 0.3],
        [0.4, 0.7],
        [0.6, 0.2],
        [0.3, 0.8],
        [0.7, 0.4],
        [0.5, 0.6],
    ];
    for i in 0..4 {
        let x = opt.ask(2, &mut rng).unwrap();
        let y = y_fit.slice(ndarray::s![i * 2..i * 2 + 2, ..]);
        opt.tell(&x.view(), &y, &mut rng).unwrap();
    }
    let w0 = opt
        .trust_region()
        .morbo()
        .expect("morbo tr")
        .weights()
        .to_owned();
    let _ = opt.ask(2, &mut rng).unwrap();
    let w1 = opt
        .trust_region()
        .morbo()
        .expect("morbo tr")
        .weights()
        .to_owned();
    assert!(!relative_eq!(w0, w1, epsilon = 1e-12));
}
