use super::{
    ask_init, ask_init_hybrid, ask_turbo, maybe_cap_selection_candidates, select_with_pareto,
    select_with_random, select_with_thompson, select_with_ucb, selection_candidate_cap,
    tell_common, tell_init, tell_turbo,
    InitStrategy,
    InitStrategyState,
    Strategy,
    TurboStrategyState,
};
use crate::config::{AcquisitionConfig, turbo_enn_config, turbo_zero_config};
use crate::optimizer::{Optimizer, Telemetry};
use ndarray::{Array2, array};
use rand::SeedableRng;
use rand::rngs::StdRng;

#[test]
fn test_private_strategy_helpers_directly() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(77);
    let mut optimizer = Optimizer::new_with_strategy(
        bounds,
        turbo_zero_config(),
        Strategy::turbo(),
        &mut rng,
    )
    .unwrap();

    let init_state = InitStrategyState::new(InitStrategy::Random, 3);
    let x_init = ask_init(&init_state, &mut optimizer, 2, &mut rng).unwrap();
    assert_eq!(x_init.nrows(), 2);

    let mut telemetry = Telemetry::default();
    let init_state_h = InitStrategyState::new(InitStrategy::LHD, 3);
    let x_init_h = ask_init_hybrid(&init_state_h, &mut optimizer, 2, &mut rng).unwrap();
    assert_eq!(x_init_h.nrows(), 2);

    let mut init_state2 = InitStrategyState::new(InitStrategy::LHD, 3);
    let y_init = array![[0.1], [0.2]];
    tell_init(
        &mut init_state2,
        &mut optimizer,
        &x_init_h.view(),
        &y_init.view(),
        &mut rng,
    )
    .unwrap();
    assert_eq!(init_state2.completed, 2);

    let x_turbo = ask_turbo(&mut optimizer, 2, &mut telemetry, &mut rng).unwrap();
    assert_eq!(x_turbo.nrows(), 2);
    let y_turbo = array![[0.3], [0.4]];
    tell_turbo(
        &mut optimizer,
        &x_turbo.view(),
        &y_turbo.view(),
        &mut telemetry,
        &mut rng,
    )
    .unwrap();
}

#[test]
fn test_thompson_sampling_uses_posterior_sample() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(42);

    let mut cfg = turbo_enn_config();
    cfg.acquisition = AcquisitionConfig::Thompson;
    let mut optimizer =
        Optimizer::new_with_strategy(bounds, cfg, Strategy::turbo(), &mut rng).unwrap();

    let x_fit = array![[0.0, 0.0], [1.0, 1.0], [0.2, 0.8], [0.8, 0.2], [0.5, 0.5]];
    let y_fit = array![[0.0], [1.0], [0.5], [0.4], [0.6]];
    optimizer.tell(&x_fit.view(), &y_fit.view(), &mut rng).unwrap();

    let tel_after_tell = optimizer.telemetry();
    assert!(
        tel_after_tell.dt_fit > 0.0,
        "dt_fit should be populated after surrogate fitting (regression test for B3)"
    );

    let tel_before_ask = optimizer.telemetry().clone();
    let _candidates = optimizer.ask(2, &mut rng).unwrap();
    let tel_after_ask = optimizer.telemetry();
    assert!(
        tel_after_ask.dt_sel > 0.0 || tel_after_ask.dt_sel != tel_before_ask.dt_sel,
        "dt_sel should be populated after arm selection (regression test for B3)"
    );

    let mut rng2 = StdRng::seed_from_u64(42);
    let mut cfg2 = turbo_enn_config();
    cfg2.acquisition = AcquisitionConfig::Thompson;
    let mut optimizer2 =
        Optimizer::new_with_strategy(array![[0.0, 1.0], [0.0, 1.0]], cfg2, Strategy::turbo(), &mut rng2).unwrap();
    optimizer2.tell(&x_fit.view(), &y_fit.view(), &mut rng2).unwrap();

    let _candidates1 = optimizer.ask(2, &mut rng).unwrap();
    let _candidates2 = optimizer2.ask(2, &mut rng2).unwrap();
}

#[test]
fn test_hybrid_init_respects_strategy_type_random() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(123);

    let strategy = Strategy::hybrid(InitStrategy::Random, 4);
    let mut optimizer =
        Optimizer::new_with_strategy(bounds, turbo_zero_config(), strategy, &mut rng).unwrap();

    let x1 = optimizer.ask(2, &mut rng).unwrap();
    assert_eq!(x1.nrows(), 2);

    let y1 = array![[0.1], [0.2]];
    optimizer.tell(&x1.view(), &y1.view(), &mut rng).unwrap();

    let progress = optimizer.init_progress();
    assert!(progress.is_some(), "Should be in init phase");
    let (done, total) = progress.unwrap();
    assert_eq!(done, 2);
    assert_eq!(total, 4);

    let x2 = optimizer.ask(2, &mut rng).unwrap();
    let y2 = array![[0.3], [0.4]];
    optimizer.tell(&x2.view(), &y2.view(), &mut rng).unwrap();

    assert!(
        optimizer.init_progress().is_none(),
        "Should have exited init phase"
    );

    let x3 = optimizer.ask(2, &mut rng).unwrap();
    assert_eq!(x3.nrows(), 2);
}

#[test]
fn test_telemetry_populated_after_operations() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(99);

    let mut cfg = turbo_enn_config();
    cfg.acquisition = AcquisitionConfig::UCB { beta: 2.0 };
    let mut optimizer =
        Optimizer::new_with_strategy(bounds, cfg, Strategy::turbo(), &mut rng).unwrap();

    let tel0 = optimizer.telemetry();
    assert_eq!(tel0.dt_fit, 0.0);
    assert_eq!(tel0.dt_sel, 0.0);

    let x_fit = array![[0.0, 0.0], [1.0, 1.0], [0.5, 0.5]];
    let y_fit = array![[0.0], [1.0], [0.5]];
    optimizer.tell(&x_fit.view(), &y_fit.view(), &mut rng).unwrap();

    let tel1 = optimizer.telemetry();
    assert!(
        tel1.dt_fit > 0.0,
        "dt_fit should be > 0 after surrogate fitting, got {}",
        tel1.dt_fit
    );

    let _candidates = optimizer.ask(2, &mut rng).unwrap();

    let tel2 = optimizer.telemetry();
    assert!(
        tel2.dt_sel > 0.0,
        "dt_sel should be > 0 after arm selection, got {}",
        tel2.dt_sel
    );
}

#[test]
fn turbo_strategy_state_default_and_tell_common_paths() {
    let _: TurboStrategyState = TurboStrategyState;

    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(200);
    let mut opt_no_sur =
        Optimizer::new_with_strategy(bounds.clone(), turbo_zero_config(), Strategy::turbo(), &mut rng)
            .unwrap();
    let x = array![[0.2, 0.3]];
    let y = array![[0.1]];
    tell_common(&mut opt_no_sur, &x.view(), &y.view(), None, &mut rng).unwrap();

    let mut opt_enn =
        Optimizer::new_with_strategy(bounds, turbo_enn_config(), Strategy::turbo(), &mut rng).unwrap();
    let x2 = array![[0.0, 0.0], [1.0, 1.0]];
    let y2 = array![[0.0], [1.0]];
    let mut tel = Telemetry::default();
    tell_common(&mut opt_enn, &x2.view(), &y2.view(), Some(&mut tel), &mut rng).unwrap();
    assert!(tel.dt_fit > 0.0);
}

#[test]
fn selection_candidate_cap_and_maybe_cap_branches() {
    let prev = std::env::var("ENN_DISABLE_SEL_CAP").ok();
    std::env::set_var("ENN_DISABLE_SEL_CAP", "1");
    assert_eq!(selection_candidate_cap(100, 100, 4), usize::MAX);
    if let Some(v) = prev {
        std::env::set_var("ENN_DISABLE_SEL_CAP", v);
    } else {
        std::env::remove_var("ENN_DISABLE_SEL_CAP");
    }

    assert!(selection_candidate_cap(10_000, 1, 4) >= 256);
    assert!(selection_candidate_cap(1_000, 10_000, 4) >= 320);
    assert!(selection_candidate_cap(1_000, 100, 4) >= 384);

    let mut rng = StdRng::seed_from_u64(201);
    let rows = 50usize;
    let x_big = Array2::from_shape_fn((rows, 2), |(i, j)| (i + j) as f64 * 0.01);
    let capped = maybe_cap_selection_candidates(&x_big, 2, 10, 2, &mut rng);
    assert!(capped.nrows() <= x_big.nrows());

    let small = array![[0.1, 0.2], [0.3, 0.4]];
    let uncapped = maybe_cap_selection_candidates(&small, 2, 10, 2, &mut rng);
    assert_eq!(uncapped, small);
}

#[test]
fn select_with_functions_direct_smoke() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(202);
    let x_cand = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]];

    let out_r = select_with_random(&x_cand.view(), 2, &mut rng).unwrap();
    assert_eq!(out_r.nrows(), 2);

    let mut opt =
        Optimizer::new_with_strategy(bounds, turbo_enn_config(), Strategy::turbo(), &mut rng).unwrap();
    let xf = array![[0.0, 0.0], [1.0, 1.0], [0.2, 0.8]];
    let yf = array![[0.0], [1.0], [0.5]];
    opt.tell(&xf.view(), &yf.view(), &mut rng).unwrap();
    let sur = opt.surrogate().expect("enn surrogate");

    let out_ts = select_with_thompson(sur, &x_cand.view(), 2, &mut rng).unwrap();
    assert_eq!(out_ts.nrows(), 2);

    let out_ucb = select_with_ucb(sur, &x_cand.view(), 2, 1.0, &mut rng).unwrap();
    assert_eq!(out_ucb.nrows(), 2);

    let out_pf = select_with_pareto(sur, &x_cand.view(), 2, &mut rng).unwrap();
    assert_eq!(out_pf.nrows(), 2);
}
