use super::{
    ask_init, ask_init_hybrid, ask_turbo, select_with_pareto, select_with_random,
    select_with_thompson, select_with_ucb, tell_common, tell_init, tell_turbo, InitStrategy,
    InitStrategyState, Strategy, TurboStrategyState,
};
use crate::config::{turbo_enn_config, turbo_zero_config, AcquisitionConfig};
use crate::optimizer::{Optimizer, Telemetry};
use ndarray::array;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[test]
fn test_private_strategy_helpers_directly() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(77);
    let mut optimizer =
        Optimizer::new_with_strategy(bounds, turbo_zero_config(), Strategy::turbo(), &mut rng)
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
        None,
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
        None,
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
    optimizer
        .tell(&x_fit.view(), &y_fit.view(), None, &mut rng)
        .unwrap();

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
    let mut optimizer2 = Optimizer::new_with_strategy(
        array![[0.0, 1.0], [0.0, 1.0]],
        cfg2,
        Strategy::turbo(),
        &mut rng2,
    )
    .unwrap();
    optimizer2
        .tell(&x_fit.view(), &y_fit.view(), None, &mut rng2)
        .unwrap();

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
    optimizer.tell(&x1.view(), &y1.view(), None, &mut rng).unwrap();

    let progress = optimizer.init_progress();
    assert!(progress.is_some(), "Should be in init phase");
    let (done, total) = progress.unwrap();
    assert_eq!(done, 2);
    assert_eq!(total, 4);

    let x2 = optimizer.ask(2, &mut rng).unwrap();
    let y2 = array![[0.3], [0.4]];
    optimizer.tell(&x2.view(), &y2.view(), None, &mut rng).unwrap();

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
    optimizer
        .tell(&x_fit.view(), &y_fit.view(), None, &mut rng)
        .unwrap();

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
    let mut opt_no_sur = Optimizer::new_with_strategy(
        bounds.clone(),
        turbo_zero_config(),
        Strategy::turbo(),
        &mut rng,
    )
    .unwrap();
    let x = array![[0.2, 0.3]];
    let y = array![[0.1]];
    tell_common(&mut opt_no_sur, &x.view(), &y.view(), None, None, &mut rng).unwrap();

    let mut opt_enn =
        Optimizer::new_with_strategy(bounds, turbo_enn_config(), Strategy::turbo(), &mut rng)
            .unwrap();
    let x2 = array![[0.0, 0.0], [1.0, 1.0]];
    let y2 = array![[0.0], [1.0]];
    let mut tel = Telemetry::default();
    tell_common(
        &mut opt_enn,
        &x2.view(),
        &y2.view(),
        None,
        Some(&mut tel),
        &mut rng,
    )
    .unwrap();
    assert!(tel.dt_fit > 0.0);
}

#[test]
fn ask_scores_full_configured_candidate_pool() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(201);
    let mut config = turbo_enn_config();
    let num_candidates = 37usize;
    config.candidates.num_candidates_factor = 1.0;
    config.candidates.num_candidates_per_arm = Some(num_candidates);
    let num_arms = 2usize;
    let expected_pool = config.candidates.num_candidates(2, num_arms);
    let mut opt =
        Optimizer::new_with_strategy(bounds, config, Strategy::turbo(), &mut rng).unwrap();
    let xf = array![[0.0, 0.0], [1.0, 1.0], [0.2, 0.8]];
    let yf = array![[0.0], [1.0], [0.5]];
    opt.tell(&xf.view(), &yf.view(), None, &mut rng).unwrap();
    let _ = opt.ask(num_arms, &mut rng).unwrap();
    assert_eq!(
        opt.telemetry().num_candidates, expected_pool,
        "ask must score the full configured RAASP pool (no silent cap)"
    );
    assert!(
        expected_pool >= num_candidates * num_arms,
        "pool size must honor per-arm setting (got {expected_pool}, want at least {})",
        num_candidates * num_arms
    );
}

#[test]
fn select_with_functions_direct_smoke() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(202);
    let x_cand = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]];

    let out_r = select_with_random(&x_cand.view(), 2, &mut rng).unwrap();
    assert_eq!(out_r.nrows(), 2);

    let mut opt =
        Optimizer::new_with_strategy(bounds, turbo_enn_config(), Strategy::turbo(), &mut rng)
            .unwrap();
    let xf = array![[0.0, 0.0], [1.0, 1.0], [0.2, 0.8]];
    let yf = array![[0.0], [1.0], [0.5]];
    opt.tell(&xf.view(), &yf.view(), None, &mut rng).unwrap();
    let sur = opt.surrogate().expect("enn surrogate");

    let out_ts = select_with_thompson(&opt, sur, &x_cand.view(), 2, &mut rng).unwrap();
    assert_eq!(out_ts.nrows(), 2);

    let out_ucb = select_with_ucb(&opt, sur, &x_cand.view(), 2, 1.0, &mut rng).unwrap();
    assert_eq!(out_ucb.nrows(), 2);

    let out_pf = select_with_pareto(sur, &x_cand.view(), 2, &mut rng).unwrap();
    assert_eq!(out_pf.nrows(), 2);
}

#[test]
fn select_with_pareto_scores_naturalized_under_y_bounds() {
    use crate::acquisition::ParetoAcquisition;
    use crate::config::{ConfigOverrides, SurrogateConfig};
    use crate::optimizer_factory::create_optimizer_enn_with_overrides;

    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(303);
    let overrides = ConfigOverrides {
        y_bounds: Some(array![[0.0, 1.0]]),
        acquisition: Some(AcquisitionConfig::Pareto),
        ..Default::default()
    };
    let mut opt =
        create_optimizer_enn_with_overrides(bounds, 3, 0, &mut rng, Some(&overrides)).unwrap();
    if let SurrogateConfig::ENN(enn) = &opt.config().surrogate {
        assert!(enn.y_bounds.is_some());
    }
    let xf = array![[0.0, 0.0], [1.0, 1.0], [0.2, 0.8], [0.7, 0.3]];
    let yf = array![[0.1], [0.9], [0.4], [0.6]];
    opt.tell(&xf.view(), &yf.view(), None, &mut rng).unwrap();
    let sur = opt.surrogate().expect("enn surrogate");
    let x_cand = array![
        [0.1, 0.2],
        [0.3, 0.4],
        [0.5, 0.6],
        [0.7, 0.8],
        [0.15, 0.85],
        [0.55, 0.25]
    ];
    let mut rng_a = StdRng::seed_from_u64(404);
    let mut rng_b = StdRng::seed_from_u64(404);
    let out = select_with_pareto(sur, &x_cand.view(), 2, &mut rng_a).unwrap();
    let pred_nat = sur.naturalize_prediction(sur.predict(&x_cand.view()).unwrap());
    let idx = ParetoAcquisition::new()
        .select(&pred_nat.mu.view(), &pred_nat.se.view(), 2, &mut rng_b)
        .unwrap();
    let expected = x_cand.select(ndarray::Axis(0), &idx);
    assert_eq!(out, expected, "Pareto must rank naturalized μ/σ under y_bounds");
}

#[test]
fn select_with_thompson_scores_naturalized_under_y_bounds() {
    use crate::config::ConfigOverrides;
    use crate::optimizer_factory::create_optimizer_enn_with_overrides;

    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(505);
    let overrides = ConfigOverrides {
        y_bounds: Some(array![[0.0, 1.0]]),
        acquisition: Some(AcquisitionConfig::Thompson),
        ..Default::default()
    };
    let mut opt =
        create_optimizer_enn_with_overrides(bounds, 3, 0, &mut rng, Some(&overrides)).unwrap();
    let xf = array![[0.0, 0.0], [1.0, 1.0], [0.2, 0.8], [0.7, 0.3]];
    let yf = array![[0.1], [0.9], [0.4], [0.6]];
    opt.tell(&xf.view(), &yf.view(), None, &mut rng).unwrap();
    let sur = opt.surrogate().expect("enn surrogate");
    let x_cand = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]];
    let mut rng_sel = StdRng::seed_from_u64(606);
    let out = select_with_thompson(&opt, sur, &x_cand.view(), 2, &mut rng_sel).unwrap();
    assert_eq!(out.nrows(), 2);

    // Same RNG seed → same warped draws; ranking must use naturalized values.
    let mut rng_ref = StdRng::seed_from_u64(606);
    let samples = sur.sample(&x_cand.view(), 2, &mut rng_ref).unwrap();
    let mut flat = ndarray::Array2::zeros((x_cand.nrows(), 1));
    for i in 0..x_cand.nrows() {
        flat[[i, 0]] = samples[[0, i, 0]];
    }
    let nat = sur.naturalize_observations_y(flat.clone());
    let mut warped_order: Vec<usize> = (0..x_cand.nrows()).collect();
    warped_order.sort_by(|&a, &b| flat[[b, 0]].partial_cmp(&flat[[a, 0]]).unwrap());
    let mut nat_order: Vec<usize> = (0..x_cand.nrows()).collect();
    nat_order.sort_by(|&a, &b| nat[[b, 0]].partial_cmp(&nat[[a, 0]]).unwrap());
    let expected = x_cand.select(ndarray::Axis(0), &nat_order[..2]);
    assert_eq!(
        out, expected,
        "scalar Thompson must rank naturalized draws; warped_order={warped_order:?} nat_order={nat_order:?}"
    );
}
