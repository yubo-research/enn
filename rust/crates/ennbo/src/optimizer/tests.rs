use super::{Optimizer, Telemetry};
use crate::config::{lhd_only_config, turbo_zero_config, ConfigOverrides};
use crate::error::ENNError;
use crate::optimizer_factory::{
    create_optimizer_enn_with_overrides, create_optimizer_lhd_with_overrides,
    create_optimizer_zero_with_overrides,
};
use ndarray::array;
use rand::rngs::StdRng;
use rand::SeedableRng;

#[test]
fn test_optimizer_creation() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(42);

    let optimizer = Optimizer::new(bounds, turbo_zero_config(), &mut rng).unwrap();

    assert_eq!(optimizer.num_dim(), 2);
    assert!(optimizer.surrogate().is_none());
}

#[test]
fn test_optimizer_ask() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(42);

    let mut optimizer = Optimizer::new(bounds, lhd_only_config(), &mut rng).unwrap();

    let candidates = optimizer.ask(5, &mut rng).unwrap();
    assert_eq!(candidates.nrows(), 5);
    assert_eq!(candidates.ncols(), 2);
}

#[test]
fn test_add_observations_returns_delta() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(42);
    let mut optimizer = Optimizer::new(bounds, turbo_zero_config(), &mut rng).unwrap();
    let x = array![[0.1, 0.2]];
    let y = array![[1.0]];
    let delta = optimizer.add_observations(&x.view(), &y.view()).unwrap();
    assert_eq!(delta.old_n, 0);
    assert_eq!(delta.new_n, 1);
    assert_eq!(delta.x_new_view().nrows(), 1);
    assert_eq!(delta.y_new_view().nrows(), 1);
}

#[test]
fn test_add_observations() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(42);

    let mut optimizer = Optimizer::new(bounds, turbo_zero_config(), &mut rng).unwrap();

    let x = array![[0.5, 0.5], [0.3, 0.7]];
    let y = array![[1.0], [2.0]];

    optimizer.add_observations(&x.view(), &y.view()).unwrap();

    assert_eq!(optimizer.x_obs().unwrap().nrows(), 2);
    assert_eq!(optimizer.y_obs().unwrap().nrows(), 2);
}

#[test]
fn test_add_observations_mismatched_rows_returns_error() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(42);

    let mut optimizer = Optimizer::new(bounds, turbo_zero_config(), &mut rng).unwrap();

    let x = array![[0.5, 0.5], [0.3, 0.7], [0.1, 0.9]];
    let y = array![[1.0], [2.0]];

    let err = optimizer
        .add_observations(&x.view(), &y.view())
        .expect_err("expected InvalidShape for mismatched row counts");
    assert!(matches!(err, ENNError::InvalidShape { .. }));
}

#[test]
fn test_tell() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(42);

    let mut optimizer = Optimizer::new(bounds, lhd_only_config(), &mut rng).unwrap();

    let candidates = optimizer.ask(5, &mut rng).unwrap();
    let y = array![[1.0], [2.0], [3.0], [4.0], [5.0]];

    optimizer
        .tell(&candidates.view(), &y.view(), &mut rng)
        .unwrap();

    assert!(optimizer.telemetry().dt_tell > 0.0);
}

#[test]
fn test_create_optimizer_factories_and_telemetry_defaults() {
    let t = Telemetry::default();
    assert_eq!(t.dt_fit, 0.0);
    assert_eq!(t.dt_gen, 0.0);
    assert_eq!(t.dt_sel, 0.0);
    assert_eq!(t.dt_tell, 0.0);

    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(123);

    let mut enn =
        create_optimizer_enn_with_overrides(bounds.clone(), 3, 2, &mut rng, None).unwrap();
    let mut zero = create_optimizer_zero_with_overrides(bounds.clone(), 2, &mut rng, None).unwrap();
    let mut lhd = create_optimizer_lhd_with_overrides(bounds, 2, &mut rng, None).unwrap();

    assert_eq!(enn.telemetry().dt_fit, 0.0);
    assert_eq!(zero.telemetry().dt_sel, 0.0);
    assert_eq!(lhd.telemetry().dt_gen, 0.0);

    let x0 = enn.ask(1, &mut rng).unwrap();
    let y0 = array![[0.1]];
    enn.tell(&x0.view(), &y0.view(), &mut rng).unwrap();

    let x1 = zero.ask(1, &mut rng).unwrap();
    let y1 = array![[0.2]];
    zero.tell(&x1.view(), &y1.view(), &mut rng).unwrap();

    let x2 = lhd.ask(1, &mut rng).unwrap();
    let y2 = array![[0.3]];
    lhd.tell(&x2.view(), &y2.view(), &mut rng).unwrap();
}

#[test]
fn fallback_observations_without_surrogate() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(99);
    let mut optimizer = Optimizer::new(bounds, turbo_zero_config(), &mut rng).unwrap();
    let x1 = array![[1.0, 2.0, 3.0]];
    let y1 = array![[0.5, 1.5]];
    optimizer.add_observations(&x1.view(), &y1.view()).unwrap();
    assert_eq!(optimizer.obs_count(), 1);
    let xa = optimizer.x_obs().unwrap();
    assert_eq!(xa.shape(), &[1, 3]);
    let ya = optimizer.y_obs().unwrap();
    assert_eq!(ya.shape(), &[1, 2]);
    assert!((optimizer.obs_access().obs_row_x(0).unwrap()[[0]] - 1.0).abs() < 1e-12);
}

#[test]
fn test_noise_aware_config_and_incumbent_after_tell() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(55);
    let overrides = ConfigOverrides {
        noise_aware: Some(true),
        ..Default::default()
    };
    let mut opt =
        create_optimizer_enn_with_overrides(bounds, 3, 0, &mut rng, Some(&overrides)).unwrap();
    assert!(opt.config().noise_aware);

    let x = array![
        [0.2, 0.3],
        [0.4, 0.5],
        [0.6, 0.7],
        [0.8, 0.9],
    ];
    let y = array![[0.0], [1.0], [2.0], [0.5]];
    opt.tell(&x.view(), &y.view(), &mut rng).unwrap();
    assert!(opt.incumbent_x_unit().is_some());
    assert_eq!(opt.incumbent_tracker.observation_count(), 4);
}
