use super::observation_store::ObservationStore;
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
fn observation_store_cache_and_edges() {
    let mut store = ObservationStore::new();
    assert!(store.is_empty());
    assert_eq!(store.len(), 0);
    assert!(store.x_obs_array().is_none());
    assert!(store.y_obs_array().is_none());

    let x1 = array![1.0, 2.0, 3.0];
    let y1 = array![0.5, 1.5];
    store.push(x1.clone(), y1.clone());
    assert_eq!(store.len(), 1);
    assert!(!store.is_empty());

    let xa1 = store.x_obs_array().unwrap();
    assert_eq!(xa1.shape(), &[1, 3]);
    let xa2 = store.x_obs_array().unwrap();
    assert_eq!(xa1, xa2);

    let ya1 = store.y_obs_array().unwrap();
    let ya2 = store.y_obs_array().unwrap();
    assert_eq!(ya1.shape(), &[1, 2]);
    assert_eq!(ya1, ya2);

    assert_eq!(store.x_at(0), &x1);
    assert_eq!(store.y_at(0), &y1);
    let idxs: Vec<usize> = (0..store.len()).collect();
    assert_eq!(idxs, vec![0]);

    let x2 = array![0.0, 0.0, 0.0];
    let y2 = array![1.0, 0.0];
    store.push(x2, y2);
    let single_row = ObservationStore::build_array2(&[array![9.0, 8.0]]);
    assert_eq!(single_row.shape(), &[1, 2]);
    assert_eq!(single_row[[0, 0]], 9.0);

    store.replace(vec![array![1.0]], vec![array![2.0]]);
    assert_eq!(store.len(), 1);
    assert!(store.x_obs_array().unwrap()[[0, 0]] - 1.0 < 1e-12);
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
