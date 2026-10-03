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

    let candidates = optimizer.ask(5).unwrap();
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

    let candidates = optimizer.ask(5).unwrap();
    let y = array![[1.0], [2.0], [3.0], [4.0], [5.0]];

    optimizer
        .tell(&candidates.view(), &y.view(), None)
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

    let mut enn =
        create_optimizer_enn_with_overrides(bounds.clone(), Some(3), Some(2), 0, None).unwrap();
    let mut zero = create_optimizer_zero_with_overrides(bounds.clone(), Some(2), 0, None).unwrap();
    let mut lhd = create_optimizer_lhd_with_overrides(bounds, Some(2), 0, None).unwrap();

    assert_eq!(enn.telemetry().dt_fit, 0.0);
    assert_eq!(zero.telemetry().dt_sel, 0.0);
    assert_eq!(lhd.telemetry().dt_gen, 0.0);

    let x0 = enn.ask(1).unwrap();
    let y0 = array![[0.1]];
    enn.tell(&x0.view(), &y0.view(), None).unwrap();

    let x1 = zero.ask(1).unwrap();
    let y1 = array![[0.2]];
    zero.tell(&x1.view(), &y1.view(), None).unwrap();

    let x2 = lhd.ask(1).unwrap();
    let y2 = array![[0.3]];
    lhd.tell(&x2.view(), &y2.view(), None).unwrap();
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

    let yw = optimizer.obs_access().y_obs_warped().unwrap();
    assert_eq!(yw.shape(), &[1, 2]);
    assert!((yw[[0, 0]] - 0.5).abs() < 1e-12);
}

#[test]
fn test_noise_aware_config_and_incumbent_after_tell() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let overrides = ConfigOverrides {
        noise_aware: Some(true),
        ..Default::default()
    };
    let mut opt =
        create_optimizer_enn_with_overrides(bounds, Some(3), Some(0), 0, Some(&overrides)).unwrap();
    assert!(opt.config().noise_aware);

    let x = array![
        [0.2, 0.3],
        [0.4, 0.5],
        [0.6, 0.7],
        [0.8, 0.9],
    ];
    let y = array![[0.0], [1.0], [2.0], [0.5]];
    opt.tell(&x.view(), &y.view(), None).unwrap();
    assert!(opt.incumbent_x_unit().is_some());
    assert_eq!(opt.incumbent_tracker.observation_count(), 4);
}

#[test]
fn reset_incumbent_tracker_desyncs_count_from_obs() {


    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let overrides = ConfigOverrides {
        noise_aware: Some(true),
        ..Default::default()
    };
    let mut opt =
        create_optimizer_enn_with_overrides(bounds, Some(3), Some(0), 0, Some(&overrides)).unwrap();
    let x = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]];
    let y = array![[0.0], [1.0], [0.5]];
    opt.tell(&x.view(), &y.view(), None).unwrap();
    assert_eq!(opt.incumbent_tracker.observation_count(), opt.obs_count());
    opt.reset_incumbent_tracker();
    assert_eq!(opt.incumbent_tracker.observation_count(), 0);
    assert!(opt.obs_count() > 0);
}

#[test]
fn local_restart_drops_rows_and_asks_a_new_lhd() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut opt =
        create_optimizer_enn_with_overrides(bounds, Some(3), Some(4), 0, None).unwrap();
    let x = opt.ask(4).unwrap();
    let y = ndarray::Array2::from_elem((4, 1), 1.0);
    opt.tell(&x.view(), &y.view(), None).unwrap();
    assert_eq!(opt.obs_count(), 4);
    opt.begin_local_restart().unwrap();
    assert_eq!(opt.obs_count(), 0);
    assert_eq!(opt.reinit_left(), Some(4));
    assert!(opt.incumbent_x().is_none());
    let x2 = opt.ask(4).unwrap();
    assert_eq!(x2.nrows(), 4);
    let y2 = ndarray::Array2::from_elem((4, 1), 2.0);
    opt.tell(&x2.view(), &y2.view(), None).unwrap();
    assert_eq!(opt.obs_count(), 4);
    assert_eq!(opt.reinit_left(), None);
    let y_obs = opt.y_obs().unwrap();
    assert!(y_obs.iter().all(|&v| (v - 2.0).abs() < 1e-9));
}

#[test]
fn turbo_length_restart_keeps_incumbent_tracker_synced() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let overrides = ConfigOverrides {
        noise_aware: Some(true),
        ..Default::default()
    };
    let mut opt =
        create_optimizer_enn_with_overrides(bounds, Some(3), Some(0), 0, Some(&overrides)).unwrap();

    for i in 0..64 {
        let x = array![[0.1 + 0.01 * (i as f64), 0.2]];
        let y = array![[-(i as f64)]];
        opt.tell(&x.view(), &y.view(), None).unwrap();
        assert_eq!(
            opt.incumbent_tracker.observation_count(),
            opt.obs_count(),
            "tracker must stay synced after tell (incl. post-restart)"
        );
    }
    assert!(
        opt.trust_region().needs_restart(),
        "length should be below the minimum while rows are still fitted"
    );
    assert_eq!(opt.restart_generation(), 0);
    assert!(opt.obs_count() > 0);
}

#[test]
fn collapse_keeps_posterior_until_the_next_ask() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut opt =
        create_optimizer_enn_with_overrides(bounds, Some(3), Some(1), 0, None).unwrap();
    let x0 = opt.ask(1).unwrap();
    opt.tell(&x0.view(), &array![[100.0]].view(), None).unwrap();
    let mut last_x = x0;
    for i in 0..40 {
        last_x = opt.ask(1).unwrap();
        opt.tell(&last_x.view(), &array![[-(i as f64)]].view(), None)
            .unwrap();
        if opt.trust_region().needs_restart() {
            break;
        }
    }
    assert!(opt.trust_region().needs_restart());
    assert!(opt.obs_count() > 1);
    let mu = opt.posterior_mu(&last_x.view()).unwrap();
    assert!(mu[[0, 0]].is_finite());
    let restarted = opt.ask(1).unwrap();
    assert_eq!(restarted.nrows(), 1);
    assert_eq!(opt.obs_count(), 0);
    assert_eq!(opt.reinit_left(), Some(1));
    opt.tell(&restarted.view(), &array![[5.0]].view(), None)
        .unwrap();
    assert_eq!(opt.obs_count(), 1);
    let y_obs = opt.y_obs().unwrap();
    assert!((y_obs[[0, 0]] - 5.0).abs() < 1e-9);
}

#[test]
fn ask_lies_within_natural_bounds_and_rejects_zero_arms() {
    let bounds = array![[-2.0, 3.0], [10.0, 12.0]];
    let mut rng = StdRng::seed_from_u64(7);
    let mut opt = Optimizer::new(bounds.clone(), lhd_only_config(), &mut rng).unwrap();
    assert!(opt.ask(0).is_err());
    let x = opt.ask(4).unwrap();
    for i in 0..x.nrows() {
        for j in 0..x.ncols() {
            assert!(x[[i, j]] >= bounds[[j, 0]] && x[[i, j]] <= bounds[[j, 1]]);
        }
    }
    let empty = array![[0.0, 0.0]];
    let empty = empty.slice(ndarray::s![0..0, ..]);
    opt.tell(&empty, &array![[1.0]].slice(ndarray::s![0..0, ..]), None)
        .unwrap();
    opt.tell(&x.view(), &ndarray::Array2::ones((4, 1)).view(), None)
        .unwrap();
    assert!(opt
        .tell(&x.view(), &ndarray::Array2::ones((4, 1)).view(), Some(&ndarray::Array2::ones((4, 1)).view()))
        .is_err());
}

#[test]
fn kiss_optimizer_bounds_and_seed_helpers() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut rng = StdRng::seed_from_u64(1);
    let mut opt = Optimizer::new(bounds, turbo_zero_config(), &mut rng).unwrap();
    assert_eq!(opt.bounds().nrows(), 2);
    assert_eq!(opt.restart_generation(), 0);
    let _ = opt.sobol_seed_base();
    opt.increment_restart_generation();
    assert_eq!(opt.restart_generation(), 1);
}
