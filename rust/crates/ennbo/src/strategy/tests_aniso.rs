//! Anisotropic trust region under DiskAuto: side lengths and RAASP coordinate rates.

use super::draw_turbo_candidates;
use crate::config::{turbo_enn_config, SurrogateConfig};
use crate::layout::EnnLayout;
use crate::metric_weights::trust_region_sides;
use crate::optimizer::Optimizer;
use crate::strategy::Strategy;
use crate::surrogate::ENNSurrogateConfig;
use crate::trust_region::{TRLengthConfig, TurboTrustRegion};
use ndarray::{array, Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

fn turbo_tr(num_dim: usize, length: f64) -> TurboTrustRegion {
    TurboTrustRegion::new(num_dim, TRLengthConfig::new(length, 0.01, 1.6))
}

#[test]
fn equal_weights_give_the_cube() {
    for w in [vec![1.0, 1.0, 1.0], vec![3.0, 3.0]] {
        let s = Array1::from(trust_region_sides(&w));
        let tr = turbo_tr(s.len(), 0.4);
        let center = Array1::from_elem(s.len(), 0.5);
        let (lb, ub) = tr.compute_bounds_1d(&center.view(), Some(&s.view()));
        for j in 0..s.len() {
            assert!((center[j] - lb[j] - 0.2).abs() < 1e-12);
            assert!((ub[j] - center[j] - 0.2).abs() < 1e-12);
        }
    }
}

#[test]
fn unequal_weights_scale_sides_and_keep_volume() {
    let s = Array1::from(trust_region_sides(&[1.0, 4.0]));
    assert!((s[0] - 2.0).abs() < 1e-12 && (s[1] - 0.5).abs() < 1e-12);
    let length = 0.4;
    let tr = turbo_tr(2, length);
    let center = array![0.5, 0.5];
    let (lb, ub) = tr.compute_bounds_1d(&center.view(), Some(&s.view()));
    for j in 0..2 {
        assert!((ub[j] - center[j] - s[j] * length / 2.0).abs() < 1e-12);
        assert!((center[j] - lb[j] - s[j] * length / 2.0).abs() < 1e-12);
    }
    let volume: f64 = (&ub - &lb).iter().product();
    assert!((volume - length.powi(2)).abs() < 1e-12);
}

#[test]
fn tied_weights_share_a_side_and_boundary_sides_stay_in_cube() {
    let s = Array1::from(trust_region_sides(&[2.0, 2.0, 8.0]));
    assert!((s[0] - s[1]).abs() < 1e-12 && s[2] < s[0]);
    let tr = turbo_tr(3, 1.2);
    let center = array![0.02, 0.98, 0.5];
    let (lb, ub) = tr.compute_bounds_1d(&center.view(), Some(&s.view()));
    for j in 0..3 {
        assert!(lb[j] >= 0.0 && ub[j] <= 1.0 && lb[j] <= center[j] && center[j] <= ub[j]);
    }
}

#[test]
fn morbo_bounds_use_the_same_sides() {
    use crate::morbo_trust_region::{MorboTRSettings, MorboTrustRegion, Rescalarize};
    let length = TRLengthConfig::default().length_init;
    let settings = MorboTRSettings {
        num_metrics: 2,
        alpha: 0.05,
        length: TRLengthConfig::default(),
        rescalarize: Rescalarize::OnRestart,
        noise_aware: false,
    };
    let mut rng = StdRng::seed_from_u64(9);
    let tr = MorboTrustRegion::new(2, settings, &mut rng).unwrap();
    let s = Array1::from(trust_region_sides(&[1.0, 4.0]));
    let center = array![0.5, 0.5];
    let (lb, ub) = tr.compute_bounds_1d(&center.view(), Some(&s.view()));
    for j in 0..2 {
        let half = s[j] * length / 2.0;
        assert!((lb[j] - (center[j] - half).clamp(0.0, 1.0)).abs() < 1e-12);
        assert!((ub[j] - (center[j] + half).clamp(0.0, 1.0)).abs() < 1e-12);
    }
}

fn disk_auto_optimizer(dir: &std::path::Path, num_dim: usize, seed: u64) -> Optimizer {
    let layout = EnnLayout::DiskAuto {
        work_dir: dir.to_path_buf(),
    };
    let mut cfg = turbo_enn_config();
    cfg.surrogate = SurrogateConfig::ENN(ENNSurrogateConfig {
        k: 10,
        num_fit_candidates: 4,
        num_fit_samples: 4,
        layout,
        ..Default::default()
    });
    let bounds = Array2::from_shape_fn((num_dim, 2), |(_, j)| j as f64);
    let mut rng = StdRng::seed_from_u64(seed);
    Optimizer::new_with_strategy(bounds, cfg, Strategy::turbo(), &mut rng).unwrap()
}

#[test]
fn raasp_rates_stay_uniform_under_unequal_auto_weights() {
    let dir = tempfile::TempDir::new().unwrap();
    let num_dim = 20;
    let mut opt = disk_auto_optimizer(dir.path(), num_dim, 3);
    let mut rng = StdRng::seed_from_u64(4);
    let x = Array2::from_shape_fn((300, num_dim), |_| rng.gen::<f64>());
    let y = Array2::from_shape_fn((300, 1), |(i, _)| (6.0 * x[[i, 0]]).sin());
    opt.tell(&x.view(), &y.view(), None).unwrap();

    let s = opt.surrogate().and_then(|s| s.lengthscales()).expect("DiskAuto sides");
    assert_eq!(s.len(), num_dim);
    assert!(s[0] < 0.5 * s[1], "x0 should get the shortest side: {s:?}");

    let center = Array1::from_elem(num_dim, 0.5);
    let (lb, ub) = opt.trust_region().compute_bounds_1d(&center.view(), Some(&s.view()));
    let num_candidates = 4000;
    let cand =
        draw_turbo_candidates(&mut opt, &center.view(), &lb, &ub, num_candidates, &mut rng).unwrap();
    let rates: Vec<f64> = (0..num_dim)
        .map(|j| {
            let moved = (0..num_candidates).filter(|&i| cand[[i, j]] != center[j]).count();
            moved as f64 / num_candidates as f64
        })
        .collect();
    let mean = rates.iter().sum::<f64>() / num_dim as f64;
    for (j, r) in rates.iter().enumerate() {
        assert!((r - mean).abs() < 0.06, "dim {j} rate {r} vs mean {mean}");
    }
}
