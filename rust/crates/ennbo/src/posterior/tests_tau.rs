//! QA tests for the zero-mean prior precision `tau`:
//! mu = sum(w y) / (tau + sum w), se = y_scale / sqrt(tau + sum w).
use super::light::compute_posterior_light;
use super::*;
use ndarray::{array, Array2};

fn ref_stats(d: &[f64], y: &[f64], epi: f64, ale: f64, tau: f64) -> (f64, f64) {
    let w: Vec<f64> = d.iter().map(|&d| 1.0 / (EPS_VAR + epi * d + ale)).collect();
    let denom = tau + w.iter().sum::<f64>();
    let mu = w.iter().zip(y).map(|(w, y)| w * y).sum::<f64>() / denom;
    (mu, 1.0 / denom.sqrt())
}

fn stats(tau: f64, scale: f64) -> WeightedStatsOut {
    let dist2s = array![[0.1, 0.2, 0.4]];
    let y = array![[1.0], [2.0], [4.0]];
    let params = ENNParams::new(3, 1.5, 0.2).unwrap().with_tau(tau).unwrap();
    let ys = array![scale];
    let s = compute_weighted_stats_impl(
        &dist2s.view(),
        &y.view(),
        None,
        None,
        &params,
        false,
        &ys.view(),
    )
    .unwrap();
    (
        s.mu[[0, 0]],
        s.se[[0, 0]],
        s.w_normalized.sum(),
        s.l2[[0, 0]],
    )
}
type WeightedStatsOut = (f64, f64, f64, f64);

#[test]
fn tau_matches_reference_formula() {
    for tau in [0.0, 0.01, 0.3, 1.0, 10.0, 1000.0] {
        let (mu, se, _, _) = stats(tau, 1.0);
        let (rmu, rse) = ref_stats(&[0.1, 0.2, 0.4], &[1.0, 2.0, 4.0], 1.5, 0.2, tau);
        assert!((mu - rmu).abs() < 1e-12 * rmu.abs().max(1.0), "tau={tau}");
        assert!((se - rse).abs() < 1e-12 * rse, "tau={tau}");
    }
}

#[test]
fn tau_zero_is_identical_to_no_prior() {
    let plain = ENNParams::new(3, 1.5, 0.2).unwrap();
    assert_eq!(plain.tau, 0.0);
    let (mu, se, wsum, _) = stats(0.0, 1.0);
    assert!((wsum - 1.0).abs() < 1e-12);
    let (rmu, rse) = ref_stats(&[0.1, 0.2, 0.4], &[1.0, 2.0, 4.0], 1.5, 0.2, 0.0);
    assert_eq!((mu - rmu).abs() < 1e-12, (se - rse).abs() < 1e-12);
}

#[test]
fn tau_shrinks_mean_and_se_monotonically() {
    let mut prev_mu = f64::INFINITY;
    let mut prev_se = f64::INFINITY;
    for tau in [0.0, 0.01, 0.1, 1.0, 10.0, 100.0] {
        let (mu, se, wsum, _) = stats(tau, 1.0);
        assert!(mu > 0.0 && mu < prev_mu, "mu must shrink toward 0");
        assert!(se < prev_se, "se must decrease as prior adds information");
        assert!(wsum <= 1.0 + 1e-12);
        prev_mu = mu;
        prev_se = se;
    }
}

#[test]
fn tau_large_limit_is_prior() {
    let tau = 1e6;
    let (mu, se, wsum, l2) = stats(tau, 2.0);
    assert!(mu.abs() < 1e-4);
    assert!((se - 2.0 / tau.sqrt()).abs() < 1e-3 * se);
    assert!(wsum < 1e-4 && l2 < 1e-4);
}

#[test]
fn tau_weights_are_consistent_with_mu_and_l2() {
    let (mu, _, wsum, l2) = stats(0.7, 1.0);
    let w: Vec<f64> = [0.1, 0.2, 0.4]
        .iter()
        .map(|d| 1.0 / (EPS_VAR + 1.5 * d + 0.2))
        .collect();
    let denom = 0.7 + w.iter().sum::<f64>();
    assert!((wsum - w.iter().sum::<f64>() / denom).abs() < 1e-12);
    assert!((l2 - w.iter().map(|w| (w / denom).powi(2)).sum::<f64>().sqrt()).abs() < 1e-12);
    assert!(mu > 0.0);
}

#[test]
fn tau_se_scales_with_y_scale() {
    let (_, se1, _, _) = stats(0.5, 1.0);
    let (_, se3, _, _) = stats(0.5, 3.0);
    assert!((se3 - 3.0 * se1).abs() < 1e-12);
}

#[test]
fn tau_multi_metric_and_observation_noise_are_finite() {
    let dist2s = array![[0.1, 0.2], [0.3, 0.4]];
    let y = array![[1.0, -1.0], [2.0, 0.5], [3.0, 1.0], [0.0, 2.0]];
    let params = ENNParams::new(2, 1.0, 0.3).unwrap().with_tau(0.5).unwrap();
    let ys = array![1.0, 2.0];
    for noise in [false, true] {
        let s = compute_weighted_stats_impl(
            &dist2s.view(),
            &y.view(),
            None,
            None,
            &params,
            noise,
            &ys.view(),
        )
        .unwrap();
        assert!(s.mu.iter().chain(s.se.iter()).all(|v| v.is_finite()));
        assert!(s.se.iter().all(|&v| v > 0.0));
        assert!(s.se_ale.iter().all(|&v| v >= 0.0));
        assert_eq!(s.se_ale.iter().any(|&v| v > 0.0), noise);
    }
}

#[test]
fn tau_light_path_matches_full_path() {
    let n = 24;
    let x = Array2::from_shape_fn((n, 3), |(i, j)| i as f64 * 0.1 + j as f64 * 0.02);
    let y = Array2::from_shape_fn((n, 2), |(i, j)| i as f64 + j as f64);
    let model = crate::test_helpers::test_model(x.clone(), y, None, false);
    let flags = PosteriorFlags::new();
    for tau in [0.0, 0.2, 5.0] {
        let p = ENNParams::new(4, 1.0, 0.1).unwrap().with_tau(tau).unwrap();
        let (mu, se, _, _, _) = compute_posterior_light(&model, &x.view(), &p, &flags).unwrap();
        let full = compute_posterior_internals(&model, &x.view(), &p, &flags).unwrap();
        assert!((mu - &full.mu).mapv(f64::abs).iter().all(|&d| d < 1e-12));
        assert!((se - &full.se).mapv(f64::abs).iter().all(|&d| d < 1e-12));
    }
}

#[test]
fn tau_batch_posterior_matches_individual() {
    let x = Array2::from_shape_fn((20, 2), |(i, j)| i as f64 * 0.05 + j as f64 * 0.1);
    let y = Array2::from_shape_fn((20, 1), |(i, _)| (i as f64).sin() + 2.0);
    let model = crate::test_helpers::test_model(x.clone(), y, None, false);
    let flags = PosteriorFlags::new();
    let paramss: Vec<ENNParams> = [0.0, 0.5, 20.0]
        .iter()
        .map(|&t| ENNParams::new(5, 0.7, 0.1).unwrap().with_tau(t).unwrap())
        .collect();
    let batch = model.batch_posterior(&x.view(), &paramss, &flags).unwrap();
    for (i, p) in paramss.iter().enumerate() {
        let one = model.posterior(&x.view(), p, &flags).unwrap();
        for r in 0..x.nrows() {
            assert!((batch.mu[[i, r, 0]] - one.mu[[r, 0]]).abs() < 1e-12);
            assert!((batch.se[[i, r, 0]] - one.se[[r, 0]]).abs() < 1e-12);
        }
    }
}

#[test]
fn tau_draws_have_se_epi_marginal_spread() {
    let x = array![[0.0], [1.0], [2.0]];
    let y = array![[1.0], [1.5], [2.0]];
    let model = crate::test_helpers::test_model(x, y, None, false);
    let q = array![[0.4]];
    let p = ENNParams::new(3, 1.0, 0.0).unwrap().with_tau(2.0).unwrap();
    let internals =
        compute_posterior_internals(&model, &q.view(), &p, &PosteriorFlags::new()).unwrap();
    let seeds: Vec<i64> = (0..4000).collect();
    let draws = draw_from_internals(&model, &internals, &seeds).unwrap();
    let v: Vec<f64> = (0..seeds.len()).map(|s| draws[[s, 0, 0]]).collect();
    let mean = v.iter().sum::<f64>() / v.len() as f64;
    let sd = (v.iter().map(|d| (d - mean).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
    assert!((mean - internals.mu[[0, 0]]).abs() < 0.05 * internals.se_epi[[0, 0]] + 1e-3);
    assert!((sd / internals.se_epi[[0, 0]] - 1.0).abs() < 0.08);
}

#[test]
fn tau_conditional_posterior_is_finite_and_shrunk() {
    let x = array![[0.0], [1.0], [2.0]];
    let y = array![[3.0], [3.5], [4.0]];
    let model = crate::test_helpers::test_model(x, y, None, false);
    let q = array![[0.5]];
    let mu = |tau: f64| {
        let p = ENNParams::new(3, 1.0, 0.1).unwrap().with_tau(tau).unwrap();
        model
            .posterior(&q.view(), &p, &PosteriorFlags::new())
            .unwrap()
            .mu[[0, 0]]
    };
    // tau shrinks toward the training mean (3.5), not toward 0.
    assert!((mu(5.0) - 3.5).abs() < (mu(0.0) - 3.5).abs());
}

/// tau acts on centered, scaled data: shifting y by c and scaling by s must shift and scale
/// mu by the same amounts and scale se by s, for every tau, on both the full and light paths.
#[test]
fn tau_posterior_is_equivariant_to_shift_and_scale_of_y() {
    let n = 24;
    let x = Array2::from_shape_fn((n, 2), |(i, j)| i as f64 * 0.1 + j as f64 * 0.03);
    let y = Array2::from_shape_fn((n, 1), |(i, _)| ((i * 7) % 11) as f64 * 0.3);
    let (c, s) = (50.0, 4.0);
    let y2 = y.mapv(|v| s * v + c);
    let m1 = crate::test_helpers::test_model(x.clone(), y, None, false);
    let m2 = crate::test_helpers::test_model(x.clone(), y2, None, false);
    let q = Array2::from_shape_fn((5, 2), |(i, j)| 0.4 * i as f64 + 3.0 + j as f64);
    let flags = PosteriorFlags::new();
    for tau in [0.0, 0.3, 5.0, 300.0] {
        let p = ENNParams::new(4, 1.0, 0.1).unwrap().with_tau(tau).unwrap();
        let a = compute_posterior_internals(&m1, &q.view(), &p, &flags).unwrap();
        let b = compute_posterior_internals(&m2, &q.view(), &p, &flags).unwrap();
        assert!((a.mu.mapv(|v| s * v + c) - &b.mu).mapv(f64::abs).iter().all(|&d| d < 1e-9));
        assert!((a.se.mapv(|v| s * v) - &b.se).mapv(f64::abs).iter().all(|&d| d < 1e-9));
        let (lm, _, _, _, _) = compute_posterior_light(&m2, &q.view(), &p, &flags).unwrap();
        assert!((lm - &b.mu).mapv(f64::abs).iter().all(|&d| d < 1e-9));
    }
}

#[test]
fn tau_large_limit_shrinks_to_training_mean() {
    let x = Array2::from_shape_fn((12, 1), |(i, _)| i as f64);
    let y = Array2::from_shape_fn((12, 1), |(i, _)| 100.0 + i as f64);
    let m = crate::test_helpers::test_model(x, y.clone(), None, false);
    let p = ENNParams::new(3, 1.0, 0.1).unwrap().with_tau(1e9).unwrap();
    let q = array![[1000.0]];
    let r = compute_posterior_internals(&m, &q.view(), &p, &PosteriorFlags::new()).unwrap();
    assert!((r.mu[[0, 0]] - y.mean().unwrap()).abs() < 1e-4);
}
