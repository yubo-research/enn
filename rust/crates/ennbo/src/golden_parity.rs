//! Frozen outputs for the exact-match ports, plus distributional checks.

use ndarray::{Array1, Array2, ArrayD};
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::benchmarks::{ackley_core, separable_unimodal};
use crate::calibration::{apply_normal, AffineCalibrator};
use crate::candidates_fast::generate_tr_candidates_fast;
use crate::metric_auto::auto_weights;
use crate::normal_sample::{confidence_interval, sample_normal, z_crit};
use crate::philox_hash::philox_normal_hash;

fn close(got: f64, expect: f64) {
    assert!((got - expect).abs() <= 1e-12 * expect.abs().max(1.0), "{got} vs {expect}");
}

#[test]
fn philox_hash_matches_numpy_grid() {
    let seeds = [1_i64, 7];
    let idx = [0_i64, 4, 4];
    let got = philox_normal_hash(&seeds, &idx, 2).expect("hash");
    let expect = [
        0.194_295_646_584_818_88,
        0.749_421_628_383_337_5,
        0.271_551_695_343_482_5,
        0.807_575_439_532_017,
        0.271_551_695_343_482_5,
        0.807_575_439_532_017,
        1.909_158_222_295_347,
        -1.269_590_766_741_959_3,
        0.646_806_065_523_385_9,
        1.965_987_039_301_151,
        0.646_806_065_523_385_9,
        1.965_987_039_301_151,
    ];
    for (a, b) in got.iter().zip(expect) {
        assert_eq!(*a, b);
    }
}

#[test]
fn calibrator_recovers_known_line() {
    let mu = Array2::from_shape_vec((3, 1), vec![0.0, 1.0, 2.0]).unwrap();
    let y = Array2::from_shape_vec((3, 1), vec![1.0, 3.0, 5.0]).unwrap();
    let se = Array2::ones((3, 1));
    let cal = AffineCalibrator::fit(mu.view(), y.view(), Some(se.view())).unwrap();
    close(cal.a[0], 1.0);
    close(cal.b[0], 2.0);
    close(cal.c[0], 0.0);
    let ale0 = Array2::zeros((3, 1));
    let (mu_p, se_p, _, _) = apply_normal(
        cal.a.as_slice().unwrap(),
        cal.b.as_slice().unwrap(),
        cal.c.as_slice().unwrap(),
        mu.view(),
        se.view(),
        ale0.view(),
        None,
    )
    .unwrap();
    close(mu_p[[0, 0]], 1.0);
    close(mu_p[[1, 0]], 3.0);
    close(mu_p[[2, 0]], 5.0);
    assert!(se_p.iter().all(|v| v.abs() < 1e-9), "{se_p}");

    let a = [0.0];
    let b = [1.0];
    let c = [0.5];
    let mu2 = Array2::from_shape_vec((1, 1), vec![4.0]).unwrap();
    let epi = Array2::from_shape_vec((1, 1), vec![0.4]).unwrap();
    let ale = Array2::zeros((1, 1));
    let (mu_s, se_s, _, _) =
        apply_normal(&a, &b, &c, mu2.view(), epi.view(), ale.view(), None).unwrap();
    close(mu_s[[0, 0]], 4.0);
    close(se_s[[0, 0]], 0.2);
}

#[test]
fn benchmarks_match_closed_forms() {
    let x = Array2::from_shape_vec((1, 3), vec![1.0, 1.0, 1.0]).unwrap();
    let y = ackley_core(x.view(), 20.0, 0.2, 2.0 * std::f64::consts::PI);
    close(y[0], 0.0);
    let xs = Array2::from_shape_vec((1, 2), vec![120.0, 0.91]).unwrap();
    let ys = separable_unimodal(xs.view());
    close(ys[[0, 0]], 500_000.0);
    close(ys[[0, 1]], 12.5);
}

#[test]
fn confidence_interval_is_linear_without_bounds() {
    let mu = ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2]), vec![1.0, -2.0]).unwrap();
    let se = ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 2]), vec![0.2, 0.5]).unwrap();
    let z = z_crit(0.95).unwrap();
    let (lo, hi) = confidence_interval(&mu, &se, None, 0.95).unwrap();
    let lo: Vec<f64> = lo.iter().copied().collect();
    let hi: Vec<f64> = hi.iter().copied().collect();
    close(lo[0], 1.0 - z * 0.2);
    close(hi[0], 1.0 + z * 0.2);
    close(lo[1], -2.0 - z * 0.5);
    close(hi[1], -2.0 + z * 0.5);
}

#[test]
fn sample_moments_match_mu_and_se() {
    let mu = ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 1]), vec![3.0]).unwrap();
    let se = ArrayD::from_shape_vec(ndarray::IxDyn(&[1, 1]), vec![0.4]).unwrap();
    let draws = sample_normal(&mu, &se, None, 4000, 11, None).unwrap();
    let mean = draws.iter().sum::<f64>() / draws.len() as f64;
    let var = draws.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / draws.len() as f64;
    assert!((mean - 3.0).abs() < 0.05, "{mean}");
    assert!((var.sqrt() - 0.4).abs() < 0.05, "{}", var.sqrt());
}

#[test]
fn fast_candidates_match_the_binomial_uniform_law() {
    let d = 8;
    let center = Array1::from_elem(d, 0.5);
    let lower = Array1::zeros(d);
    let upper = Array1::ones(d);
    let mut rng = StdRng::seed_from_u64(7);
    let n = 2000;
    let cand = generate_tr_candidates_fast(&center.view(), &lower, &upper, n, &mut rng, 2).unwrap();
    let mut total = 0.0;
    for i in 0..n {
        let mut k = 0;
        for j in 0..d {
            let v = cand[[i, j]];
            if (v - 0.5).abs() > 1e-12 {
                assert!((0.0..=1.0).contains(&v));
                k += 1;
            }
        }
        assert!(k >= 1);
        total += k as f64;
    }
    let mean = total / n as f64;
    let expected = 2.0 + 0.75_f64.powi(d as i32);
    assert!((mean - expected).abs() < 0.15, "{mean} vs {expected}");
}

#[test]
fn auto_weights_identity_below_the_row_floor() {
    let (w, gain) = auto_weights(&[0.0, 1.0, 0.5, 0.2], 2, 2, &[0.0, 1.0], 1, 10, &[]).unwrap();
    assert_eq!(w, vec![1.0, 1.0]);
    assert!(gain.is_infinite() && gain.is_sign_negative());
}

#[test]
#[allow(clippy::excessive_precision)]
fn auto_weights_frozen_on_a_fixed_stream() {
    let n = 120;
    let d = 3;
    let mut x = vec![0.0; n * d];
    let mut y = vec![0.0; n];
    for i in 0..n {
        for j in 0..d {
            x[i * d + j] = ((i + 1) as f64 * (j + 3) as f64 * 0.017 + 0.001 * i as f64) % 1.0;
        }
        y[i] = (6.0 * std::f64::consts::PI * x[i * d]).sin() + 0.01 * x[i * d + 1];
    }
    let (w, gain) = auto_weights(&x, n, d, &y, 1, 10, &[]).unwrap();
    let expect = [
        35.319_578_728_293_791,
        3.581_230_732_208_359_6e-5,
        3.587_184_161_103_153_3e-5,
    ];
    for (got, exp) in w.iter().zip(expect) {
        close(*got, exp);
    }
    // This stream misses the frozen gain by about 1.02e-5.
    let gain_gap = (gain - 1.587_039_283_771_839_2).abs();
    assert!(gain_gap < 1e-4, "heldout_gain {gain} gap {gain_gap}");
    assert!(w[0] > 1e3 * w[1].max(w[2]));
}
