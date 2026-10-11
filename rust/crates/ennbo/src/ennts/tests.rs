use ndarray::Array2;

use super::{EnnTs, EnnTsConfig, QueryOptions};

fn wide_config() -> EnnTsConfig {
    EnnTsConfig {
        bucket_width: 1.0e9,
        k: 2,
        num_tables: 3,
        num_projections: 2,
        probe: 0,
        aleatoric_variance_scale: 0.0,
        ..EnnTsConfig::new(5)
    }
}

fn column_series(n: usize) -> (Vec<i64>, Array2<f64>, Array2<f64>) {
    let ts: Vec<i64> = (0..n as i64).collect();
    let x = Array2::from_shape_fn((n, 2), |(i, j)| if j == 0 { i as f64 } else { 0.0 });
    let y = Array2::from_shape_fn((n, 1), |(i, _)| i as f64);
    (ts, x, y)
}

fn first_window_query() -> Array2<f64> {
    let mut q = Array2::zeros((1, 10));
    for i in 0..5 {
        q[[0, i * 2]] = i as f64;
    }
    q
}

fn options(cutoff: i64, k: usize, require_target: bool) -> QueryOptions {
    QueryOptions {
        cutoff,
        k,
        require_target,
    }
}

#[test]
fn rejects_zero_lookback() {
    let cfg = EnnTsConfig {
        lookback: 0,
        ..EnnTsConfig::new(5)
    };
    assert!(EnnTs::new(cfg).is_err());
}

#[test]
fn lookback_is_fixed_for_the_model() {
    let mut short = EnnTs::new(EnnTsConfig::new(3)).unwrap();
    let mut long = EnnTs::new(EnnTsConfig::new(7)).unwrap();
    let x = Array2::zeros((4, 1));
    let y = Array2::zeros((4, 1));
    let ts = [0, 1, 2, 3];
    short.add(0, &ts, x.view(), Some(y.view())).unwrap();
    long.add(1, &ts, x.view(), Some(y.view())).unwrap();
    assert_eq!(short.lookback(), 3);
    assert_eq!(long.lookback(), 7);
    assert_eq!(short.num_windows(), 2);
    assert_eq!(long.num_windows(), 0);
    short.set_variance_scales(2.0, 0.2).unwrap();
    assert_eq!(short.lookback(), 3);
}

#[test]
fn rejects_time_reversal() {
    let mut model = EnnTs::new(wide_config()).unwrap();
    let x = Array2::zeros((2, 2));
    let y = Array2::zeros((2, 1));
    let err = model.add(0, &[1, 1], x.view(), Some(y.view()));
    assert!(err.is_err());
}

#[test]
fn gap_skips_incomplete_window() {
    let mut model = EnnTs::new(wide_config()).unwrap();
    let x = Array2::zeros((6, 2));
    let y = Array2::zeros((6, 1));
    model
        .add(0, &[0, 1, 2, 3, 4, 6], x.view(), Some(y.view()))
        .unwrap();
    assert_eq!(model.num_windows(), 1);
    assert_eq!(model.num_obs(), 6);
}

#[test]
fn separate_series_do_not_form_a_window() {
    let mut model = EnnTs::new(wide_config()).unwrap();
    let x = Array2::zeros((3, 2));
    let y = Array2::zeros((3, 1));
    model.add(0, &[0, 1, 2], x.view(), Some(y.view())).unwrap();
    model.add(1, &[0, 1, 2], x.view(), Some(y.view())).unwrap();
    assert_eq!(model.num_windows(), 0);
}

#[test]
fn posterior_mean_matches_available_target() {
    let mut model = EnnTs::new(wide_config()).unwrap();
    let (ts, x, y) = column_series(6);
    model.add(0, &ts, x.view(), Some(y.view())).unwrap();
    let q = first_window_query();
    let post = model.posterior(q.view(), 5, Some(1), false).unwrap();
    assert!((post.mu[[0, 0]] - 5.0).abs() < 1e-8);
}

#[test]
fn cutoff_excludes_a_target_that_is_not_yet_available() {
    let mut model = EnnTs::new(wide_config()).unwrap();
    let (ts, x, y) = column_series(8);
    model.add(0, &ts, x.view(), Some(y.view())).unwrap();
    let q = first_window_query();
    let early = model.search(q.view(), &options(4, 2, true)).unwrap();
    assert_eq!(early.series_id[[0, 0]], -1);
    let ready = model.search(q.view(), &options(5, 2, true)).unwrap();
    assert_eq!(ready.end_timestamp[[0, 0]], 4);
    assert!(ready.distance[[0, 0]] < 1e-8);
}

#[test]
fn add_y_makes_a_window_eligible() {
    let mut model = EnnTs::new(wide_config()).unwrap();
    let (ts, x, _) = column_series(6);
    model.add(0, &ts, x.view(), None).unwrap();
    let q = first_window_query();
    let hidden = model.search(q.view(), &options(5, 1, true)).unwrap();
    assert_eq!(hidden.series_id[[0, 0]], -1);
    model
        .add_y(0, &[5], Array2::from_elem((1, 1), 5.0).view())
        .unwrap();
    let shown = model.search(q.view(), &options(5, 1, true)).unwrap();
    assert_eq!(shown.end_timestamp[[0, 0]], 4);
    assert!((shown.target[[0, 0, 0]] - 5.0).abs() < 1e-8);
}

#[test]
fn wide_buckets_match_exact_search() {
    let mut model = EnnTs::new(wide_config()).unwrap();
    let n = 12;
    let ts: Vec<i64> = (0..n as i64).collect();
    let x = Array2::from_shape_fn((n, 2), |(i, j)| ((i * 3 + j) % 5) as f64);
    let y = Array2::from_shape_fn((n, 1), |(i, _)| i as f64 * 0.1);
    model.add(0, &ts, x.view(), Some(y.view())).unwrap();
    let q = query_from_rows(&x, 6);
    let opts = options(11, 2, true);
    let approx = model.search(q.view(), &opts).unwrap();
    let exact = model.search_exact(q.view(), &opts).unwrap();
    assert_eq!(approx.end_timestamp, exact.end_timestamp);
    assert_eq!(approx.series_id, exact.series_id);
}

fn slot_count(model: &EnnTs) -> usize {
    model
        .buckets
        .iter()
        .map(|table| table.values().map(Vec::len).sum::<usize>())
        .sum()
}

#[test]
fn metric_morph_keeps_one_slot_per_table_and_the_new_distance() {
    let mut model = EnnTs::new(wide_config()).unwrap();
    let (ts, x, y) = column_series(12);
    model.add(0, &ts, x.view(), Some(y.view())).unwrap();
    let before = model.num_windows();
    assert_eq!(slot_count(&model), model.config.num_tables * before);
    assert_eq!(model.metric_morph_remaining(), 0);
    let width = model.lookback() * model.num_dim().unwrap();
    let mut scale = vec![1.0; width];
    scale[0] = 0.25;
    model.adopt_scale_for_test(scale.clone()).unwrap();
    assert_eq!(model.metric_morph_remaining(), before);
    assert_eq!(slot_count(&model), model.config.num_tables * before);
    let extra_x = Array2::from_elem((1, 2), 11.0);
    let extra_y = Array2::from_elem((1, 1), 11.0);
    model
        .add(0, &[12], extra_x.view(), Some(extra_y.view()))
        .unwrap();
    let moved = 6.min(before);
    assert_eq!(model.metric_morph_remaining(), before - moved);
    assert_eq!(
        slot_count(&model),
        model.config.num_tables * model.num_windows()
    );
    let mut q = first_window_query();
    q[[0, 0]] = 1.0;
    let opts = options(11, 1, true);
    let approx = model.search(q.view(), &opts).unwrap();
    let exact = model.search_exact(q.view(), &opts).unwrap();
    assert_eq!(approx.end_timestamp[[0, 0]], exact.end_timestamp[[0, 0]]);
    let stored = window_row(&x, exact.end_timestamp[[0, 0]] as usize);
    let mut expect = 0.0;
    for j in 0..width {
        let delta = (q[[0, j]] - stored[j]) / scale[j];
        expect += delta * delta;
    }
    assert!(expect > 0.0);
    assert!((exact.distance[[0, 0]] - expect.sqrt()).abs() < 1e-8);
}

#[test]
fn auto_metric_refits_once_a_hundred_windows_are_labeled() {
    let mut cfg = wide_config();
    cfg.metric_learning = crate::metric_auto::MetricLearning::Auto;
    cfg.lookback = 2;
    let mut model = EnnTs::new(cfg).unwrap();
    let n = 110;
    let ts: Vec<i64> = (0..n as i64).collect();
    let x = Array2::from_shape_fn((n, 2), |(i, j)| {
        if j == 0 {
            0.05 * i as f64
        } else {
            ((i * 17) % 13) as f64
        }
    });
    let y = Array2::from_shape_fn((n, 1), |(i, _)| 0.05 * (i as f64 + 1.0));
    model.add(0, &ts, x.view(), Some(y.view())).unwrap();
    let weights = model.metric_weights().expect("learner");
    assert_eq!(weights.len(), 4);
    assert!(weights.iter().all(|w| w.is_finite() && *w > 0.0));
    assert!(model.heldout_gain().is_some());
    assert!(
        weights[0] + weights[2] > weights[1] + weights[3],
        "feature 0 should outweigh the noise feature, weights={weights:?}, gain={:?}",
        model.heldout_gain()
    );
    assert_eq!(
        slot_count(&model),
        model.config.num_tables * model.num_windows()
    );
}

fn window_row(x: &Array2<f64>, end: usize) -> Vec<f64> {
    let mut out = Vec::new();
    for row in (end + 1 - 5)..=end {
        out.push(x[[row, 0]]);
        out.push(x[[row, 1]]);
    }
    out
}

fn query_from_rows(x: &Array2<f64>, end: usize) -> Array2<f64> {
    let mut q = Array2::zeros((1, 10));
    for i in 0..5 {
        let row = end - 4 + i;
        q[[0, i * 2]] = x[[row, 0]];
        q[[0, i * 2 + 1]] = x[[row, 1]];
    }
    q
}
