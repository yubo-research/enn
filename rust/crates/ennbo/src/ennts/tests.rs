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

fn query_from_rows(x: &Array2<f64>, end: usize) -> Array2<f64> {
    let mut q = Array2::zeros((1, 10));
    for i in 0..5 {
        let row = end - 4 + i;
        q[[0, i * 2]] = x[[row, 0]];
        q[[0, i * 2 + 1]] = x[[row, 1]];
    }
    q
}
