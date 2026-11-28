use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;

use super::O1Store;

pub(super) fn exact(x: &Array2<f64>, q: &[f64], k: usize) -> Vec<i64> {
    let mut d: Vec<(f64, i64)> = x
        .rows()
        .into_iter()
        .enumerate()
        .map(|(i, r)| (r.iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum(), i as i64))
        .collect();
    d.sort_by(|a, b| a.0.total_cmp(&b.0));
    d.into_iter().take(k).map(|p| p.1).collect()
}

fn build(n: usize, d: usize, seed: u64) -> (TempDir, O1Store, Array2<f64>) {
    std::env::set_var("ENN_O1_KIND", "lsm");
    std::env::set_var("ENN_LSM_L0", "300");
    std::env::set_var("ENN_LSM_SPAN", "3");
    let mut rng = StdRng::seed_from_u64(seed);
    let x = Array2::from_shape_fn((n, d), |_| rng.gen::<f64>());
    let y = Array2::from_shape_fn((n, 1), |(i, _)| x[[i, 0]]);
    let dir = TempDir::new().unwrap();
    let mut store = O1Store::open_forest(dir.path().to_path_buf(), d, 1).unwrap();
    for s in (0..n).step_by(250) {
        let e = (s + 250).min(n);
        store
            .append_rows(&x.slice(ndarray::s![s..e, ..]), &y.slice(ndarray::s![s..e, ..]), None)
            .unwrap();
    }
    (dir, store, x)
}

#[test]
fn lsm_with_unbounded_budget_is_exact_and_survives_reopen_and_metric_change() {
    exact_through_reopen_and_metric_change();
}

#[test]
fn lsm_raw_boxes_are_exact_through_reopen_and_metric_change() {
    std::env::set_var("ENN_LSM_BOX", "raw");
    exact_through_reopen_and_metric_change();
}

#[test]
fn lsm_axis_keys_with_raw_boxes_are_exact_through_reopen_and_metric_change() {
    std::env::set_var("ENN_LSM_PROJ", "axis");
    std::env::set_var("ENN_LSM_BOX", "raw");
    exact_through_reopen_and_metric_change();
}

fn exact_through_reopen_and_metric_change() {
    let (n, d) = (5000, 6);
    let (dir, mut store, x) = build(n, d, 11);
    assert!(dir.path().join(super::lsm::LSM_META).exists());
    store.set_probe_budget(1, usize::MAX).unwrap();
    let mut rng = StdRng::seed_from_u64(3);
    let q = Array2::from_shape_fn((20, d), |_| rng.gen::<f64>());
    let (_, idx) = store.search(&q.view(), 10, false).unwrap();
    for i in 0..q.nrows() {
        assert_eq!(idx.row(i).to_vec(), exact(&x, &q.row(i).to_vec(), 10), "query {i}");
    }
    store
        .set_metric_scale(&ndarray::Array1::from(vec![1.0, 2.0, 0.5, 1.0, 3.0, 1.0]), true)
        .unwrap();
    let more = Array2::from_shape_fn((3000, d), |_| rng.gen::<f64>());
    let y = Array2::from_shape_fn((3000, 1), |(i, _)| more[[i, 0]]);
    store.append_rows(&more.view(), &y.view(), None).unwrap();
    drop(store);
    let mut store = O1Store::open_forest(dir.path().to_path_buf(), d, 1).unwrap();
    store.set_probe_budget(1, usize::MAX).unwrap();
    let all = ndarray::concatenate![ndarray::Axis(0), x, more];
    let w = [1.0, 0.25, 4.0, 1.0, 1.0 / 9.0, 1.0];
    let (_, idx) = store.search(&q.view(), 10, false).unwrap();
    for i in 0..q.nrows() {
        let mut dd: Vec<(f64, i64)> = all
            .rows()
            .into_iter()
            .enumerate()
            .map(|(j, r)| (r.iter().zip(q.row(i)).zip(&w).map(|((a, b), c)| c * (a - b) * (a - b)).sum(), j as i64))
            .collect();
        dd.sort_by(|a, b| a.0.total_cmp(&b.0));
        let want: Vec<i64> = dd.into_iter().take(10).map(|p| p.1).collect();
        assert_eq!(idx.row(i).to_vec(), want, "query {i} after metric change");
    }
    assert!(store.occupancy().unwrap().splits > 0);
}

#[test]
fn lsm_incremental_merges_stay_exact_while_in_progress() {
    std::env::set_var("ENN_LSM_STEP", "1");
    let (n, d) = (6000, 5);
    let (_dir, mut store, x) = build(n, d, 21);
    store.set_probe_budget(1, usize::MAX).unwrap();
    let mut rng = StdRng::seed_from_u64(4);
    let q = Array2::from_shape_fn((15, d), |_| rng.gen::<f64>());
    let (_, idx) = store.search(&q.view(), 10, false).unwrap();
    for i in 0..q.nrows() {
        assert_eq!(idx.row(i).to_vec(), exact(&x, &q.row(i).to_vec(), 10), "query {i}");
    }
    assert!(store.occupancy().unwrap().splits > 0);
}

#[test]
fn lsm_incremental_merges_with_metric_changes_stay_exact() {
    std::env::set_var("ENN_LSM_STEP", "1");
    std::env::set_var("ENN_LSM_CHUNK", "50");
    let (n, d) = (6000, 4);
    let mut rng = StdRng::seed_from_u64(31);
    let (_dir, mut store, x) = build(300, d, 30);
    let mut all = x;
    for s in 0..((n - 300) / 300) {
        let k = 1.0 + (s % 3) as f64;
        store
            .set_metric_scale(&ndarray::Array1::from(vec![k, 1.0 / k, 1.0, k]), true)
            .unwrap();
        let more = Array2::from_shape_fn((300, d), |_| rng.gen::<f64>());
        let y = Array2::from_shape_fn((300, 1), |(i, _)| more[[i, 0]]);
        store.append_rows(&more.view(), &y.view(), None).unwrap();
        all = ndarray::concatenate![ndarray::Axis(0), all, more];
    }
    store
        .set_metric_scale(&ndarray::Array1::from(vec![1.0; d]), true)
        .unwrap();
    store.set_probe_budget(1, usize::MAX).unwrap();
    let q = Array2::from_shape_fn((10, d), |_| rng.gen::<f64>());
    let (_, idx) = store.search(&q.view(), 10, false).unwrap();
    for i in 0..q.nrows() {
        assert_eq!(idx.row(i).to_vec(), exact(&all, &q.row(i).to_vec(), 10), "query {i}");
    }
}
