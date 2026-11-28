use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;

use super::lsm_tests::exact;
use super::O1Store;

fn graph_store(d: usize, buf_bytes: usize) -> (TempDir, O1Store) {
    std::env::set_var("ENN_O1_KIND", "graph");
    std::env::set_var("ENN_G_BUF", buf_bytes.to_string());
    std::env::set_var("ENN_G_ROWS", "20");
    std::env::set_var("ENN_G_FAN", "6");
    std::env::set_var("ENN_G_M", "4");
    let dir = TempDir::new().unwrap();
    let store = O1Store::open_forest(dir.path().to_path_buf(), d, 1).unwrap();
    (dir, store)
}

fn append(store: &mut O1Store, x: &Array2<f64>, step: usize) {
    for s in (0..x.nrows()).step_by(step) {
        let e = (s + step).min(x.nrows());
        let y = Array2::from_shape_fn((e - s, 1), |(i, _)| x[[s + i, 0]]);
        store.append_rows(&x.slice(ndarray::s![s..e, ..]), &y.view(), None).unwrap();
    }
}

#[test]
fn graph_with_unbounded_budget_is_exact_after_reopen() {
    exact_after_reopen(2000, 100);
}

#[test]
fn graph_rows_held_only_in_the_buffer_are_found_and_survive_reopen() {
    exact_after_reopen(1 << 20, 0);
}

fn exact_after_reopen(buf_bytes: usize, min_splits: u64) {
    let d = 5;
    let (dir, mut store) = graph_store(d, buf_bytes);
    let mut rng = StdRng::seed_from_u64(7);
    let x = Array2::from_shape_fn((4000, d), |_| rng.gen::<f64>());
    append(&mut store, &x, 333);
    assert!(dir.path().join(super::graph::GRAPH_META).exists());
    drop(store);
    std::env::remove_var("ENN_O1_KIND");
    std::env::set_var("ENN_G_WIDTH", "4");
    let mut store = O1Store::open_forest(dir.path().to_path_buf(), d, 1).unwrap();
    store.set_probe_budget(1, usize::MAX).unwrap();
    let q = Array2::from_shape_fn((15, d), |_| rng.gen::<f64>());
    let (_, idx) = store.search(&q.view(), 10, false).unwrap();
    for i in 0..q.nrows() {
        assert_eq!(idx.row(i).to_vec(), exact(&x, &q.row(i).to_vec(), 10), "query {i}");
    }
    let occ = store.occupancy().unwrap();
    assert!(occ.splits >= min_splits, "splits {}", occ.splits);
    if min_splits > 0 {
        assert!(occ.mean_depth >= 4.0, "height {}", occ.mean_depth);
    }
}

#[test]
fn graph_small_budget_finds_most_neighbours_of_a_trajectory() {
    let d = 4;
    let (_dir, mut store) = graph_store(d, 0);
    let mut rng = StdRng::seed_from_u64(9);
    let mut x = Array2::zeros((3000, d));
    for i in 1..3000 {
        for k in 0..d {
            x[[i, k]] = x[[i - 1, k]] + rng.gen::<f64>() - 0.5;
        }
    }
    append(&mut store, &x, 100);
    store.set_probe_budget(1, 400).unwrap();
    let (mut hit, mut tot) = (0, 0);
    for j in (0..3000).step_by(150) {
        let q: Vec<f64> = (0..d).map(|k| x[[j, k]] + 0.1).collect();
        let qa = Array2::from_shape_vec((1, d), q.clone()).unwrap();
        let (_, idx) = store.search(&qa.view(), 10, false).unwrap();
        let want = exact(&x, &q, 10);
        hit += idx.row(0).iter().filter(|i| want.contains(i)).count();
        tot += 10;
    }
    assert!(hit as f64 >= 0.8 * tot as f64, "recall {hit}/{tot}");
}

#[test]
fn graph_box_pruning_stays_exact_after_a_metric_change() {
    let d = 3;
    let (_dir, mut store) = graph_store(d, 1500);
    let mut rng = StdRng::seed_from_u64(5);
    let x = Array2::from_shape_fn((3000, d), |_| rng.gen::<f64>());
    append(&mut store, &x, 500);
    store.set_metric_scale(&ndarray::Array1::from(vec![0.2, 1.0, 5.0]), true).unwrap();
    store.set_probe_budget(1, usize::MAX).unwrap();
    let w = [25.0, 1.0, 0.04];
    let q = Array2::from_shape_fn((10, d), |_| rng.gen::<f64>());
    let (_, idx) = store.search(&q.view(), 5, false).unwrap();
    for i in 0..q.nrows() {
        let mut dd: Vec<(f64, i64)> = (0..x.nrows())
            .map(|j| ((0..d).map(|k| w[k] * (x[[j, k]] - q[[i, k]]).powi(2)).sum(), j as i64))
            .collect();
        dd.sort_by(|a, b| a.0.total_cmp(&b.0));
        let want: Vec<i64> = dd.into_iter().take(5).map(|p| p.1).collect();
        assert_eq!(idx.row(i).to_vec(), want, "query {i}");
    }
}
