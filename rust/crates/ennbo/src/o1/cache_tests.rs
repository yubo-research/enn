use super::forest_tests::{assert_invariants, insert_all, trajectory, D};
use super::*;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;

fn queries(n: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array2::from_shape_fn((n, D), |_| rng.gen::<f64>())
}

fn drifting_store(dir: &TempDir, cache_bytes: usize) -> O1Store {
    let mut store = O1Store::open_forest(dir.path().to_path_buf(), D, 1).unwrap();
    store.set_cache_budget(cache_bytes);
    store.index.forest_mut().params.rebuild_below = 0;
    let x = trajectory(3000, 21);
    insert_all(&mut store, &x.slice(ndarray::s![..1500, ..]).to_owned(), 50);
    store.set_repair(5, 0.25, true).unwrap();
    let scale = Array1::from_iter((0..D).map(|j| if j < D / 2 { 0.2 } else { 1.0 }));
    store.set_metric_scale(&scale, true).unwrap();
    insert_all(&mut store, &x.slice(ndarray::s![1500.., ..]).to_owned(), 25);
    store
}

#[test]
fn cached_store_answers_like_uncached_through_splits_rebuilds_and_repair() {
    let (plain_dir, cached_dir) = (TempDir::new().unwrap(), TempDir::new().unwrap());
    let plain = drifting_store(&plain_dir, 0);
    let cached = drifting_store(&cached_dir, 48 << 10);
    let q = queries(30, 4);
    assert_eq!(plain.search(&q.view(), 10, false).unwrap(), cached.search(&q.view(), 10, false).unwrap());
    let (po, co) = (assert_invariants(&plain), assert_invariants(&cached));
    assert!(co.repairs_started > 0 && co.rebuilds > 0);
    assert_eq!((po.disk_nodes, po.splits, po.repairs_finished), (co.disk_nodes, co.splits, co.repairs_finished));
    assert_eq!(po.node_writes, co.node_writes);
    assert!(co.node_reads < po.node_reads);
    let stats = cached.cache_stats().unwrap();
    assert!(stats.node_hits > 0 && stats.row_hits > 0 && stats.node_evictions > 0);
    assert!(stats.node_bytes + stats.row_bytes <= stats.budget_bytes);
    assert!(plain.cache_stats().is_none());
    drop(cached);
    let reopened = O1Store::open_forest(cached_dir.path().to_path_buf(), D, 1).unwrap();
    assert_eq!(plain.search(&q.view(), 10, false).unwrap(), reopened.search(&q.view(), 10, false).unwrap());
}

#[test]
fn repeated_queries_hit_the_cache_and_rows_y_come_from_it() {
    let dir = TempDir::new().unwrap();
    let mut store = O1Store::open_forest(dir.path().to_path_buf(), D, 1).unwrap();
    insert_all(&mut store, &queries(4000, 8), 400);
    store.set_io_width(1, 0).unwrap();
    store.set_cache_budget(8 << 20);
    let q = queries(1, 9);
    let first = store.search(&q.view(), 10, false).unwrap();
    let cold = store.cache_stats().unwrap();
    assert_eq!(cold.node_hits, 0);
    assert!(cold.row_misses > cold.row_hits);
    let reads = store.node_io().unwrap().0;
    assert_eq!(first, store.search(&q.view(), 10, false).unwrap());
    let warm = store.cache_stats().unwrap();
    assert_eq!((warm.node_misses, warm.row_misses), (cold.node_misses, cold.row_misses));
    assert_eq!(store.node_io().unwrap().0, reads);
    let nn = first.1[[0, 0]] as usize;
    assert_eq!(store.row_y(nn).unwrap(), Array1::from(vec![store.row_x(nn).unwrap()[0]]));
    assert_eq!(store.cache_stats().unwrap().row_misses, warm.row_misses + 1);
    store.set_cache_budget(0);
    assert!(store.cache_stats().is_none());
    assert_eq!(first, store.search(&q.view(), 10, false).unwrap());
}

#[test]
fn profile_counts_each_node_read_once() {
    let dir = TempDir::new().unwrap();
    let mut store = O1Store::open_forest(dir.path().to_path_buf(), D, 1).unwrap();
    assert_eq!(store.cache_stats().unwrap().node_budget_bytes, DEFAULT_NODE_CACHE_BYTES);
    store.set_cache_budget(0);
    insert_all(&mut store, &trajectory(3000, 2), 300);
    let q = queries(7, 3);
    store.search(&q.view(), 10, false).unwrap();
    assert_eq!(store.take_profile(), ProfileStats::default());
    store.set_profile(true);
    let reads = store.node_io().unwrap().0;
    store.search(&q.view(), 10, false).unwrap();
    let p = store.take_profile();
    assert_eq!(p.queries, 7);
    assert!(p.split_visits > 0 && p.leaf_visits > 0 && p.candidates > 0);
    assert!(p.leaf_pages >= p.leaf_visits);
    assert_eq!(store.node_io().unwrap().0 - reads, p.split_visits + p.leaf_pages);
    assert!(p.search_ns >= p.hash_ns + p.route_ns + p.leaf_ns + p.row_ns + p.distance_ns);
}

#[test]
fn split_cache_answers_like_uncached_and_keeps_classes_within_their_budgets() {
    let (plain_dir, split_dir) = (TempDir::new().unwrap(), TempDir::new().unwrap());
    let plain = drifting_store(&plain_dir, 0);
    let mut split = drifting_store(&split_dir, 0);
    split.set_cache_split(64 << 10, 16 << 10);
    let q = queries(30, 5);
    let want = plain.search(&q.view(), 10, false).unwrap();
    assert_eq!(want, split.search(&q.view(), 10, false).unwrap());
    assert_eq!(want, split.search(&q.view(), 10, false).unwrap());
    let s = split.cache_stats().unwrap();
    assert_eq!((s.budget_bytes, s.node_budget_bytes), (80 << 10, 64 << 10));
    assert!(s.node_bytes <= 64 << 10 && s.row_bytes <= 16 << 10);
    assert!(s.node_hits > 0 && s.row_evictions > 0);
}
