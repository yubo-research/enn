use super::forest_tree::depth_limit;
use super::*;
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;

pub(super) const D: usize = 6;

pub(super) fn trajectory(n: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    let x0: Vec<f64> = (0..D).map(|_| rng.gen::<f64>()).collect();
    Array2::from_shape_fn((n, D), |(i, j)| {
        let r = 1.0 / (1.0 + i as f64 / 10.0);
        0.5 + (x0[j] - 0.5) * r * (1.0 + 0.1 * (rng.gen::<f64>() - 0.5))
    })
}

pub(super) fn insert_all(store: &mut O1Store, x: &Array2<f64>, batch: usize) {
    for lo in (0..x.nrows()).step_by(batch) {
        let hi = (lo + batch).min(x.nrows());
        let xs = x.slice(ndarray::s![lo..hi, ..]);
        let ys = Array2::from_shape_fn((hi - lo, 1), |(i, _)| xs[[i, 0]]);
        store.append_rows(&xs, &ys.view(), None).unwrap();
    }
}

fn forest_store(dir: &TempDir, x: &Array2<f64>, batch: usize) -> O1Store {
    let mut store = O1Store::open_forest(dir.path().to_path_buf(), D, 1).unwrap();
    insert_all(&mut store, x, batch);
    store
}

pub(super) fn assert_invariants(store: &O1Store) -> O1Occupancy {
    let occ = store.occupancy().unwrap();
    let p = ForestParams::default();
    assert_eq!(occ.reachable_rows, store.len());
    assert!(occ.table_rows.iter().all(|&r| r == store.len()));
    assert!(occ.max_arrivals as usize <= p.leaf_cap);
    occ
}

fn exact_top(store: &O1Store, q: &[f64], k: usize) -> Vec<i64> {
    let mut all: Vec<(f64, i64)> = (0..store.len())
        .map(|i| (store.row_x(i).unwrap().iter().zip(q).map(|(a, b)| (a - b) * (a - b)).sum(), i as i64))
        .collect();
    all.sort_by(|a, b| a.0.total_cmp(&b.0));
    all.into_iter().take(k).map(|p| p.1).collect()
}

fn recall(store: &O1Store, queries: &Array2<f64>, k: usize) -> f64 {
    let (_, idx) = store.search(&queries.view(), k, false).unwrap();
    let mut hit = 0;
    for (qi, q) in queries.rows().into_iter().enumerate() {
        let truth = exact_top(store, &q.to_vec(), k);
        hit += idx.row(qi).iter().filter(|i| truth.contains(i)).count();
    }
    hit as f64 / (k * queries.nrows()) as f64
}

#[test]
fn trajectory_rows_stay_reachable_with_bounded_leaves_and_depth() {
    let n = 4000;
    let p = ForestParams::default();
    let dir = TempDir::new().unwrap();
    let mut store = O1Store::open_forest(dir.path().to_path_buf(), D, 1).unwrap();
    insert_all(&mut store, &trajectory(n, 3), 1);
    let occ = assert_invariants(&store);
    assert!(occ.hash_bits <= depth_limit(n, p.leaf_cap, p.alpha) + 1);
    assert!(occ.rebuilds > 0);
    assert!(occ.splits > 0);
    assert_eq!(occ.overflow_keys.iter().sum::<usize>(), 0);
    assert!(occ.free_nodes < occ.disk_nodes);
    let min = forest_pack::REPACK_MIN_NODES;
    for tree in &store.index.forest().trees {
        assert!(tree.packed_at > 0 && tree.nodes.num_nodes() < 2 * tree.packed_at.max(min));
    }
    let dir = TempDir::new().unwrap();
    let mut store = O1Store::open_forest(dir.path().to_path_buf(), D, 1).unwrap();
    store.set_par_rebuild(100).unwrap();
    insert_all(&mut store, &trajectory(n, 3), 1);
    let planned = assert_invariants(&store);
    assert!(planned.hash_bits <= depth_limit(n, p.leaf_cap, p.alpha) + 1 && planned.rebuilds > 0);
    assert!(planned.max_rebuild_rows >= 100 && planned.free_nodes < planned.disk_nodes);
    let routed = |s: &O1Store| s.index.forest().maint_stats().route_rows;
    let dir = TempDir::new().unwrap();
    let mut base = O1Store::open_forest(dir.path().to_path_buf(), D, 1).unwrap();
    base.set_split_rule(forest_rules::SplitRule::Pivot).unwrap();
    insert_all(&mut base, &trajectory(n / 3, 3), 1);
    let m = base.index.forest().maint_stats();
    assert!(routed(&store) == 0 && routed(&base) > 0 && m.gathered >= m.rebuild_gathered);
}

#[test]
fn duplicates_chain_pages_without_exceeding_capacity() {
    let dir = TempDir::new().unwrap();
    let mut x = trajectory(300, 5);
    for i in 0..200 {
        x.row_mut(i + 50).fill(0.25);
    }
    let store = forest_store(&dir, &x, 7);
    let occ = assert_invariants(&store);
    assert!(occ.overflow_keys.iter().all(|&c| c >= 1));
    let q = ndarray::arr2(&[[0.25; D]]);
    let (dist, _) = store.search(&q.view(), 10, false).unwrap();
    assert!(dist.iter().all(|&d| d == 0.0));
}

#[test]
fn reopen_restores_the_same_candidates_and_survives_rehash() {
    let dir = TempDir::new().unwrap();
    let x = trajectory(2000, 9);
    let mut rng = StdRng::seed_from_u64(1);
    let queries: Vec<Vec<f64>> = (0..5).map(|_| (0..D).map(|_| rng.gen::<f64>()).collect()).collect();
    let before: Vec<Vec<u32>> = {
        let store = forest_store(&dir, &x, 100);
        queries.iter().map(|q| store.index.candidate_ids(q, &Default::default(), &store.x).unwrap()).collect()
    };
    let mut reopened = O1Store::open_forest(dir.path().to_path_buf(), D, 1).unwrap();
    let after: Vec<Vec<u32>> = queries.iter().map(|q| reopened.index.candidate_ids(q, &Default::default(), &reopened.x).unwrap()).collect();
    assert_eq!(before, after);
    assert_eq!(reopened.index.forest().node_io().1, 0);
    let scale = ndarray::Array1::from_iter((0..D).map(|j| 1.0 + j as f64));
    let io_before = reopened.node_io().unwrap();
    reopened.set_repair(0, 0.25, true).unwrap();
    reopened.set_metric_scale(&scale, true).unwrap();
    let io_after = reopened.node_io().unwrap();
    assert!(io_after.0 >= io_before.0 && io_after.1 > io_before.1);
    assert_eq!(reopened.occupancy().unwrap().global_rebuilds, 1);
    assert_eq!(reopened.metric_scale().unwrap(), scale);
    assert_eq!(reopened.index.hash_scale(), scale.as_slice().unwrap());
    assert_invariants(&reopened);
    reopened.set_trees(3).unwrap();
    assert!(reopened.set_trees(0).is_err());
    let three: Vec<Vec<u32>> = queries.iter().map(|q| reopened.index.candidate_ids(q, &Default::default(), &reopened.x).unwrap()).collect();
    drop(reopened);
    let kept = O1Store::open_forest(dir.path().to_path_buf(), D, 1).unwrap();
    assert_eq!(kept.index.forest().trees.len(), 3);
    assert_eq!(three, queries.iter().map(|q| kept.index.candidate_ids(q, &Default::default(), &kept.x).unwrap()).collect::<Vec<_>>());
    assert_eq!(kept.occupancy().unwrap().table_rows.len(), 3);
    drop(kept);
    std::fs::remove_file(dir.path().join(forest::FOREST_STATE_FILE)).unwrap();
    let rebuilt = O1Store::open_forest(dir.path().to_path_buf(), D, 1).unwrap();
    assert_invariants(&rebuilt);
}

#[test]
fn uniform_recall_is_high_and_budget_bounds_candidates() {
    let dir = TempDir::new().unwrap();
    let mut rng = StdRng::seed_from_u64(11);
    let x = Array2::from_shape_fn((5000, D), |_| rng.gen::<f64>());
    let mut store = forest_store(&dir, &x, 500);
    assert_invariants(&store);
    let q = Array2::from_shape_fn((20, D), |_| rng.gen::<f64>());
    assert!(recall(&store, &q, 10) > 0.9);
    store.set_probe_budget(2, 64).unwrap();
    let narrow = recall(&store, &q, 10);
    for row in q.rows() {
        assert!(store.index.candidate_ids(&row.to_vec(), &Default::default(), &store.x).unwrap().len() <= 64);
    }
    assert!(narrow < 0.9);
    store.set_probe_budget(40, 200).unwrap();
    for row in q.rows() {
        let ids = store.index.candidate_ids(&row.to_vec(), &Default::default(), &store.x).unwrap();
        assert!(ids.len() <= 200 && ids.windows(2).all(|w| w[0] < w[1]));
    }
    assert!(recall(&store, &q, 10) > narrow);
    store.set_probe_budget(1000, 32000).unwrap();
    store.set_bound(1.0).unwrap();
    let bounded = recall(&store, &q, 10);
    store.set_bound(0.01).unwrap();
    assert!(bounded > 0.95 && recall(&store, &q, 10) < bounded);
    let lsh_dir = TempDir::new().unwrap();
    let mut lsh = O1Store::open_or_create(lsh_dir.path().to_path_buf(), D, 1).unwrap();
    assert!(lsh.set_probe_budget(2, 64).is_err());
    assert!(lsh.set_bound(1.0).is_err());
}

#[test]
fn tree_ordered_rows_keep_answers_and_survive_reopen() {
    let x = trajectory(6000, 11);
    let mut rng = StdRng::seed_from_u64(12);
    let q = Array2::from_shape_fn((20, D), |_| rng.gen::<f64>());
    let (plain_dir, dir) = (TempDir::new().unwrap(), TempDir::new().unwrap());
    let mut plain = O1Store::open_forest(plain_dir.path().to_path_buf(), D, 1).unwrap();
    plain.set_row_order(OrderParams { rows_per_insert: 0, ..OrderParams::default() }).unwrap();
    insert_all(&mut plain, &x, 250);
    assert_eq!(plain.maint_stats().unwrap().ordered_rows, 0);
    let mut store = O1Store::open_forest(dir.path().to_path_buf(), D, 1).unwrap();
    assert!(store.set_row_order(OrderParams { rows_per_insert: 2, grow: 1.0, min_rows: 1000 }).is_err());
    store.set_row_order(OrderParams { rows_per_insert: 2, grow: 1.3, min_rows: 1000 }).unwrap();
    insert_all(&mut store, &x, 250);
    let m = store.maint_stats().unwrap();
    assert!(m.order_builds >= 2 && m.ordered_rows >= 3000 && m.ordered_rows < 6000, "{m:?}");
    assert_eq!(store.search(&q.view(), 10, false).unwrap(), plain.search(&q.view(), 10, false).unwrap());
    store.set_io_width(4, 0).unwrap();
    for i in 0..q.nrows() {
        let one = q.slice(ndarray::s![i..i + 1, ..]);
        assert_eq!(store.search(&one, 10, false).unwrap(), plain.search(&one, 10, false).unwrap());
    }
    assert_eq!(store.order_rows().unwrap(), 6000);
    assert_eq!(store.row_x(4321).unwrap(), plain.row_x(4321).unwrap());
    assert_invariants(&store);
    drop(store);
    std::fs::write(dir.path().join("o1f_xo_copy_99.bin"), b"stale").unwrap();
    std::fs::write(dir.path().join("o1f_tree_999.build"), b"stale").unwrap();
    let mut reopened = O1Store::open_forest(dir.path().to_path_buf(), D, 1).unwrap();
    assert_eq!(reopened.maint_stats().unwrap().ordered_rows, 6000);
    assert!(!dir.path().join("o1f_xo_copy_99.bin").exists() && !dir.path().join("o1f_tree_999.build").exists());
    let exact = plain.search(&q.view(), 10, false).unwrap();
    assert_eq!(reopened.search(&q.view(), 10, false).unwrap(), exact);
    reopened.set_rerank(0).unwrap();
    assert_eq!(reopened.search(&q.view(), 10, false).unwrap(), exact);
    reopened.set_rerank(30).unwrap();
    reopened.set_profile(true);
    let coded = reopened.search(&q.view(), 10, false).unwrap();
    let p = reopened.take_profile();
    assert!(p.coded > p.reranked && p.reranked == 30 * q.nrows() as u64, "{p:?}");
    assert_eq!(coded.0, exact.0);
    drop(reopened);
    std::fs::remove_file(dir.path().join(row_order::ROW_ORDER_FILE)).unwrap();
    let bare = O1Store::open_forest(dir.path().to_path_buf(), D, 1).unwrap();
    assert_eq!(bare.maint_stats().unwrap().ordered_rows, 0);
    assert!(std::fs::read_dir(dir.path()).unwrap().all(|e| !e.unwrap().file_name().to_string_lossy().starts_with("o1f_xo_")));
    assert_eq!(bare.search(&q.view(), 10, false).unwrap(), plain.search(&q.view(), 10, false).unwrap());
}
