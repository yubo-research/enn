use super::forest_repair::shadow_path;
use super::forest_tests::{assert_invariants, insert_all, trajectory, D};
use super::row_order::Rows;
use super::*;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;

pub(super) fn uniform(n: usize, seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array2::from_shape_fn((n, D), |_| rng.gen::<f64>())
}

pub(super) fn relevant(first: bool) -> Array1<f64> {
    Array1::from_iter((0..D).map(|j| if (j < D / 2) == first { 0.1 } else { 1.0 }))
}

pub(super) fn weighted_recall(
    store: &O1Store,
    q: &Array2<f64>,
    scale: &Array1<f64>,
    k: usize,
) -> f64 {
    let (_, idx) = store.search(&q.view(), k, false).unwrap();
    let mut hit = 0;
    for (qi, row) in q.rows().into_iter().enumerate() {
        let mut all: Vec<(f64, i64)> = (0..store.len())
            .map(|i| {
                let x = store.row_x(i).unwrap();
                let d: f64 = x
                    .iter()
                    .zip(row.iter())
                    .zip(scale.iter())
                    .map(|((a, b), s)| ((a - b) / s).powi(2))
                    .sum();
                (d, i as i64)
            })
            .collect();
        all.sort_by(|a, b| a.0.total_cmp(&b.0));
        let truth: Vec<i64> = all.into_iter().take(k).map(|p| p.1).collect();
        hit += idx.row(qi).iter().filter(|i| truth.contains(i)).count();
    }
    hit as f64 / (k * q.nrows()) as f64
}

pub(super) fn store_at(dir: &TempDir, x: &Array2<f64>, scale: &Array1<f64>) -> O1Store {
    let params = O1IndexParams::Forest(ForestParams {
        num_trees: 8,
        ..ForestParams::default()
    });
    let mut store = O1Store::open_with_index(dir.path().to_path_buf(), (D, 1), params).unwrap();
    store.index.forest_mut().params.rebuild_below = 0;
    store.set_repair(0, 0.25, true).unwrap();
    insert_all(&mut store, &x.slice(ndarray::s![..64, ..]).to_owned(), 64);
    store.set_metric_scale(scale, true).unwrap();
    insert_all(&mut store, &x.slice(ndarray::s![64.., ..]).to_owned(), 100);
    store.set_repair(10, 0.25, true).unwrap();
    store
}

#[test]
fn small_indexes_rebuild_at_once() {
    let dir = TempDir::new().unwrap();
    let mut store = O1Store::open_forest(dir.path().to_path_buf(), D, 1).unwrap();
    insert_all(&mut store, &uniform(1000, 6), 1000);
    store.set_metric_scale(&relevant(true), true).unwrap();
    let occ = assert_invariants(&store);
    assert_eq!((occ.stale_trees, occ.global_rebuilds), (0, 1));
}

#[test]
fn metric_change_is_repaired_in_bounded_steps_without_losing_rows() {
    let dir = TempDir::new().unwrap();
    let x = uniform(3000, 1);
    let mut store = store_at(&dir, &x, &relevant(true));
    let q = uniform(30, 2);
    let writes = store.node_io().unwrap().1;
    store.set_metric_scale(&relevant(false), true).unwrap();
    assert_eq!(store.node_io().unwrap().1, writes);
    let occ = assert_invariants(&store);
    assert_eq!((occ.stale_trees, occ.global_rebuilds), (8, 1));
    let stale_recall = weighted_recall(&store, &q, &relevant(false), 10);
    let budget = (
        store.index.forest_mut().params.max_leaves,
        store.index.forest_mut().params.max_candidates,
    );
    store.set_probe_budget(1 << 20, 1 << 20).unwrap();
    assert_eq!(weighted_recall(&store, &q, &relevant(false), 10), 1.0);
    store.set_probe_budget(budget.0, budget.1).unwrap();
    let mut steps = 0;
    while store.occupancy().unwrap().stale_trees > 0 {
        assert!(store.maintain_index(700).unwrap() <= 700);
        assert_invariants(&store);
        steps += 1;
    }
    let occ = assert_invariants(&store);
    assert_eq!(
        (occ.repairs_finished, occ.repair_rows, steps),
        (8, 8 * 3000, 35)
    );
    assert_eq!(store.maintain_index(700).unwrap(), 0);
    let fresh = weighted_recall(&store, &q, &relevant(false), 10);
    assert!(
        fresh > 0.95 && fresh > stale_recall + 0.2,
        "{stale_recall} -> {fresh}"
    );
}

#[test]
fn inserts_drive_repair_and_reopen_resumes_it() {
    let dir = TempDir::new().unwrap();
    let x = uniform(2600, 3);
    let cursor = {
        let mut store = store_at(
            &dir,
            &x.slice(ndarray::s![..2000, ..]).to_owned(),
            &relevant(true),
        );
        store.set_metric_scale(&relevant(false), true).unwrap();
        insert_all(
            &mut store,
            &x.slice(ndarray::s![2000..2050, ..]).to_owned(),
            50,
        );
        let occ = assert_invariants(&store);
        assert_eq!((occ.repair_rows, occ.repairs_started), (500, 1));
        occ.repair_cursor.unwrap()
    };
    let mut store = O1Store::open_forest(dir.path().to_path_buf(), D, 1).unwrap();
    let occ = assert_invariants(&store);
    assert_eq!(
        (occ.repair_cursor, occ.global_rebuilds, occ.stale_trees),
        (Some(cursor), 0, 8)
    );
    store.set_repair(10, 0.25, true).unwrap();
    insert_all(
        &mut store,
        &x.slice(ndarray::s![2050..2500, ..]).to_owned(),
        50,
    );
    let occ = assert_invariants(&store);
    assert_eq!(occ.repairs_finished, 2);
    assert_eq!(occ.stale_trees, 6);
    let t = (0..8)
        .find(|&t| shadow_path(dir.path(), t).exists())
        .unwrap();
    drop(store);
    std::fs::remove_file(dir.path().join(super::forest::FOREST_STATE_FILE)).unwrap();
    let mut store = O1Store::open_forest(dir.path().to_path_buf(), D, 1).unwrap();
    assert_eq!(assert_invariants(&store).repair_cursor, None);
    assert!(!shadow_path(dir.path(), t).exists());
    store.index.forest_mut().params.rebuild_below = 0;
    store.set_repair(10, 0.25, true).unwrap();
    store.set_metric_scale(&relevant(true), true).unwrap();
    insert_all(
        &mut store,
        &x.slice(ndarray::s![2500..2510, ..]).to_owned(),
        10,
    );
    assert!(assert_invariants(&store).repair_cursor.is_some());
    let n0 = store.len();
    let extra = x.slice(ndarray::s![2510.., ..]).to_owned();
    store
        .y
        .mmap_append(&Array2::<f64>::zeros((extra.nrows(), 1)).view())
        .unwrap();
    store.x.mmap_append(&extra.view()).unwrap();
    store.write_meta().unwrap();
    let f = store.index.forest_mut();
    for t in 0..3 {
        for id in n0..n0 + 30 * (t + 1) {
            f.trees[t]
                .insert(id as u32, Rows::plain(&store.x), &f.invs[t], 0.7)
                .unwrap();
        }
    }
    let inode = |t: usize| {
        std::os::unix::fs::MetadataExt::ino(
            &std::fs::metadata(dir.path().join(format!("o1f_tree_{t}.bin"))).unwrap(),
        )
    };
    let before: Vec<u64> = (0..8).map(inode).collect();
    drop(store);
    let store = O1Store::open_forest(dir.path().to_path_buf(), D, 1).unwrap();
    let occ = assert_invariants(&store);
    assert_eq!((store.len(), occ.repair_cursor), (2600, None));
    assert_eq!((0..8).map(inode).collect::<Vec<u64>>(), before);
    assert!((0..8).all(|t| !shadow_path(dir.path(), t).exists()));
}

#[test]
fn returning_metric_abandons_useless_shadow_work() {
    let dir = TempDir::new().unwrap();
    let x = uniform(2000, 4);
    let mut store = store_at(&dir, &x, &relevant(true));
    store.set_metric_scale(&relevant(false), true).unwrap();
    store.maintain_index(4500).unwrap();
    let occ = store.occupancy().unwrap();
    assert_eq!((occ.repairs_finished, occ.repair_cursor), (2, Some(500)));
    store.set_metric_scale(&relevant(true), true).unwrap();
    let occ = assert_invariants(&store);
    assert_eq!(
        (occ.repairs_abandoned, occ.repair_cursor, occ.stale_trees),
        (1, None, 2)
    );
    assert!(!shadow_path(dir.path(), 2).exists());
    store
        .set_metric_scale(&(relevant(true) * 1.1), true)
        .unwrap();
    assert_eq!(store.occupancy().unwrap().stale_trees, 2);
}

#[test]
fn lost_shadow_file_restarts_repair_and_trajectories_stay_bounded() {
    let dir = TempDir::new().unwrap();
    let x = trajectory(3000, 8);
    {
        let mut store = store_at(&dir, &x, &relevant(true));
        store.set_metric_scale(&relevant(false), true).unwrap();
        store.maintain_index(1200).unwrap();
    }
    std::fs::remove_file(shadow_path(dir.path(), 0)).unwrap();
    let mut store = O1Store::open_forest(dir.path().to_path_buf(), D, 1).unwrap();
    let occ = assert_invariants(&store);
    assert_eq!(
        (occ.repair_cursor, occ.global_rebuilds, occ.stale_trees),
        (None, 0, 8)
    );
    store.maintain_index(8 * 3000).unwrap();
    let occ = assert_invariants(&store);
    let p = ForestParams::default();
    assert_eq!((occ.stale_trees, p.repair_rows), (0, 30));
    assert!(occ.hash_bits <= super::forest_tree::depth_limit(3000, p.leaf_cap, p.alpha) + 1);
}

#[test]
fn catch_up_room_is_capped_only_when_a_rate_is_set() {
    use super::forest_repair::room_at;
    assert_eq!(room_at(None, 30_000, 30, 30), 30_000);
    assert_eq!(room_at(Some(3), 30_000, 30, 30), 3_000);
    assert_eq!(room_at(Some(100), 30_000, 30, 30), 30_000);
    assert_eq!(room_at(Some(3), 30_000, 30, 0), 30_000);
    assert_eq!(room_at(Some(3), 30_000, 0, 30), 30_000);
}
