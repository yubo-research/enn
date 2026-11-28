use super::forest_repair_tests::{relevant, store_at, uniform, weighted_recall};
use super::forest_rules::SplitRule;
use super::forest_tests::{assert_invariants, insert_all, recall, trajectory, D};
use super::*;
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;

fn deferred(dir: &TempDir, rule: Option<SplitRule>, defer_rows: u32, local: u32) -> O1Store {
    let mut store = O1Store::open_forest(dir.path().to_path_buf(), D, 1).unwrap();
    if let Some(rule) = rule {
        store.set_split_rule(rule).unwrap();
    }
    store.set_defer(defer_rows, local).unwrap();
    store
}

const LOCAL: u32 = 300 * (4 + 8 * D as u32);

fn queries(seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array2::from_shape_fn((20, D), |_| 0.5 + 0.02 * (rng.gen::<f64>() - 0.5))
}

#[test]
fn deferred_rebuilds_keep_every_row_once_and_finish() {
    for rule in [None, Some(SplitRule::Pivot)] {
        let dir = TempDir::new().unwrap();
        let mut store = deferred(&dir, rule, 8, 1);
        let x = trajectory(6000, 3);
        insert_all(&mut store, &x, 10);
        assert_invariants(&store);
        let m = store.index.forest().maint_stats();
        assert!(
            m.deferred > 0 && m.rebuilt_rows > 0 && m.scan_nodes > 0,
            "{m:?}"
        );
        assert!(m.max_rebuild_rows > 64);
        eprintln!(
            "STRESS {} {:?} {}",
            recall(&store, &queries(4), 10),
            m,
            store.occupancy().unwrap().hash_bits
        );
    }
}

#[test]
fn compaction_drops_an_unfinished_job_without_losing_rows() {
    let dir = TempDir::new().unwrap();
    let mut store = deferred(&dir, None, 1, 1);
    let x = trajectory(4000, 7);
    insert_all(&mut store, &x.slice(ndarray::s![..3000, ..]).to_owned(), 1);
    let active = store.index.forest().maint_stats().job_active;
    store.compact_index().unwrap();
    assert_eq!(store.index.forest().maint_stats().job_active, 0);
    insert_all(&mut store, &x.slice(ndarray::s![3000.., ..]).to_owned(), 1);
    assert_invariants(&store);
    assert!(active <= 1);
}

#[test]
fn deferred_and_immediate_rebuilds_agree_on_recall() {
    let x = trajectory(4000, 9);
    let dir = TempDir::new().unwrap();
    let mut legacy = O1Store::open_forest(dir.path().to_path_buf(), D, 1).unwrap();
    insert_all(&mut legacy, &x, 1);
    let dir2 = TempDir::new().unwrap();
    let mut store = deferred(&dir2, None, 8, LOCAL);
    insert_all(&mut store, &x, 1);
    let q = queries(5);
    eprintln!(
        "RECALL defer={} legacy={} m={:?} occ={:?}",
        recall(&store, &q, 10),
        recall(&legacy, &q, 10),
        store.index.forest().maint_stats(),
        store.occupancy().unwrap().hash_bits
    );
    assert!(recall(&store, &q, 10) >= recall(&legacy, &q, 10) - 0.1);
}

fn repair_until_fresh(store: &mut O1Store) -> usize {
    let mut steps = 0;
    while store.occupancy().unwrap().stale_trees > 0 && steps < 1000 {
        store.maintain_index(700).unwrap();
        assert_invariants(store);
        steps += 1;
    }
    steps
}

#[test]
fn bulk_repair_rebuilds_stale_trees_and_survives_reopen() {
    let dir = TempDir::new().unwrap();
    let x = uniform(3000, 1);
    {
        let mut store = store_at(&dir, &x, &relevant(true));
        store.set_defer(8, LOCAL).unwrap();
        store.set_metric_scale(&relevant(false), true).unwrap();
        store.maintain_index(700).unwrap();
    }
    let mut store = O1Store::open_forest(dir.path().to_path_buf(), D, 1).unwrap();
    store.set_defer(8, LOCAL).unwrap();
    assert!(repair_until_fresh(&mut store) < 1000);
    let occ = assert_invariants(&store);
    assert_eq!(occ.stale_trees, 0);
    assert!(store.index.forest().maint_stats().deferred >= 8);
    assert!(weighted_recall(&store, &uniform(30, 2), &relevant(false), 10) > 0.95);
}
