use super::forest_repair_tests::uniform;
use super::forest_tests::{assert_invariants, insert_all, recall, trajectory, D};
use super::*;
use tempfile::TempDir;

fn buffered(dir: &TempDir, cap: usize) -> O1Store {
    let params = O1IndexParams::Forest(ForestParams {
        num_trees: 4,
        buffer: cap,
        ..ForestParams::default()
    });
    let mut store = O1Store::open_with_index(dir.path().to_path_buf(), (D, 1), params).unwrap();
    store.set_probe_budget(1 << 20, 1 << 20).unwrap();
    store
}

#[test]
fn buffered_rows_are_searchable_exactly_and_flush_into_every_tree() {
    let dir = TempDir::new().unwrap();
    let mut store = buffered(&dir, 1000);
    let x = uniform(2600, 3);
    insert_all(&mut store, &x, 100);
    let forest = store.index.forest();
    assert_eq!(forest.num_rows, 2000);
    assert_eq!(forest.maint_stats().flushes, 2);
    let q = x.slice(ndarray::s![2500..2510, ..]).to_owned();
    let (_, idx) = store.search(&q.view(), 1, false).unwrap();
    assert!(idx.iter().enumerate().all(|(i, &id)| id == 2500 + i as i64));
    assert_eq!(recall(&store, &uniform(20, 4), 10), 1.0);
    store.maintain_index(0).unwrap();
    assert_eq!(store.index.forest().num_rows, 2600);
    assert_invariants(&store);
    assert_eq!(recall(&store, &uniform(20, 5), 10), 1.0);
}

#[test]
fn reopen_indexes_rows_left_in_the_buffer() {
    let dir = TempDir::new().unwrap();
    let x = trajectory(1500, 2);
    {
        let mut store = buffered(&dir, 1000);
        insert_all(&mut store, &x, 100);
        assert_eq!(store.index.forest().num_rows, 1000);
    }
    let store = buffered(&dir, 1000);
    assert_eq!(store.len(), 1500);
    assert_eq!(store.index.forest().num_rows, 1500);
    assert_invariants(&store);
}

#[test]
fn grouped_order_is_a_permutation_that_keeps_leaf_mates_adjacent() {
    let dir = TempDir::new().unwrap();
    let mut store = buffered(&dir, 1);
    insert_all(&mut store, &uniform(3000, 6), 3000);
    let forest = store.index.forest();
    let tree = &forest.trees[0];
    let rows = super::row_order::Rows::plain(&store.x);
    let ids: Vec<u32> = (0..3000).collect();
    let order = tree.batch_order(&ids, rows, &forest.invs[0]).unwrap();
    let mut sorted = order.clone();
    sorted.sort_unstable();
    assert_eq!(sorted, ids);
    let leaves: Vec<u32> = order
        .iter()
        .map(|&id| {
            let x = store.row_x(id as usize).unwrap().to_vec();
            tree.descend_from(tree.nodes.root(), &x, &forest.invs[0], rows)
                .unwrap()
                .1
        })
        .collect();
    let mut runs = std::collections::HashSet::new();
    for (k, leaf) in leaves.iter().enumerate() {
        if k == 0 || leaves[k - 1] != *leaf {
            assert!(runs.insert(*leaf), "leaf {leaf} split across runs");
        }
    }
    assert!(runs.len() > 50);
}

#[test]
fn repair_runs_while_rows_wait_and_never_duplicates_them() {
    use super::forest_repair_tests::{relevant, weighted_recall};
    let dir = TempDir::new().unwrap();
    let params = O1IndexParams::Forest(ForestParams {
        num_trees: 8,
        buffer: 700,
        ..ForestParams::default()
    });
    let mut store = O1Store::open_with_index(dir.path().to_path_buf(), (D, 1), params).unwrap();
    store.index.forest_mut().params.rebuild_below = 0;
    store.set_repair(0, 0.25, true).unwrap();
    let x = uniform(6000, 8);
    insert_all(&mut store, &x.slice(ndarray::s![..64, ..]).to_owned(), 64);
    store.set_metric_scale(&relevant(true), true).unwrap();
    insert_all(&mut store, &x.slice(ndarray::s![64..3000, ..]).to_owned(), 100);
    store.set_repair(10, 0.25, true).unwrap();
    store.set_metric_scale(&relevant(false), true).unwrap();
    let stale = store.occupancy().unwrap().stale_trees;
    insert_all(&mut store, &x.slice(ndarray::s![3000.., ..]).to_owned(), 100);
    assert!(store.index.forest().num_rows < store.len());
    assert!(store.occupancy().unwrap().stale_trees < stale, "{stale}");
    store.maintain_index(0).unwrap();
    while store.occupancy().unwrap().stale_trees > 0 {
        store.maintain_index(5000).unwrap();
    }
    assert_invariants(&store);
    assert!(weighted_recall(&store, &uniform(30, 9), &relevant(false), 10) > 0.95);
}

#[test]
fn deferred_rebuilds_keep_buffered_rows() {
    let dir = TempDir::new().unwrap();
    let mut store = buffered(&dir, 1000);
    store.set_defer(8, 1).unwrap();
    insert_all(&mut store, &trajectory(5000, 3), 100);
    store.maintain_index(0).unwrap();
    let occ = store.occupancy().unwrap();
    assert_eq!(occ.reachable_rows, store.len(), "{occ:?}");
    assert_invariants(&store);
}
