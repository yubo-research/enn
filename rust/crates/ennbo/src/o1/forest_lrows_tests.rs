use super::forest_lrows::NearRows;
use super::forest_node::Node;
use super::forest_repair_tests::{relevant, store_at, uniform, weighted_recall};
use super::forest_tests::{assert_invariants, D};
use super::*;
use tempfile::TempDir;

fn assert_leaves_hold_rows(store: &mut O1Store) -> usize {
    store.index.forest_mut().compact().unwrap();
    let x: Vec<Vec<f64>> = (0..store.len())
        .map(|i| store.row_x(i).unwrap().to_vec())
        .collect();
    let mut held = 0;
    for tree in &store.index.forest_mut().trees {
        let rows = tree.lrows.as_ref().expect("clustered tree has a leaf file");
        let mut leaves = 0;
        for page in 1..tree.nodes.num_nodes() {
            let Ok(Node::Leaf { ids, .. }) = tree.nodes.read(page) else {
                continue;
            };
            leaves += 1;
            let mut near = NearRows::new();
            rows.get(page, None, &mut near).unwrap();
            for id in ids {
                if let Some(v) = near.get(&id) {
                    assert_eq!(v, &x[id as usize], "page {page} row {id}");
                }
                held += usize::from(near.contains_key(&id));
            }
        }
        assert_eq!(rows.regions(), leaves);
    }
    held
}

fn clustered_build() -> (TempDir, O1Store, usize) {
    std::env::set_var("ENN_CLUSTER_LEAVES", "lazy");
    let dir = TempDir::new().unwrap();
    let x = uniform(3000, 1);
    let mut store = store_at(&dir, &x, &relevant(true));
    let first = assert_leaves_hold_rows(&mut store);
    store.set_metric_scale(&relevant(false), true).unwrap();
    while store.occupancy().unwrap().stale_trees > 0 {
        store.maintain_index(700).unwrap();
    }
    assert_invariants(&store);
    assert_eq!(store.index.forest_mut().order.as_ref().map(|o| o.rows()), None);
    let q = uniform(30, 2);
    store.set_probe_budget(1 << 20, 1 << 20).unwrap();
    assert_eq!(weighted_recall(&store, &q, &relevant(false), 10), 1.0);
    (dir, store, first)
}

#[test]
fn lazy_clustered_leaves_hold_rows_from_their_last_split_and_answer_exactly() {
    let (dir, mut store, first) = clustered_build();
    assert!(first > 4 * 3000 && first < 8 * 3000, "{first}");
    let held = assert_leaves_hold_rows(&mut store);
    assert!(held > 4 * 3000 && held < 8 * 3000, "{held}");
    drop(store);
    let mut store = O1Store::open_forest(dir.path().to_path_buf(), D, 1).unwrap();
    assert_eq!(assert_leaves_hold_rows(&mut store), held);
}
