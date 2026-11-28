use super::forest_repair_tests::uniform;
use super::forest_tests::{insert_all, trajectory, D};
use super::io_streams::{snapshot, Stream};
use super::*;
use tempfile::TempDir;

fn one_tree(dir: &TempDir) -> O1Store {
    let params = O1IndexParams::Forest(ForestParams {
        num_trees: 1,
        ..ForestParams::default()
    });
    O1Store::open_with_index(dir.path().to_path_buf(), (D, 1), params).unwrap()
}

fn node_bytes(dir: &TempDir) -> Vec<u8> {
    let mut files: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.file_name().unwrap().to_string_lossy().starts_with("o1f_tree_"))
        .collect();
    files.sort();
    files.iter().flat_map(|p| std::fs::read(p).unwrap()).collect()
}

#[test]
fn warm_up_finds_the_insert_leaf_and_changes_nothing() {
    for x in [uniform(4000, 11), trajectory(4000, 12)] {
        let dir = TempDir::new().unwrap();
        let mut store = one_tree(&dir);
        insert_all(&mut store, &x, 500);
        store.index.forest().commit().unwrap();
        let before = node_bytes(&dir);
        let forest = store.index.forest();
        let (tree, inv) = (&forest.trees[0], &forest.invs[0]);
        let rows = super::row_order::Rows::plain(&store.x);
        let ids: Vec<u32> = (0..4000).step_by(7).collect();
        for &id in &ids {
            let xr = store.row_x(id as usize).unwrap().to_vec();
            let want = tree.descend_from(tree.nodes.root(), &xr, inv, rows).unwrap().1;
            assert_eq!(tree.peek_leaf(id, rows, inv).unwrap(), want);
        }
        for width in [1, 3, 10] {
            let mut want = std::collections::HashMap::new();
            for &id in &ids {
                *want.entry(tree.peek_leaf(id, rows, inv).unwrap()).or_insert(0usize) += 1;
            }
            let mut want: Vec<(u32, usize)> = want.into_iter().collect();
            want.sort_unstable();
            assert_eq!(tree.batch_leaves(&ids, rows, inv, width).unwrap(), want);
        }
        let reads0 = tree.nodes.counters.reads.load(std::sync::atomic::Ordering::Relaxed);
        let pf0 = snapshot()[Stream::Prefetch as usize].1[3];
        for width in [1, 3, 10] {
            let (leaves, warmed) = tree.prefetch(&ids, rows, inv, width).unwrap();
            assert!(leaves > 10 && leaves <= ids.len(), "leaves {leaves}");
            assert!(warmed <= leaves);
        }
        assert!(snapshot()[Stream::Prefetch as usize].1[3] > pf0);
        let reads1 = tree.nodes.counters.reads.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(reads0, reads1, "the warm-up must not count as node reads");
        assert_eq!(node_bytes(&dir), before);
    }
}

#[test]
fn a_leaf_is_warmed_only_when_its_arrivals_overflow_it() {
    let dir = TempDir::new().unwrap();
    let mut store = one_tree(&dir);
    insert_all(&mut store, &uniform(3000, 13), 3000);
    let forest = store.index.forest();
    let (tree, inv) = (&forest.trees[0], &forest.invs[0]);
    let rows = super::row_order::Rows::plain(&store.x);
    let one = [5u32];
    let (leaves, warmed) = tree.prefetch(&one, rows, inv, 2).unwrap();
    assert_eq!(leaves, 1);
    let leaf = tree.peek_leaf(5, rows, inv).unwrap();
    let mut ids = Vec::new();
    tree.leaf_ids(leaf, &mut ids).unwrap();
    let full = ids.len() % tree.nodes.cap() == 0;
    assert_eq!(warmed, usize::from(full));
    let many = vec![5u32; tree.nodes.cap() + 1];
    assert_eq!(tree.prefetch(&many, rows, inv, 4).unwrap(), (1, 1));
}

#[test]
fn the_gate_backs_off_while_reads_are_cached_and_rearms_on_cold_reads() {
    use super::forest_prefetch::Gate;
    let mut g = Gate::default();
    let mut runs = 0;
    for _ in 0..400 {
        if g.due() {
            runs += 1;
            g.observe(0, 1000);
        }
    }
    assert!((6..=14).contains(&runs), "runs {runs}");
    while !g.due() {}
    g.observe(10, 1000);
    assert!(g.due() && g.due());
    g.observe(9, 1000);
    assert!(!g.due());
}
