use super::forest_node::{Cut, Node};
use super::forest_tests::{assert_invariants, insert_all, trajectory, D};
use super::*;
use ndarray::{Array1, Array2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;

fn pivot_store(dir: &TempDir, cache_bytes: usize, rule: SplitRule) -> O1Store {
    let mut store = O1Store::open_forest(dir.path().to_path_buf(), D, 1).unwrap();
    store.set_trees(8).unwrap();
    store.set_split_rule(rule).unwrap();
    store.set_cache_budget(cache_bytes);
    store.index.forest_mut().params.rebuild_below = 0;
    let x = trajectory(2400, 13);
    insert_all(&mut store, &x.slice(ndarray::s![..1200, ..]).to_owned(), 40);
    store.set_repair(5, 0.25, true).unwrap();
    let scale = Array1::from_iter((0..D).map(|j| if j < D / 2 { 0.3 } else { 1.0 }));
    store.set_metric_scale(&scale, true).unwrap();
    insert_all(&mut store, &x.slice(ndarray::s![1200.., ..]).to_owned(), 20);
    store
}

fn cut_counts(store: &O1Store) -> (usize, usize, usize) {
    let f = store.index.forest();
    let (mut seeded, mut pivots, mut dirs) = (0, 0, 0);
    for tree in &f.trees {
        for id in 1..tree.nodes.num_nodes() {
            match tree.nodes.read(id).unwrap() {
                Node::Split { cut: Cut::Seed(_), .. } => seeded += 1,
                Node::Split { cut: Cut::Pivot(a, b), .. } => {
                    assert!(a != b && (a as usize) < store.len() && (b as usize) < store.len());
                    pivots += 1;
                }
                Node::Split { cut: Cut::Dir(d), .. } => {
                    assert_eq!(d.len(), store.num_dim());
                    dirs += 1;
                }
                _ => {}
            }
        }
    }
    (seeded, pivots, dirs)
}

#[test]
fn pivot_splits_keep_invariants_cache_equivalence_and_reopen() {
    for rule in [SplitRule::Pivot, SplitRule::PivotI8, SplitRule::PivotF32] {
        check_pivot_rule(rule);
    }
}

fn check_pivot_rule(rule: SplitRule) {
    let (plain_dir, cached_dir) = (TempDir::new().unwrap(), TempDir::new().unwrap());
    let plain = pivot_store(&plain_dir, 0, rule);
    let cached = pivot_store(&cached_dir, 48 << 10, rule);
    let (seeded, pivots, dirs) = cut_counts(&plain);
    let (routed, other) = if rule == SplitRule::Pivot { (pivots, dirs) } else { (dirs, pivots) };
    assert!(routed > 10 * seeded.max(1) && other == 0);
    let mut rng = StdRng::seed_from_u64(4);
    let q = Array2::from_shape_fn((30, D), |_| rng.gen::<f64>());
    let answer = plain.search(&q.view(), 10, false).unwrap();
    assert_eq!(answer, cached.search(&q.view(), 10, false).unwrap());
    let serial: Vec<Vec<u32>> = q.rows().into_iter().map(|r| plain.index.candidate_ids(&r.to_vec(), &Default::default(), &plain.x).unwrap()).collect();
    let (mut plain, mut cached) = (plain, cached);
    for width in [2, 8] {
        plain.set_io_width(width, 0).unwrap();
        plain.set_pread_pivots(width == 2).unwrap();
        cached.set_io_width(width, if width == 2 { 0 } else { 30_000 }).unwrap();
        let wide: Vec<Vec<u32>> = q.rows().into_iter().map(|r| plain.index.candidate_ids(&r.to_vec(), &Default::default(), &plain.x).unwrap()).collect();
        assert_eq!(serial, wide);
        for i in 0..3 {
            let one = plain.search(&q.slice(ndarray::s![i..i + 1, ..]), 10, false).unwrap();
            assert_eq!((one.0.row(0).to_owned(), one.1.row(0).to_owned()), (answer.0.row(i).to_owned(), answer.1.row(i).to_owned()));
        }
        assert_eq!(answer, plain.search(&q.view(), 10, false).unwrap());
        assert_eq!(answer, cached.search(&q.view(), 10, false).unwrap());
    }
    let occ = assert_invariants(&plain);
    assert!(occ.repairs_started > 0 && occ.splits > 0);
    assert_invariants(&cached);
    assert!(cached.cache_stats().unwrap().row_hits > 0);
    drop(plain);
    let mut reopened = O1Store::open_forest(plain_dir.path().to_path_buf(), D, 1).unwrap();
    assert_eq!(answer, reopened.search(&q.view(), 10, false).unwrap());
    assert_eq!(cut_counts(&reopened), (seeded, pivots, dirs));
    let before = same_page_edges(&reopened);
    assert!(reopened.compact_index().unwrap() > 0);
    assert!(same_page_edges(&reopened) > before);
    assert_eq!(cut_counts(&reopened), (seeded, pivots, dirs));
    assert_eq!(answer, reopened.search(&q.view(), 10, false).unwrap());
    assert_eq!(assert_invariants(&reopened).free_nodes, 0);
    drop(reopened);
    let mut packed = O1Store::open_forest(plain_dir.path().to_path_buf(), D, 1).unwrap();
    assert_eq!(answer, packed.search(&q.view(), 10, false).unwrap());
    insert_all(&mut packed, &trajectory(300, 21), 50);
    assert_invariants(&packed);
}

fn same_page_edges(store: &O1Store) -> f64 {
    let (mut same, mut all) = (0, 0);
    for tree in &store.index.forest().trees {
        let rec = tree.nodes.record_bytes() as u64;
        let page = |id: u32| u64::from(id) * rec / super::forest_pack::PAGE_BYTES;
        for id in 1..tree.nodes.num_nodes() {
            if let Node::Split { left, right, .. } = tree.nodes.read(id).unwrap() {
                same += usize::from(page(id) == page(left)) + usize::from(page(id) == page(right));
                all += 2;
            }
        }
    }
    same as f64 / all.max(1) as f64
}

#[test]
fn seeded_rule_writes_no_pivots_and_lsh_rejects_the_rule() {
    let dir = TempDir::new().unwrap();
    let mut store = O1Store::open_forest(dir.path().to_path_buf(), D, 1).unwrap();
    store.set_split_rule(SplitRule::Seeded).unwrap();
    store.set_split_candidates(16).unwrap();
    insert_all(&mut store, &trajectory(600, 2), 50);
    let (seeded, pivots, dirs) = cut_counts(&store);
    assert!(seeded > 0 && pivots + dirs == 0);
    assert_invariants(&store);
    let lsh_dir = TempDir::new().unwrap();
    let mut lsh = O1Store::open_or_create(lsh_dir.path().to_path_buf(), D, 1).unwrap();
    assert!(lsh.set_split_rule(SplitRule::Pivot).is_err());
    let wide_dir = TempDir::new().unwrap();
    let mut wide = O1Store::open_forest(wide_dir.path().to_path_buf(), 40, 1).unwrap();
    let mut rng = StdRng::seed_from_u64(8);
    insert_all(&mut wide, &Array2::from_shape_fn((300, 40), |_| rng.gen::<f64>()), 100);
    assert_eq!(wide.occupancy().unwrap().node_bytes, 24 + 128);
    wide.set_split_rule(SplitRule::PivotF32).unwrap();
    let (_, pivots, dirs) = cut_counts(&wide);
    assert!(pivots == 0 && dirs > 5);
    assert_eq!(assert_invariants(&wide).node_bytes, 24 + 160);
    assert!(lsh.set_io_width(4, 0).is_err());
}
