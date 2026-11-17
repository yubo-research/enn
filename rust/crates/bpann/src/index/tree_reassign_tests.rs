use super::*;
use crate::index::tree::TREE_FANOUT;
use crate::index::IncrementalIndex;
use crate::mmap_store::MmapColumnStore;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tempfile::TempDir;

fn store_with_rows(dir: &TempDir, n: usize, d: usize, seed: u64) -> MmapColumnStore {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let x = ndarray::Array2::from_shape_fn((n, d), |_| rng.gen::<f64>());
    let mut store =
        MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), d, None).unwrap();
    store.mmap_append(&x.view()).unwrap();
    store
}

fn scaled_row(store: &MmapColumnStore, row: u32, scale: &[f64]) -> Vec<f32> {
    let x = store.mmap_row_slice(row as usize).unwrap();
    x.iter().zip(scale).map(|(&v, s)| (v * s) as f32).collect()
}

fn tree(idx: &IncrementalIndex) -> &Tree {
    idx.tree.as_ref().expect("tree")
}

/// Every row once, counts and parents consistent, page limits kept, radii bound rows.
fn assert_valid(idx: &IncrementalIndex, store: &MmapColumnStore, scale: &[f64]) {
    let t = tree(idx);
    assert_eq!(
        t.leaf_row_ids(),
        (0..idx.indexed_rows as u32).collect::<Vec<_>>()
    );
    assert_eq!(t.header.indexed_rows, idx.indexed_rows);
    t.assert_counts_consistent();
    assert_eq!(t.store.parent(t.header.root_page_id), None);
    let s = &t.store;
    for id in (0..t.num_pages() as u32).filter(|&p| !s.is_free(p)) {
        if s.is_leaf(id) {
            assert!(s.len(id) <= TREE_LEAF_CAPACITY);
            continue;
        }
        assert!(
            (2..=TREE_FANOUT).contains(&s.len(id)),
            "{} children",
            s.len(id)
        );
        let routed: Vec<f32> = s.ids(id).iter().map(|&c| s.radius(c)).collect();
        assert_eq!(s.child_radii(id), &routed[..], "stale routes in page {id}");
        for (c, &child) in s.block(id).chunks_exact(scale.len()).zip(s.ids(id)) {
            let mut rows = Vec::new();
            rows_below(t, child, &mut rows);
            for row in rows {
                let d = dist(&scaled_row(store, row, scale), c);
                assert!(
                    d <= s.radius(child) * 1.0001 + 1e-5,
                    "row {row} outside page {child}: d={d} r={} leaf={} err={}",
                    s.radius(child),
                    s.is_leaf(child),
                    if s.is_leaf(child) {
                        s.leaf_err(child)
                    } else {
                        0.0
                    }
                );
            }
        }
    }
}

fn rows_below(t: &Tree, page_id: u32, out: &mut Vec<u32>) {
    if t.store.is_leaf(page_id) {
        out.extend(t.store.ids(page_id));
    } else {
        t.store
            .ids(page_id)
            .iter()
            .for_each(|&c| rows_below(t, c, out));
    }
}

fn sync(idx: &mut IncrementalIndex, store: &MmapColumnStore, scale: &[f64], end: usize) {
    idx.ensure_sync_for_backend(
        store,
        scale.len(),
        true,
        &scale.iter().map(|s| 1.0 / s).collect::<Vec<_>>(),
        end,
    )
    .unwrap();
}

/// Mean squared distance of each row to its leaf mean, from decoded leaf rows.
fn spread(t: &Tree) -> f64 {
    let d = t.store.num_dim();
    let (mut tot, mut n) = (0.0f64, 0usize);
    for p in (0..t.num_pages() as u32).filter(|&p| t.store.is_leaf(p) && !t.store.is_free(p)) {
        let flat = t.store.leaf_rows(p);
        let rows: Vec<&[f32]> = flat.chunks_exact(d).collect();
        if rows.is_empty() {
            continue;
        }
        let ones = vec![1.0; rows.len()];
        let all: Vec<usize> = (0..rows.len()).collect();
        let c = weighted_mean(&rows, &ones, &all);
        tot += rows
            .iter()
            .map(|r| f64::from(l2_sq_f32(r, &c)))
            .sum::<f64>();
        n += rows.len();
    }
    tot / n as f64
}

fn build(m: usize, every: usize, lloyd: usize, graph: bool) -> f64 {
    set_reassign(m, every);
    crate::index::tree_lloyd::set_lloyd(lloyd, 16, 8, graph);
    let dir = TempDir::new().unwrap();
    let scale = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let store = store_with_rows(&dir, 3_000, scale.len(), 7);
    let mut idx = IncrementalIndex::new(dir.path().join("idx"));
    for end in (300..=3_000).step_by(300) {
        sync(&mut idx, &store, &scale, end);
    }
    assert_valid(&idx, &store, &scale);
    let s = spread(idx.tree.as_ref().unwrap());
    set_reassign(0, 0);
    crate::index::tree_lloyd::set_lloyd(0, 16, 8, false);
    s
}

// One test per variant (each against the same deterministic base) keeps every
// test well under the kiss per-test time limit when the suite runs in parallel.
#[test]
fn reassignment_keeps_the_tree_valid_and_tightens_leaves() {
    let base = build(0, 0, 0, false);
    let split_only = build(16, 0, 0, false);
    assert!(split_only < base, "split-time pass: {split_only} vs {base}");
}

#[test]
fn periodic_reassignment_tightens_leaves() {
    let base = build(0, 0, 0, false);
    let periodic = build(16, 8, 0, false);
    assert!(periodic < base, "periodic pass: {periodic} vs {base}");
}

#[test]
fn lloyd_sweep_tightens_leaves() {
    let base = build(0, 0, 0, false);
    let swept = build(0, 0, 4, false);
    assert!(swept < base, "lloyd sweep: {swept} vs {base}");
}

#[test]
fn graph_lloyd_sweep_tightens_leaves() {
    let base = build(0, 0, 0, false);
    let graphed = build(0, 0, 4, true);
    assert!(graphed < base, "graph sweep: {graphed} vs {base}");
}

fn recall(t: &Tree, store: &MmapColumnStore, rows: usize, mode: usize) -> f64 {
    use crate::index::search::MmapSearchStore;
    use crate::index::tree_graph_walk::set_local_walk_graph;
    let x_scale = vec![1.0; 6];
    let s = MmapSearchStore {
        train_x: store,
        scale_x: false,
        x_scale: &x_scale,
    };
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let mut hit = 0usize;
    set_local_walk_graph(Some(mode));
    for _ in 0..30 {
        let q: Vec<f64> = (0..6).map(|_| rng.gen::<f64>()).collect();
        let got = crate::index::tree_search::search_tree_refined(t, &q, 10, &s).unwrap();
        let mut all: Vec<(f64, u32)> = (0..rows as u32)
            .map(|r| {
                let x = store.mmap_row_slice(r as usize).unwrap();
                (x.iter().zip(&q).map(|(a, b)| (a - b) * (a - b)).sum(), r)
            })
            .collect();
        all.sort_by(|a, b| a.0.total_cmp(&b.0));
        let truth: Vec<u32> = all[..10].iter().map(|p| p.1).collect();
        hit += got.iter().filter(|p| truth.contains(&p.0)).count();
    }
    set_local_walk_graph(None);
    hit as f64 / 300.0
}

#[test]
fn graph_ordered_walks_find_the_neighbors() {
    crate::index::tree_lloyd::set_lloyd(4, 16, 8, true);
    let dir = TempDir::new().unwrap();
    let scale = [1.0; 6];
    let store = store_with_rows(&dir, 3_000, scale.len(), 9);
    let mut idx = IncrementalIndex::new(dir.path().join("idx"));
    for end in (300..=3_000).step_by(300) {
        sync(&mut idx, &store, &scale, end);
    }
    crate::index::tree_lloyd::set_lloyd(0, 16, 8, false);
    let t = tree(&idx);
    let base = recall(t, &store, 3_000, 0);
    for mode in [1, 2] {
        let r = recall(t, &store, 3_000, mode);
        assert!(r >= base - 0.02 && r > 0.95, "mode {mode}: {r} vs {base}");
    }
}

fn build_buffered(flush: bool) -> f64 {
    crate::index::tree_pbuf::set_pbuf_rows(50);
    crate::index::tree_pbuf_all::set_pbuf_all(true);
    crate::index::tree_lloyd::set_lloyd_flush(flush);
    let s = build(0, 0, 4, true);
    crate::index::tree_lloyd::set_lloyd_flush(false);
    crate::index::tree_pbuf_all::set_pbuf_all(false);
    crate::index::tree_pbuf::set_pbuf_rows(0);
    s
}

#[test]
fn lloyd_passes_at_buffer_flush_tighten_leaves() {
    let base = build_buffered(false);
    let flushed = build_buffered(true);
    assert!(flushed < base, "lloyd at flush: {flushed} vs {base}");
}
