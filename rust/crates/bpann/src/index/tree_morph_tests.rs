use super::*;
use crate::index::search::MmapSearchStore;
use crate::index::tree::{TREE_FANOUT, TREE_LEAF_CAPACITY};
use crate::index::tree_search::search_tree;
use crate::index::IncrementalIndex;
use crate::mmap_store::MmapColumnStore;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tempfile::TempDir;

fn store_with_rows(dir: &TempDir, n: usize, d: usize, seed: u64) -> MmapColumnStore {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let x = ndarray::Array2::from_shape_fn((n, d), |_| rng.gen::<f64>());
    let mut store = MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), d, None).unwrap();
    store.mmap_append(&x.view()).unwrap();
    store
}

fn scaled_row(store: &MmapColumnStore, row: u32, scale: &[f64]) -> Vec<f32> {
    let x = store.mmap_row_slice(row as usize).unwrap();
    x.iter().zip(scale).map(|(&v, s)| (v * s) as f32).collect()
}

/// Every row once, counts and parents consistent, page limits kept, radii bound rows.
fn assert_valid(idx: &IncrementalIndex, store: &MmapColumnStore, scale: &[f64]) {
    let index = &idx.indices[0];
    let mut ids = index.leaf_row_ids();
    ids.sort_unstable();
    assert_eq!(ids, (0..idx.indexed_rows as u32).collect::<Vec<_>>());
    assert_eq!(index.header.indexed_rows, idx.indexed_rows);
    let recount = TreeCounts::from_index(index).expect("valid tree");
    assert_eq!(idx.counts.parent(index.header.root_page_id), None);
    for page in &index.pages {
        let id = page.page_id();
        assert_eq!(idx.counts.count(id), recount.count(id));
        assert_eq!(idx.counts.parent(id), recount.parent(id));
        match page {
            Page::Leaf { row_ids, .. } => {
                assert!(row_ids.len() <= TREE_LEAF_CAPACITY);
                assert_eq!(idx.counts.block(id).len(), row_ids.len() * scale.len());
            }
            Page::Internal { child_page_ids, .. } => {
                assert!((2..=TREE_FANOUT).contains(&child_page_ids.len()), "{} children", child_page_ids.len());
                let block = idx.counts.block(id);
                for (c, &child) in block.chunks_exact(scale.len()).zip(child_page_ids) {
                    let mut rows = Vec::new();
                    rows_below(index, child, &mut rows);
                    for row in rows {
                        let d = dist(&scaled_row(store, row, scale), c);
                        assert!(d <= idx.counts.radius(child) * 1.0001 + 1e-5, "row {row} outside page {child}");
                    }
                }
            }
        }
    }
}

fn rows_below(index: &BpannIndex, page_id: u32, out: &mut Vec<u32>) {
    match index.page_by_id(page_id).unwrap() {
        Page::Leaf { row_ids, .. } => out.extend(row_ids),
        Page::Internal { child_page_ids, .. } => child_page_ids.iter().for_each(|&c| rows_below(index, c, out)),
    }
}

/// Mean squared distance of a row to the mean of its leaf.
fn leaf_spread(idx: &IncrementalIndex, d: usize) -> f64 {
    let (mut total, mut n) = (0.0f64, 0usize);
    for page in &idx.indices[0].pages {
        let Page::Leaf { page_id, row_ids, .. } = page else { continue };
        let rows: Vec<&[f32]> = idx.counts.block(*page_id).chunks_exact(d).collect();
        let mean: Vec<f64> = (0..d).map(|j| rows.iter().map(|r| f64::from(r[j])).sum::<f64>() / rows.len() as f64).collect();
        total += rows.iter().map(|r| r.iter().zip(&mean).map(|(&x, m)| (f64::from(x) - m).powi(2)).sum::<f64>()).sum::<f64>();
        n += row_ids.len();
    }
    total / n as f64
}

fn sync(idx: &mut IncrementalIndex, store: &MmapColumnStore, scale: &[f64], end: usize) {
    idx.ensure_sync_for_backend(store, scale.len(), true, &scale.iter().map(|s| 1.0 / s).collect::<Vec<_>>(), end).unwrap();
}

#[test]
fn morph_lap_keeps_the_tree_valid_and_reaches_a_fresh_partition() {
    let (n0, d) = (8_000usize, 4usize);
    let scale = [1.0, 1.0, 0.02, 0.02];
    let dir = TempDir::new().unwrap();
    let store = store_with_rows(&dir, 2 * n0, d, 1);
    let mut idx = IncrementalIndex::new(dir.path().join("index"));
    sync(&mut idx, &store, &[1.0; 4], n0);
    idx.rescale_coords(&scale);
    let before = leaf_spread(&idx, d);
    idx.start_morph();
    let mut end = n0;
    while idx.morphing() {
        end += 100;
        assert!(end <= n0 + MORPH_LAPS * 2 * end / MORPH_WORK_PER_ADD + 1000, "morph still running at {end}");
        sync(&mut idx, &store, &scale, end);
    }
    assert_valid(&idx, &store, &scale);
    let after = leaf_spread(&idx, d);
    let fresh_dir = TempDir::new().unwrap();
    let mut fresh = IncrementalIndex::new(fresh_dir.path().join("index"));
    sync(&mut fresh, &store, &scale, end);
    let target = leaf_spread(&fresh, d);
    assert!(after < 0.2 * before, "spread {before} -> {after}");
    assert!(after < 1.2 * target, "spread {after} vs fresh {target}");
}

#[test]
fn morph_moves_references_only_and_keeps_every_page() {
    let (n0, d) = (6_000usize, 3usize);
    let scale = [1.0, 0.01, 0.01];
    let dir = TempDir::new().unwrap();
    let store = store_with_rows(&dir, n0, d, 5);
    let mut idx = IncrementalIndex::new(dir.path().join("index"));
    sync(&mut idx, &store, &[1.0; 3], n0);
    idx.rescale_coords(&scale);
    let mut pages: Vec<u32> = idx.indices[0].pages.iter().map(Page::page_id).collect();
    pages.sort_unstable();
    let before = leaf_spread(&idx, d);
    idx.start_morph();
    while idx.morphing() {
        idx.morph.advance(&mut idx.indices[0], &mut idx.counts);
    }
    let mut after_pages: Vec<u32> = idx.indices[0].pages.iter().map(Page::page_id).collect();
    after_pages.sort_unstable();
    assert_eq!(after_pages, pages);
    assert_valid(&idx, &store, &scale);
    assert!(leaf_spread(&idx, d) < 0.2 * before);
}

#[test]
fn pages_with_two_leaves_are_pooled_with_their_neighbors() {
    let (n0, d) = (1_600usize, 3usize);
    let scale = [1.0, 0.01, 0.01];
    let dir = TempDir::new().unwrap();
    let store = store_with_rows(&dir, n0, d, 6);
    let rows: Vec<u32> = (0..n0 as u32).collect();
    let mut vs: Vec<f32> = (0..n0).flat_map(|r| scaled_row(&store, r as u32, &[1.0; 3])).collect();
    let (index, counts) = crate::index::tree_bulk::bulk_build(&rows, &mut vs, d, dir.path().join("index")).unwrap();
    let pairs = index.pages.iter().filter(|p| matches!(p, Page::Internal { child_page_ids, .. } if child_page_ids.len() == 2));
    assert_eq!(pairs.count(), 16);
    let mut idx = IncrementalIndex::new(dir.path().join("index"));
    (idx.indices, idx.counts, idx.indexed_rows) = (vec![index], counts, n0);
    idx.rescale_coords(&scale);
    let before = leaf_spread(&idx, d);
    idx.start_morph();
    while idx.morphing() {
        idx.morph.advance(&mut idx.indices[0], &mut idx.counts);
    }
    assert_valid(&idx, &store, &scale);
    assert!(leaf_spread(&idx, d) < 0.2 * before);
}

#[test]
fn bisect_gives_near_equal_groups_of_every_member() {
    let pts: Vec<Vec<f32>> = (0..37).map(|i| vec![(i * 7 % 37) as f32, 0.0]).collect();
    let points: Vec<&[f32]> = pts.iter().map(Vec::as_slice).collect();
    let groups = groups_of(&points, 5);
    assert_eq!(groups.len(), 5);
    assert!(groups.iter().all(|g| (7..=8).contains(&g.len())));
    let mut all: Vec<usize> = groups.concat();
    all.sort_unstable();
    assert_eq!(all, (0..37).collect::<Vec<_>>());
    let lo = |g: &[usize]| g.iter().map(|&i| pts[i][0] as i32).min().unwrap();
    let hi = |g: &[usize]| g.iter().map(|&i| pts[i][0] as i32).max().unwrap();
    let mut spans: Vec<(i32, i32)> = groups.iter().map(|g| (lo(g), hi(g))).collect();
    spans.sort_unstable();
    assert!(spans.windows(2).all(|w| w[0].1 < w[1].0), "groups overlap: {spans:?}");
}

#[test]
fn unbounded_search_stays_exact_during_a_morph() {
    let (n0, d) = (3_000usize, 3usize);
    let scale = [0.1, 1.0, 3.0];
    let dir = TempDir::new().unwrap();
    let store = store_with_rows(&dir, n0 + 400, d, 2);
    let mut idx = IncrementalIndex::new(dir.path().join("index"));
    sync(&mut idx, &store, &[1.0; 3], n0);
    idx.rescale_coords(&scale);
    idx.start_morph();
    let x_scale = scale.map(|v| 1.0 / v);
    let s = MmapSearchStore {
        train_x: &store,
        scale_x: true,
        x_scale: &x_scale,
    };
    let mut rng = ChaCha8Rng::seed_from_u64(9);
    for end in (n0 + 50..=n0 + 400).step_by(50) {
        sync(&mut idx, &store, &scale, end);
        assert_valid(&idx, &store, &scale);
        let q: Vec<f32> = scale.iter().map(|&s| rng.gen::<f32>() * s as f32).collect();
        let got = search_tree(&idx.indices[0], &idx.counts, &q, 5, usize::MAX, &s).unwrap();
        let mut all: Vec<(f32, u32)> =
            (0..end as u32).map(|r| (crate::distance::l2_sq_f32(&q, &scaled_row(&store, r, &scale)), r)).collect();
        all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        assert_eq!(got.iter().map(|g| g.0).collect::<Vec<_>>(), all[..5].iter().map(|a| a.1).collect::<Vec<_>>());
    }
}

#[test]
fn morphed_tree_persists_and_is_adopted_on_reopen() {
    let dir = TempDir::new().unwrap();
    let store = store_with_rows(&dir, 2_500, 2, 3);
    let scale = [1.0, 0.05];
    let mut idx = IncrementalIndex::new(dir.path().join("index"));
    sync(&mut idx, &store, &[1.0; 2], 2_000);
    idx.rescale_coords(&scale);
    idx.start_morph();
    sync(&mut idx, &store, &scale, 2_500);
    assert!(!idx.morphing());
    idx.persist_to_disk_for_backend(&store, 2, true, &[1.0, 20.0], dir.path(), 1).unwrap();
    let mut reopened = IncrementalIndex::new(idx.index_dir.clone());
    assert!(reopened.adopt_persisted(BpannIndex::open(idx.index_dir.clone()).unwrap()));
    sync(&mut reopened, &store, &scale, 2_500);
    assert_valid(&reopened, &store, &scale);
}

#[test]
fn inactive_morph_spends_nothing_and_a_lone_leaf_is_left_alone() {
    let dir = TempDir::new().unwrap();
    let store = store_with_rows(&dir, 40, 2, 4);
    let mut idx = IncrementalIndex::new(dir.path().join("index"));
    sync(&mut idx, &store, &[1.0; 2], 30);
    idx.start_morph();
    assert!(idx.morphing());
    sync(&mut idx, &store, &[1.0; 2], 40);
    assert!(!idx.morphing());
    assert_eq!(idx.indices[0].pages.len(), 1);
    assert_valid(&idx, &store, &[1.0; 2]);
}

#[test]
fn morph_leaves_a_tree_that_fits_the_metric_no_looser() {
    let (n0, d) = (6_400usize, 4usize);
    let dir = TempDir::new().unwrap();
    let store = store_with_rows(&dir, n0, d, 7);
    let rows: Vec<u32> = (0..n0 as u32).collect();
    let mut vs: Vec<f32> = (0..n0).flat_map(|r| scaled_row(&store, r as u32, &[1.0; 4])).collect();
    let (index, counts) = crate::index::tree_bulk::bulk_build(&rows, &mut vs, d, dir.path().join("index")).unwrap();
    let mut idx = IncrementalIndex::new(dir.path().join("index"));
    (idx.indices, idx.counts, idx.indexed_rows) = (vec![index], counts, n0);
    let before = leaf_spread(&idx, d);
    idx.start_morph();
    while idx.morphing() {
        idx.morph.advance(&mut idx.indices[0], &mut idx.counts);
    }
    assert_valid(&idx, &store, &[1.0; 4]);
    let after = leaf_spread(&idx, d);
    assert!(after <= before * (1.0 + 1e-9), "spread {before} -> {after}");
}
