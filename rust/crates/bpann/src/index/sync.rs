use std::fs;
use std::path::PathBuf;

use crate::distance::bpann_row_to_f32;
use crate::error::BpannError;
use crate::index::build::IndexHeader;
use crate::index::search::MmapSearchStore;
use crate::index::tree::Tree;
use crate::index::tree_bulk::{bulk_build, bulk_build_max_rows, BULK_BUILD_MIN_ROWS};
use crate::index::tree_morph::Morph;
use crate::index::tree_search::{leaf_budget, search_tree};
use crate::mmap_store::MmapColumnStore;
use crate::observation as obs;

/// The live BPANN index: at most one incremental tree (see [`crate::index::tree`]).
///
/// `tree` is `None` until the first row is indexed; rows `indexed_rows..` of the
/// store are not yet in it. After a metric change, `morph` re-partitions the tree
/// in place by moving row and page references below one internal page per inserted
/// row ([`crate::index::tree_morph`]).
pub struct IncrementalIndex {
    pub tree: Option<Tree>,
    pub indexed_rows: usize,
    pub index_dir: PathBuf,
    pub(crate) morph: Morph,
}

impl Clone for IncrementalIndex {
    /// Copies the tree into new scratch files (a chunk at a time).
    fn clone(&self) -> Self {
        Self {
            tree: self.tree.as_ref().map(|t| t.try_clone().expect("copy tree store")),
            indexed_rows: self.indexed_rows,
            index_dir: self.index_dir.clone(),
            morph: self.morph.clone(),
        }
    }
}

impl IncrementalIndex {
    pub fn new(index_dir: PathBuf) -> Self {
        Self {
            tree: None,
            indexed_rows: 0,
            index_dir,
            morph: Morph::default(),
        }
    }

    /// Adopt the tree persisted in `index_dir`. Returns false (and stays empty) if it
    /// is not an incremental row-id tree, e.g. one written by an older fragment layout.
    pub fn adopt_persisted(&mut self) -> Result<bool, BpannError> {
        self.reset();
        let Some(tree) = Tree::open_persisted(&self.index_dir)? else {
            return Ok(false);
        };
        self.indexed_rows = tree.header.indexed_rows;
        self.tree = Some(tree);
        Ok(true)
    }

    pub fn reset(&mut self) {
        self.tree = None;
        self.indexed_rows = 0;
        self.morph = Morph::default();
    }

    /// Re-partition the indexed rows under the current coordinates, in place,
    /// during the next inserts.
    pub fn start_morph(&mut self) {
        if let Some(tree) = &self.tree {
            self.morph.schedule(tree);
        }
    }

    pub fn morphing(&self) -> bool {
        self.morph.active()
    }

    /// Bulk-build an empty tree from up to [`bulk_build_max_rows`] unindexed rows.
    fn bulk_build_from(&mut self, train_x: &MmapColumnStore, num_dim: usize, scale: (bool, &[f64]), end: usize) -> Result<(), BpannError> {
        let end = end.min(self.indexed_rows + bulk_build_max_rows(num_dim));
        let rows: Vec<u32> = (self.indexed_rows as u32..end as u32).collect();
        let (mut v, mut vs) = (Vec::with_capacity(num_dim), Vec::with_capacity(rows.len() * num_dim));
        for &row in &rows {
            bpann_row_to_f32(train_x.mmap_row_slice(row as usize)?, scale.0, scale.1, &mut v);
            vs.extend_from_slice(&v);
        }
        self.tree = Some(bulk_build(&rows, &mut vs, num_dim, self.index_dir.clone())?);
        self.indexed_rows = end;
        Ok(())
    }

    /// Insert rows `indexed_rows..end` into the tree, one at a time, advancing a
    /// re-partition in progress after each.
    pub fn ensure_sync_for_backend(
        &mut self,
        train_x: &MmapColumnStore,
        num_dim: usize,
        scale_x: bool,
        x_scale: &[f64],
        end: usize,
    ) -> Result<(), BpannError> {
        let mut v = Vec::with_capacity(num_dim);
        if let Some(tree) = self.tree.as_mut().filter(|t| !t.rows_cached()) {
            tree.fill_blocks(|row, buf| {
                bpann_row_to_f32(train_x.mmap_row_slice(row as usize)?, scale_x, x_scale, buf);
                Ok(())
            })?;
        }
        if self.tree.is_none() && end.saturating_sub(self.indexed_rows) >= BULK_BUILD_MIN_ROWS {
            self.bulk_build_from(train_x, num_dim, (scale_x, x_scale), end)?;
        }
        for row in self.indexed_rows..end {
            bpann_row_to_f32(train_x.mmap_row_slice(row)?, scale_x, x_scale, &mut v);
            match self.tree.as_mut() {
                Some(tree) => {
                    tree.insert_row(row as u32, &v)?;
                    self.morph.advance(tree);
                }
                None => self.tree = Some(Tree::new_tree(row as u32, &v, num_dim, self.index_dir.clone())?),
            }
            self.indexed_rows = row + 1;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn persist_to_disk_for_backend(
        &mut self,
        train_x: &MmapColumnStore,
        num_dim: usize,
        scale_x: bool,
        x_scale: &[f64],
        work_dir: &std::path::Path,
        num_metrics: usize,
    ) -> Result<(), BpannError> {
        self.ensure_sync_for_backend(train_x, num_dim, scale_x, x_scale, train_x.nrows)?;
        if let Some(tree) = &self.tree {
            tree.persist()?;
        }
        obs::bpann_write_metadata(work_dir, train_x.nrows, num_dim, num_metrics, scale_x, self.indexed_rows)?;
        obs::write_indexed_rows(work_dir, self.indexed_rows)?;
        Ok(())
    }

    pub fn needs_disk_rewrite(&self, index_dirty: bool, nrows: usize) -> bool {
        if index_dirty {
            return true;
        }
        let on_disk = match on_disk_indexed_rows(&self.index_dir) {
            Ok(on_disk) => on_disk,
            Err(_) => return true,
        };
        if on_disk != nrows {
            return true;
        }
        match &self.tree {
            Some(tree) => !tree.on_disk_index_matches().unwrap_or(false),
            None => on_disk != 0 || nrows != 0,
        }
    }

    pub fn search_candidates(
        &self,
        query_f32: &[f32],
        k: usize,
        store: Option<&MmapSearchStore<'_>>,
    ) -> Result<Vec<(u32, f32)>, BpannError> {
        match (&self.tree, store) {
            (Some(tree), Some(store)) => search_tree(tree, query_f32, k, leaf_budget(self.indexed_rows), store),
            _ => Ok(Vec::new()),
        }
    }

    /// Bytes of the persisted index files.
    pub fn index_memory_bytes(&self) -> usize {
        self.tree.as_ref().map_or(0, |_| Tree::persisted_bytes(&self.index_dir))
    }
}

fn on_disk_indexed_rows(index_dir: &std::path::Path) -> Result<usize, BpannError> {
    let header_path = index_dir.join("header.json");
    if !header_path.exists() {
        return Ok(0);
    }
    let text = fs::read_to_string(&header_path)
        .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
    let header: IndexHeader = serde_json::from_str(&text)
        .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
    Ok(header.indexed_rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::tree::{TREE_FANOUT, TREE_LEAF_CAPACITY};
    use crate::index::BpannIndex;
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

    fn depth(tree: &Tree, page_id: u32) -> usize {
        if tree.store.is_leaf(page_id) {
            return 1;
        }
        1 + tree.store.ids(page_id).iter().map(|&c| depth(tree, c)).max().unwrap()
    }

    fn tree(idx: &IncrementalIndex) -> &Tree {
        idx.tree.as_ref().expect("tree")
    }

    #[test]
    fn tree_holds_every_row_once_with_bounded_pages() {
        let dir = TempDir::new().unwrap();
        let store = store_with_rows(&dir, 5000, 3, 1);
        let mut idx = IncrementalIndex::new(dir.path().join("index"));
        for end in [1, 2, 700, 701, 5000] {
            idx.ensure_sync_for_backend(&store, 3, false, &[1.0; 3], end).unwrap();
        }
        let t = tree(&idx);
        assert_eq!(idx.indexed_rows, 5000);
        assert_eq!(t.header.indexed_rows, 5000);
        assert_eq!(t.leaf_row_ids(), (0..5000).collect::<Vec<u32>>());
        for page in 0..t.num_pages() as u32 {
            let cap = if t.store.is_leaf(page) { TREE_LEAF_CAPACITY } else { TREE_FANOUT };
            assert!(t.store.len(page) <= cap);
        }
        let d = depth(t, t.header.root_page_id);
        assert!((3..=6).contains(&d), "depth {d}");
    }

    #[test]
    fn tree_search_finds_exact_neighbor_of_stored_row() {
        let dir = TempDir::new().unwrap();
        let store = store_with_rows(&dir, 3000, 2, 2);
        let mut idx = IncrementalIndex::new(dir.path().join("index"));
        idx.ensure_sync_for_backend(&store, 2, false, &[1.0; 2], 3000).unwrap();
        let s = MmapSearchStore {
            train_x: &store,
            scale_x: false,
            x_scale: &[1.0, 1.0],
        };
        for row in [0usize, 1234, 2999] {
            let q: Vec<f32> = store.mmap_row_slice(row).unwrap().iter().map(|&v| v as f32).collect();
            let got = idx.search_candidates(&q, 3, Some(&s)).unwrap();
            assert_eq!(got.len(), 3);
            assert_eq!(got[0], (row as u32, 0.0));
            assert!(got[0].1 <= got[1].1 && got[1].1 <= got[2].1);
        }
        assert!(idx.search_candidates(&[0.0, 0.0], 3, None).unwrap().is_empty());
    }

    fn rows_below(t: &Tree, page_id: u32, out: &mut Vec<u32>) {
        if t.store.is_leaf(page_id) {
            out.extend(t.store.ids(page_id));
        } else {
            t.store.ids(page_id).iter().for_each(|&c| rows_below(t, c, out));
        }
    }

    fn assert_radii_bound_rows(idx: &IncrementalIndex, store: &MmapColumnStore, scale: &[f64]) {
        let t = tree(idx);
        let d = scale.len();
        for page in (0..t.num_pages() as u32).filter(|&p| !t.store.is_leaf(p)) {
            for (c, &child) in t.store.block(page).chunks_exact(d).zip(t.store.ids(page)) {
                let mut rows = Vec::new();
                rows_below(t, child, &mut rows);
                for row in rows {
                    let x: Vec<f32> =
                        store.mmap_row_slice(row as usize).unwrap().iter().zip(scale).map(|(&v, s)| (v * s) as f32).collect();
                    let dd = crate::index::tree_counts::dist(&x, c);
                    assert!(dd <= t.store.radius(child) * 1.0001 + 1e-6, "row {row} at {dd} outside page {child}");
                }
            }
        }
    }

    #[test]
    fn radii_bound_every_row_and_unbounded_search_is_exact() {
        let dir = TempDir::new().unwrap();
        let store = store_with_rows(&dir, 4000, 3, 5);
        let mut idx = IncrementalIndex::new(dir.path().join("index"));
        idx.ensure_sync_for_backend(&store, 3, false, &[1.0; 3], 4000).unwrap();
        assert_radii_bound_rows(&idx, &store, &[1.0; 3]);
        tree(&idx).assert_counts_consistent();
        let s = MmapSearchStore {
            train_x: &store,
            scale_x: false,
            x_scale: &[1.0; 3],
        };
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        for _ in 0..20 {
            let q: Vec<f32> = (0..3).map(|_| rng.gen::<f32>()).collect();
            let got = search_tree(tree(&idx), &q, 7, usize::MAX, &s).unwrap();
            let mut all: Vec<(u32, f32)> = (0..4000u32)
                .map(|r| {
                    let x: Vec<f32> = store.mmap_row_slice(r as usize).unwrap().iter().map(|&v| v as f32).collect();
                    (r, crate::distance::l2_sq_f32(&q, &x))
                })
                .collect();
            all.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
            assert_eq!(got.iter().map(|g| g.0).collect::<Vec<_>>(), all[..7].iter().map(|a| a.0).collect::<Vec<_>>());
        }
        idx.rescale_coords(&[2.0, 0.5, 1.0]);
        assert_radii_bound_rows(&idx, &store, &[2.0, 0.5, 1.0]);
    }

    #[test]
    fn rescaled_blocks_match_blocks_rebuilt_from_rows() {
        let dir = TempDir::new().unwrap();
        let store = store_with_rows(&dir, 2000, 2, 4);
        let mut idx = IncrementalIndex::new(dir.path().join("index"));
        idx.ensure_sync_for_backend(&store, 2, false, &[1.0; 2], 2000).unwrap();
        let ratio = [2.0, 0.5];
        idx.rescale_coords(&ratio);
        let mut rebuilt = tree(&idx).try_clone().unwrap();
        rebuilt
            .fill_blocks(|row, buf| {
                let x = store.mmap_row_slice(row as usize)?;
                *buf = x.iter().zip(ratio).map(|(&v, r)| (v * r) as f32).collect();
                Ok(())
            })
            .unwrap();
        let t = tree(&idx);
        for page in (0..t.num_pages() as u32).filter(|&p| t.store.is_leaf(p)) {
            let (a, b) = (t.store.block(page), rebuilt.store.block(page));
            assert_eq!(a.len(), b.len());
            assert!(a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-5));
        }
    }

    #[test]
    fn persisted_tree_round_trips_and_legacy_layout_is_rejected() {
        let dir = TempDir::new().unwrap();
        let store = store_with_rows(&dir, 900, 2, 3);
        let mut idx = IncrementalIndex::new(dir.path().join("index"));
        assert!(!idx.needs_disk_rewrite(false, 0));
        idx.persist_to_disk_for_backend(&store, 2, false, &[1.0; 2], dir.path(), 1).unwrap();
        assert!(!idx.needs_disk_rewrite(false, 900));
        assert!(idx.needs_disk_rewrite(true, 900));
        assert!(idx.index_memory_bytes() > 0);
        assert_eq!(on_disk_indexed_rows(&idx.index_dir).unwrap(), 900);
        let mut reopened = IncrementalIndex::new(idx.index_dir.clone());
        assert!(reopened.adopt_persisted().unwrap());
        assert_eq!(reopened.indexed_rows, 900);
        assert!(!tree(&reopened).rows_cached());
        let s = MmapSearchStore {
            train_x: &store,
            scale_x: false,
            x_scale: &[1.0, 1.0],
        };
        let q = [0.3f32, 0.6];
        assert_eq!(reopened.search_candidates(&q, 5, Some(&s)).unwrap(), idx.search_candidates(&q, 5, Some(&s)).unwrap());
        reopened.ensure_sync_for_backend(&store, 2, false, &[1.0; 2], 900).unwrap();
        tree(&reopened).assert_counts_consistent();
        for q in [[0.1f32, 0.9], [0.5, 0.5]] {
            assert_eq!(
                reopened.search_candidates(&q, 5, Some(&s)).unwrap(),
                idx.search_candidates(&q, 5, Some(&s)).unwrap()
            );
        }
        let legacy_dir = dir.path().join("legacy");
        BpannIndex::build_from_vectors(&(0..100).map(|i| vec![i as f32, 0.0]).collect::<Vec<_>>(), 2, 8, 0, legacy_dir.clone())
            .unwrap();
        let mut legacy = IncrementalIndex::new(legacy_dir);
        assert!(!legacy.adopt_persisted().unwrap());
        assert!(legacy.tree.is_none() && legacy.indexed_rows == 0);
    }

    #[test]
    fn clone_is_independent_and_bulk_builds_are_capped() {
        let dir = TempDir::new().unwrap();
        let store = store_with_rows(&dir, 70_000, 1, 8);
        let mut idx = IncrementalIndex::new(dir.path().join("index"));
        idx.ensure_sync_for_backend(&store, 1, false, &[1.0], 69_000).unwrap();
        let copy = idx.clone();
        idx.ensure_sync_for_backend(&store, 1, false, &[1.0], 70_000).unwrap();
        assert_eq!((copy.indexed_rows, tree(&copy).header.indexed_rows), (69_000, 69_000));
        assert_eq!(tree(&idx).leaf_row_ids().len(), 70_000);
        assert_eq!(tree(&copy).leaf_row_ids(), (0..69_000).collect::<Vec<u32>>());
        tree(&copy).assert_counts_consistent();
    }
}
