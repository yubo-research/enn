use std::fs;
use std::path::PathBuf;

use crate::distance::bpann_row_to_f32;
use crate::error::BpannError;
use crate::index::build::{BpannIndex, IndexHeader};
use crate::index::search::MmapSearchStore;
use crate::index::tree::{insert_row, new_tree, TreeCounts};
use crate::index::tree_bulk::{bulk_build, BULK_BUILD_MIN_ROWS};
use crate::index::tree_morph::Morph;
use crate::index::tree_search::{leaf_budget, search_tree};
use crate::mmap_store::MmapColumnStore;
use crate::observation as obs;

/// The live BPANN index: at most one incremental tree (see [`crate::index::tree`]).
///
/// `indices` holds that tree (empty until the first row is indexed); rows
/// `indexed_rows..` of the store are not yet in it. After a metric change, `morph`
/// re-partitions the tree in place by moving row and page references below one
/// internal page per inserted row ([`crate::index::tree_morph`]).
#[derive(Clone)]
pub struct IncrementalIndex {
    pub indices: Vec<BpannIndex>,
    pub indexed_rows: usize,
    pub index_dir: PathBuf,
    pub(crate) counts: TreeCounts,
    pub(crate) morph: Morph,
}

impl IncrementalIndex {
    pub fn new(index_dir: PathBuf) -> Self {
        Self {
            indices: Vec::new(),
            indexed_rows: 0,
            index_dir,
            counts: TreeCounts::default(),
            morph: Morph::default(),
        }
    }

    /// Adopt a tree read from disk. Returns false (and stays empty) if it is not
    /// an incremental row-id tree, e.g. one written by an older fragment layout.
    pub fn adopt_persisted(&mut self, index: BpannIndex) -> bool {
        self.reset();
        let Some(counts) = TreeCounts::from_index(&index) else {
            return false;
        };
        self.indexed_rows = index.header.indexed_rows;
        self.counts = counts;
        self.indices = vec![index];
        true
    }

    pub fn reset(&mut self) {
        self.indices.clear();
        self.indexed_rows = 0;
        self.counts = TreeCounts::default();
        self.morph = Morph::default();
    }

    /// Re-partition the indexed rows under the current coordinates, in place,
    /// during the next inserts.
    pub fn start_morph(&mut self) {
        if let Some(index) = self.indices.first() {
            self.morph.schedule(index);
        }
    }

    pub fn morphing(&self) -> bool {
        self.morph.active()
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
        if let Some(index) = self.indices.first().filter(|_| !self.counts.rows_cached()) {
            self.counts.fill_blocks(index, |row, buf| {
                bpann_row_to_f32(train_x.mmap_row_slice(row as usize)?, scale_x, x_scale, buf);
                Ok(())
            })?;
        }
        if self.indices.is_empty() && end.saturating_sub(self.indexed_rows) >= BULK_BUILD_MIN_ROWS {
            let rows: Vec<u32> = (self.indexed_rows as u32..end as u32).collect();
            let mut vs = Vec::with_capacity(rows.len() * num_dim);
            for &row in &rows {
                bpann_row_to_f32(train_x.mmap_row_slice(row as usize)?, scale_x, x_scale, &mut v);
                vs.extend_from_slice(&v);
            }
            let (index, counts) = bulk_build(&rows, &mut vs, num_dim, self.index_dir.clone())?;
            self.indices.push(index);
            self.counts = counts;
            self.indexed_rows = end;
        }
        for row in self.indexed_rows..end {
            bpann_row_to_f32(train_x.mmap_row_slice(row)?, scale_x, x_scale, &mut v);
            match self.indices.first_mut() {
                Some(index) => {
                    insert_row(index, &mut self.counts, row as u32, &v);
                    self.morph.advance(index, &mut self.counts);
                }
                None => {
                    let (index, counts) = new_tree(row as u32, &v, num_dim, self.index_dir.clone())?;
                    self.indices.push(index);
                    self.counts = counts;
                }
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
        if let Some(index) = self.indices.first_mut() {
            self.counts.write_page_centroids(index);
            index.persist()?;
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
        match self.indices.first() {
            Some(index) => !index.on_disk_index_matches().unwrap_or(false),
            None => on_disk != 0 || nrows != 0,
        }
    }

    pub fn search_candidates(
        &self,
        query_f32: &[f32],
        k: usize,
        store: Option<&MmapSearchStore<'_>>,
    ) -> Result<Vec<(u32, f32)>, BpannError> {
        match (self.indices.first(), store) {
            (Some(index), Some(store)) => {
                search_tree(index, &self.counts, query_f32, k, leaf_budget(self.indexed_rows), store)
            }
            _ => Ok(Vec::new()),
        }
    }

    pub fn index_memory_bytes(&self) -> usize {
        self.indices.iter().map(|i| i.index_memory_bytes()).sum()
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
    use crate::index::page::Page;
    use crate::index::tree::{TREE_FANOUT, TREE_LEAF_CAPACITY};
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

    fn depth(index: &BpannIndex, page_id: u32) -> usize {
        match index.page_by_id(page_id).unwrap() {
            Page::Leaf { .. } => 1,
            Page::Internal { child_page_ids, .. } => {
                1 + child_page_ids.iter().map(|&c| depth(index, c)).max().unwrap()
            }
        }
    }

    #[test]
    fn tree_holds_every_row_once_with_bounded_pages() {
        let dir = TempDir::new().unwrap();
        let store = store_with_rows(&dir, 5000, 3, 1);
        let mut idx = IncrementalIndex::new(dir.path().join("index"));
        for end in [1, 2, 700, 701, 5000] {
            idx.ensure_sync_for_backend(&store, 3, false, &[1.0; 3], end).unwrap();
        }
        let index = &idx.indices[0];
        assert_eq!(idx.indexed_rows, 5000);
        assert_eq!(index.header.indexed_rows, 5000);
        assert_eq!(index.leaf_row_ids(), (0..5000).collect::<Vec<u32>>());
        for page in &index.pages {
            match page {
                Page::Leaf { row_ids, .. } => assert!(row_ids.len() <= TREE_LEAF_CAPACITY),
                Page::Internal { child_page_ids, .. } => assert!(child_page_ids.len() <= TREE_FANOUT),
            }
        }
        let d = depth(index, index.header.root_page_id);
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

    fn rows_below(index: &BpannIndex, page_id: u32, out: &mut Vec<u32>) {
        match index.page_by_id(page_id).unwrap() {
            Page::Leaf { row_ids, .. } => out.extend(row_ids),
            Page::Internal { child_page_ids, .. } => {
                child_page_ids.iter().for_each(|&c| rows_below(index, c, out));
            }
        }
    }

    fn assert_radii_bound_rows(idx: &IncrementalIndex, store: &MmapColumnStore, scale: &[f64]) {
        let mut index = idx.indices[0].clone();
        idx.counts.write_page_centroids(&mut index);
        let index = &index;
        for page in &index.pages {
            let Page::Internal { centroids, child_page_ids, .. } = page else { continue };
            for (c, &child) in centroids.iter().zip(child_page_ids) {
                let mut rows = Vec::new();
                rows_below(index, child, &mut rows);
                for row in rows {
                    let x: Vec<f32> =
                        store.mmap_row_slice(row as usize).unwrap().iter().zip(scale).map(|(&v, s)| (v * s) as f32).collect();
                    let d = crate::index::tree_counts::dist(&x, c);
                    assert!(d <= idx.counts.radius(child) * 1.0001 + 1e-6, "row {row} at {d} outside page {child}");
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
        let recount = TreeCounts::from_index(&idx.indices[0]).expect("valid tree");
        for page in &idx.indices[0].pages {
            assert_eq!(idx.counts.count(page.page_id()), recount.count(page.page_id()));
        }
        let s = MmapSearchStore {
            train_x: &store,
            scale_x: false,
            x_scale: &[1.0; 3],
        };
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        for _ in 0..20 {
            let q: Vec<f32> = (0..3).map(|_| rng.gen::<f32>()).collect();
            let got = search_tree(&idx.indices[0], &idx.counts, &q, 7, usize::MAX, &s).unwrap();
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
        let mut rebuilt = TreeCounts::default();
        rebuilt
            .fill_blocks(&idx.indices[0], |row, buf| {
                let x = store.mmap_row_slice(row as usize)?;
                *buf = x.iter().zip(ratio).map(|(&v, r)| (v * r) as f32).collect();
                Ok(())
            })
            .unwrap();
        for page in &idx.indices[0].pages {
            let (a, b) = (idx.counts.block(page.page_id()), rebuilt.block(page.page_id()));
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
        let opened = BpannIndex::open(idx.index_dir.clone()).unwrap();
        let mut reopened = IncrementalIndex::new(idx.index_dir.clone());
        assert!(reopened.adopt_persisted(opened));
        assert_eq!(reopened.indexed_rows, 900);
        reopened.ensure_sync_for_backend(&store, 2, false, &[1.0; 2], 900).unwrap();
        let s = MmapSearchStore {
            train_x: &store,
            scale_x: false,
            x_scale: &[1.0, 1.0],
        };
        for q in [[0.1f32, 0.9], [0.5, 0.5]] {
            assert_eq!(
                reopened.search_candidates(&q, 5, Some(&s)).unwrap(),
                idx.search_candidates(&q, 5, Some(&s)).unwrap()
            );
        }
        let legacy = BpannIndex::build_from_vectors(
            &(0..100).map(|i| vec![i as f32, 0.0]).collect::<Vec<_>>(),
            2,
            8,
            0,
            dir.path().join("legacy"),
        )
        .unwrap();
        assert!(!reopened.adopt_persisted(legacy));
        assert!(reopened.indices.is_empty() && reopened.indexed_rows == 0);
    }
}
