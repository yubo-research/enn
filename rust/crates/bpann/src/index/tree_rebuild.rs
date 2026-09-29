//! A re-partition of the tree (after a metric change) that never stalls insertion.
//!
//! [`Rebuild::start`] copies every indexed row's coordinates (already in the new
//! metric) out of the old tree's leaf blocks and bulk-builds a fresh tree over them
//! on a background thread, while the old tree keeps taking rows and serving queries.
//! Once `rows / REBUILD_SPEEDUP` rows have been added, [`Rebuild::advance`] takes the
//! finished tree, inserts the rows added since, [`CATCHUP_PER_ROW`] per added row,
//! and returns it. The swap happens at a row count, not when the thread finishes, so
//! results do not depend on timing.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::distance::bpann_row_to_f32;
use crate::error::BpannError;
use crate::index::build::BpannIndex;
use crate::index::page::Page;
use crate::index::tree::{insert_row, TreeCounts};
use crate::index::tree_bulk::bulk_build;
use crate::mmap_store::MmapColumnStore;

/// The new tree is swapped in after this fraction (inverse) of the indexed rows is added.
pub const REBUILD_SPEEDUP: usize = 16;
/// Rows added during the build are inserted into the new tree this many per added row.
pub const CATCHUP_PER_ROW: usize = 4;

type Built = Result<(BpannIndex, TreeCounts), BpannError>;

/// A re-partition in progress; clones share the background build.
#[derive(Clone)]
pub struct Rebuild {
    rows: usize,
    added: usize,
    job: Arc<Mutex<Option<JoinHandle<Built>>>>,
    built: Option<(BpannIndex, TreeCounts, usize)>,
}

impl Rebuild {
    /// Start re-partitioning the `rows` rows held by `index` (blocks in `counts`).
    pub fn start(index: &BpannIndex, counts: &TreeCounts, rows: usize, index_dir: PathBuf) -> Self {
        let num_dim = index.header.num_dim;
        let mut ids = Vec::with_capacity(rows);
        let mut vs = Vec::with_capacity(rows * num_dim.max(1));
        for page in &index.pages {
            if let Page::Leaf { page_id, row_ids, .. } = page {
                ids.extend_from_slice(row_ids);
                vs.extend_from_slice(counts.block(*page_id));
            }
        }
        let job = std::thread::spawn(move || bulk_build(&ids, &mut vs, num_dim, index_dir));
        Self {
            rows,
            added: 0,
            job: Arc::new(Mutex::new(Some(job))),
            built: None,
        }
    }

    /// Whether another clone took the background build, so this one can never finish.
    pub fn orphaned(&self) -> bool {
        self.built.is_none() && self.job.lock().map_or(true, |job| job.is_none())
    }

    /// Account for `added` new rows (the store now holds `indexed_rows` indexed
    /// rows); return the new tree once it is due and holds all of them.
    pub fn advance(
        &mut self,
        added: usize,
        store: &MmapColumnStore,
        scale: (bool, &[f64]),
        indexed_rows: usize,
    ) -> Result<Option<(BpannIndex, TreeCounts)>, BpannError> {
        self.added += added;
        if self.built.is_none() {
            if self.added < self.rows.div_ceil(REBUILD_SPEEDUP) {
                return Ok(None);
            }
            let job = self.job.lock().map_err(|_| BpannError::InvalidParameter("rebuild lock poisoned".into()))?.take();
            let Some(job) = job else { return Ok(None) };
            let (index, counts) = job.join().map_err(|_| BpannError::InvalidParameter("rebuild thread panicked".into()))??;
            self.built = Some((index, counts, self.rows));
        }
        let Some((index, counts, next)) = self.built.as_mut() else { return Ok(None) };
        let stop = (*next + CATCHUP_PER_ROW * added).min(indexed_rows);
        let mut v = Vec::new();
        for row in *next..stop {
            bpann_row_to_f32(store.mmap_row_slice(row)?, scale.0, scale.1, &mut v);
            insert_row(index, counts, row as u32, &v);
        }
        *next = stop;
        if stop < indexed_rows {
            return Ok(None);
        }
        Ok(self.built.take().map(|(index, counts, _)| (index, counts)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::search::MmapSearchStore;
    use crate::index::tree::TREE_FANOUT;
    use crate::index::tree_search::search_tree;
    use crate::index::IncrementalIndex;
    use rand::{Rng, SeedableRng};

    #[test]
    fn rebuild_whose_build_went_to_a_discarded_clone_still_swaps_in() {
        let (n0, d) = (70_000usize, 2usize);
        let dir = tempfile::TempDir::new().unwrap();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(6);
        let x = ndarray::Array2::from_shape_fn((n0 + n0 / 4, d), |_| rng.gen::<f64>());
        let mut store = MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), d, None).unwrap();
        store.mmap_append(&x.view()).unwrap();
        let mut idx = IncrementalIndex::new(dir.path().join("index"));
        idx.ensure_sync_for_backend(&store, d, false, &[1.0; 2], n0).unwrap();
        idx.start_rebuild();
        let mut m = n0 + n0 / 16;
        let mut clone = idx.clone();
        clone.ensure_sync_for_backend(&store, d, false, &[1.0; 2], m).unwrap();
        drop(clone);
        idx.ensure_sync_for_backend(&store, d, false, &[1.0; 2], m).unwrap();
        while idx.rebuilding() {
            m += 500;
            assert!(m <= n0 + n0 / 4, "rebuild stuck at {m} rows");
            idx.ensure_sync_for_backend(&store, d, false, &[1.0; 2], m).unwrap();
        }
        let mut ids = idx.indices[0].leaf_row_ids();
        ids.sort_unstable();
        assert_eq!(ids, (0..m as u32).collect::<Vec<_>>());
    }

    #[test]
    fn rebuild_swaps_in_within_an_eighth_of_the_rows_and_holds_every_row() {
        let (n0, d) = (140_000usize, 3usize);
        let dir = tempfile::TempDir::new().unwrap();
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(4);
        let x = ndarray::Array2::from_shape_fn((n0 + n0 / 8, d), |_| rng.gen::<f64>());
        let mut store = MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), d, None).unwrap();
        store.mmap_append(&x.view()).unwrap();
        let mut idx = IncrementalIndex::new(dir.path().join("index"));
        idx.ensure_sync_for_backend(&store, d, false, &[1.0; 3], n0).unwrap();
        idx.start_rebuild();
        let mut m = n0;
        while idx.rebuilding() {
            m += 500;
            assert!(m <= n0 + n0 / 8, "rebuild still running at {m} rows");
            idx.ensure_sync_for_backend(&store, d, false, &[1.0; 3], m).unwrap();
        }
        let index = &idx.indices[0];
        let mut ids = index.leaf_row_ids();
        ids.sort_unstable();
        assert_eq!(ids, (0..m as u32).collect::<Vec<_>>());
        let recount = TreeCounts::from_index(index).expect("valid tree");
        for page in &index.pages {
            assert_eq!(idx.counts.count(page.page_id()), recount.count(page.page_id()));
            match page {
                Page::Leaf { row_ids, .. } => assert!((1..=64).contains(&row_ids.len())),
                Page::Internal { child_page_ids, .. } => assert!((2..=TREE_FANOUT).contains(&child_page_ids.len())),
            }
        }
        let s = MmapSearchStore {
            train_x: &store,
            scale_x: false,
            x_scale: &[1.0; 3],
        };
        for q in [[0.1f32, 0.5, 0.9], [0.7, 0.2, 0.4]] {
            let got = search_tree(index, &idx.counts, &q, 5, usize::MAX, &s).unwrap();
            let mut all: Vec<(f32, u32)> = (0..m)
                .map(|r| {
                    let x: Vec<f32> = store.mmap_row_slice(r).unwrap().iter().map(|&v| v as f32).collect();
                    (crate::distance::l2_sq_f32(&q, &x), r as u32)
                })
                .collect();
            all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            assert_eq!(got.iter().map(|g| g.0).collect::<Vec<_>>(), all[..5].iter().map(|a| a.1).collect::<Vec<_>>());
        }
    }
}
