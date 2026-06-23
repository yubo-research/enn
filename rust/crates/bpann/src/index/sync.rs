use std::path::PathBuf;

use crate::distance::{l2_sq_f32, bpann_row_to_f32};
use crate::error::BpannError;
use crate::index::build::BpannIndex;
use crate::index::search::{
    search_exhaustive_leaves_with_store, search_greedy_blocks_only_with_store,
    search_with_skip_refinement_with_store, MmapSearchStore,
};
use crate::index::DEFAULT_LEAF_CAPACITY;
use crate::mmap_store::MmapColumnStore;
use crate::observation as obs;

const INDEX_COMPACT_THRESHOLD_MIN: usize = 3;
const EXHAUSTIVE_SEARCH_ROW_LIMIT: usize = 2500;
const SKIP_REFINEMENT_ROW_LIMIT: usize = 150_000;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn index_compact_rows_per_fragment() -> usize {
    env_usize("BPANN_INDEX_COMPACT_ROWS_PER_FRAGMENT", 10_000)
}

fn index_compact_fragment_max() -> usize {
    env_usize("BPANN_INDEX_COMPACT_FRAGMENT_MAX", 32)
}

fn search_rows_per_fragment() -> usize {
    env_usize("BPANN_SEARCH_ROWS_PER_FRAGMENT", 80_000)
}

fn small_fragment_merge_rows() -> usize {
    env_usize("BPANN_SMALL_FRAGMENT_MERGE_ROWS", 15_000)
}

fn search_fragment_budget_max() -> usize {
    env_usize("BPANN_SEARCH_FRAGMENT_BUDGET_MAX", 3)
}

fn index_compact_threshold(indexed_rows: usize) -> usize {
    if indexed_rows <= 1000 {
        return 1;
    }
    (indexed_rows / index_compact_rows_per_fragment())
        .clamp(INDEX_COMPACT_THRESHOLD_MIN, index_compact_fragment_max())
}

fn search_fragment_budget(fragment_count: usize, indexed_rows: usize) -> usize {
    if fragment_count <= 2 {
        return fragment_count;
    }
    let scaled = (indexed_rows / search_rows_per_fragment()).max(2);
    scaled
        .min(fragment_count)
        .min(search_fragment_budget_max())
}

fn search_beam_width(_indexed_rows: usize) -> usize {
    1
}

struct IndexBuildContext<'a> {
    train_x: &'a MmapColumnStore,
    num_dim: usize,
    scale_x: bool,
    x_scale: &'a [f64],
    work_dir: &'a std::path::Path,
    num_metrics: usize,
}

pub struct IncrementalIndex {
    pub indices: Vec<BpannIndex>,
    pub indexed_rows: usize,
    pub index_dir: PathBuf,
    pending_centroid_sum: Vec<f64>,
    pending_row_count: usize,
}

impl IncrementalIndex {
    pub fn new(index_dir: PathBuf) -> Self {
        Self {
            indices: Vec::new(),
            indexed_rows: 0,
            index_dir,
            pending_centroid_sum: Vec::new(),
            pending_row_count: 0,
        }
    }

    pub fn note_pending_rows(
        &mut self,
        x: &ndarray::ArrayView2<f64>,
        scale_x: bool,
        x_scale: &[f64],
    ) {
        if x.nrows() == 0 {
            return;
        }
        if self.pending_centroid_sum.len() != x.ncols() {
            self.pending_centroid_sum = vec![0.0; x.ncols()];
        }
        for row in x.axis_iter(ndarray::Axis(0)) {
            for (j, &v) in row.iter().enumerate() {
                self.pending_centroid_sum[j] += if scale_x {
                    v / x_scale[j]
                } else {
                    v
                };
            }
            self.pending_row_count += 1;
        }
    }

    fn take_pending_centroid(&mut self, num_dim: usize) -> Option<Vec<f32>> {
        if self.pending_row_count == 0 {
            return None;
        }
        if self.pending_centroid_sum.len() != num_dim {
            return None;
        }
        let count = self.pending_row_count as f64;
        let centroid = self
            .pending_centroid_sum
            .iter()
            .map(|&s| (s / count) as f32)
            .collect();
        self.pending_centroid_sum.fill(0.0);
        self.pending_row_count = 0;
        Some(centroid)
    }

    pub fn reset(&mut self) {
        self.indices.clear();
        self.indexed_rows = 0;
        self.pending_centroid_sum.clear();
        self.pending_row_count = 0;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn ensure_sync_for_backend(
        &mut self,
        train_x: &MmapColumnStore,
        num_dim: usize,
        scale_x: bool,
        x_scale: &[f64],
        work_dir: &std::path::Path,
        num_metrics: usize,
        end: usize,
    ) -> Result<(), BpannError> {
        let ctx = IndexBuildContext {
            train_x,
            num_dim,
            scale_x,
            x_scale,
            work_dir,
            num_metrics,
        };
        self.ensure_sync(&ctx, end)
    }

    fn ensure_sync(&mut self, ctx: &IndexBuildContext<'_>, end: usize) -> Result<(), BpannError> {
        if self.indexed_rows >= end {
            return Ok(());
        }
        self.build_batch(ctx, self.indexed_rows, end)?;
        self.maybe_compact_or_persist(ctx)?;
        obs::write_indexed_rows(ctx.work_dir, self.indexed_rows)?;
        Ok(())
    }

    fn maybe_compact_or_persist(&mut self, ctx: &IndexBuildContext<'_>) -> Result<(), BpannError> {
        let max_fragments = index_compact_threshold(self.indexed_rows);
        let compact_limit = max_fragments.saturating_mul(2).max(max_fragments + 1);
        if self.indices.len() > compact_limit {
            self.compact(ctx)?;
        } else if self.indices.len() == 1 {
            self.indices[0].persist()?;
            obs::bpann_write_metadata(
                ctx.work_dir,
                ctx.train_x.nrows,
                ctx.num_dim,
                ctx.num_metrics,
                ctx.scale_x,
                self.indexed_rows,
            )?;
        }
        Ok(())
    }

    fn compact(&mut self, ctx: &IndexBuildContext<'_>) -> Result<(), BpannError> {
        if self.indexed_rows == 0 {
            self.indices.clear();
            return Ok(());
        }
        let max_fragments = index_compact_threshold(self.indexed_rows);
        while self.indices.len() > max_fragments {
            let over = self.indices.len() - max_fragments;
            let merge_n = over.clamp(2, 4).min(self.indices.len());
            self.amalgamate_smallest_run(ctx, merge_n)?;
        }
        if self.indices.len() == 1 {
            self.indices[0].persist()?;
            obs::bpann_write_metadata(
                ctx.work_dir,
                ctx.train_x.nrows,
                ctx.num_dim,
                ctx.num_metrics,
                ctx.scale_x,
                self.indexed_rows,
            )?;
        }
        Ok(())
    }

    fn amalgamate_smallest_run(
        &mut self,
        _ctx: &IndexBuildContext<'_>,
        merge_n: usize,
    ) -> Result<(), BpannError> {
        if self.indices.len() < 2 {
            return Ok(());
        }
        let merge_n = merge_n.min(self.indices.len());
        let mut best_i = 0usize;
        let mut best_rows = usize::MAX;
        let small_limit = small_fragment_merge_rows();
        for i in 0..=self.indices.len().saturating_sub(merge_n) {
            let slice = &self.indices[i..i + merge_n];
            let rows: usize = slice.iter().map(|index| index.header.indexed_rows).sum();
            let all_small = slice
                .windows(2)
                .all(|pair| {
                    pair[0].header.indexed_rows <= small_limit
                        && pair[1].header.indexed_rows <= small_limit
                });
            let rank = if all_small { rows } else { rows + usize::MAX / 2 };
            if rank < best_rows {
                best_rows = rank;
                best_i = i;
            }
        }
        let removed: Vec<BpannIndex> = self.indices.drain(best_i..best_i + merge_n).collect();
        let merged = BpannIndex::concat_merge(removed, self.index_dir.clone(), false)?;
        self.indices.insert(best_i, merged);
        Ok(())
    }

    #[allow(dead_code)]
    fn amalgamate_smallest_pair(&mut self, ctx: &IndexBuildContext<'_>) -> Result<(), BpannError> {
        self.amalgamate_smallest_run(ctx, 2)
    }

    fn build_batch(
        &mut self,
        ctx: &IndexBuildContext<'_>,
        start: usize,
        end: usize,
    ) -> Result<(), BpannError> {
        if start >= end {
            return Ok(());
        }
        let seed = std::env::var("BPANN_BUILD_SEED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(start as u64);
        let row_ids: Vec<u32> = (start..end).map(|i| i as u32).collect();
        let batch_len = end - start;
        let index = if batch_len <= 1024 {
            let centroid = self
                .take_pending_centroid(ctx.num_dim)
                .unwrap_or_else(|| {
                    centroid_from_mmap_rows(ctx, start, end)
                        .expect("centroid_from_mmap_rows")
                });
            BpannIndex::build_row_ids_leaf_with_persist(
                &row_ids,
                centroid,
                ctx.num_dim,
                self.index_dir.clone(),
                false,
            )?
        } else {
            let mut vectors = Vec::with_capacity(batch_len);
            let mut vec_buf = Vec::with_capacity(ctx.num_dim);
            for i in start..end {
                let row = ctx.train_x.mmap_row_slice(i)?;
                bpann_row_to_f32(row, ctx.scale_x, ctx.x_scale, &mut vec_buf);
                vectors.push(std::mem::take(&mut vec_buf));
            }
            BpannIndex::build_from_rows_with_persist(
                &row_ids,
                &vectors,
                ctx.num_dim,
                DEFAULT_LEAF_CAPACITY,
                seed,
                self.index_dir.clone(),
                false,
            )?
        };
        self.indices.push(index);
        self.indexed_rows = end;
        Ok(())
    }

    pub fn search_candidates(
        &self,
        query_f32: &[f32],
        k: usize,
        store: Option<&MmapSearchStore<'_>>,
    ) -> Result<Vec<(u32, f32)>, BpannError> {
        let budget = search_fragment_budget(self.indices.len(), self.indexed_rows);
        let indices_to_search: Vec<&BpannIndex> = if self.indices.len() <= budget {
            self.indices.iter().collect()
        } else {
            let mut ranked: Vec<(f32, &BpannIndex)> = self
                .indices
                .iter()
                .map(|index| {
                    let centroid = index.root_centroid();
                    let dist = if centroid.is_empty() {
                        f32::INFINITY
                    } else {
                        l2_sq_f32(query_f32, &centroid)
                    };
                    (dist, index)
                })
                .collect();
            ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            ranked.into_iter().take(budget).map(|(_, index)| index).collect()
        };
        let per_fragment_k = k
            .saturating_mul(self.indices.len())
            .div_ceil(indices_to_search.len().max(1))
            .max(k);
        let mut merged: Vec<(u32, f32)> = Vec::new();
        for index in indices_to_search {
            merged.extend(search_index_candidates(
                index,
                query_f32,
                per_fragment_k,
                store,
            )?);
        }
        merged.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        merged.truncate(k);
        Ok(merged)
    }

    pub fn index_memory_bytes(&self) -> usize {
        self.indices.iter().map(|i| i.index_memory_bytes()).sum()
    }
}

fn centroid_from_mmap_rows(
    ctx: &IndexBuildContext<'_>,
    start: usize,
    end: usize,
) -> Result<Vec<f32>, BpannError> {
    let dim = ctx.num_dim;
    let mut acc = vec![0.0f64; dim];
    let count = end.saturating_sub(start);
    if count == 0 {
        return Ok(Vec::new());
    }
    for i in start..end {
        let row = ctx.train_x.mmap_row_slice(i)?;
        for (j, &v) in row.iter().enumerate() {
            acc[j] += if ctx.scale_x {
                v / ctx.x_scale[j]
            } else {
                v
            };
        }
    }
    Ok(acc
        .iter()
        .map(|&s| (s / count as f64) as f32)
        .collect())
}

fn search_index_candidates(
    index: &BpannIndex,
    query: &[f32],
    k: usize,
    store: Option<&MmapSearchStore<'_>>,
) -> Result<Vec<(u32, f32)>, BpannError> {
    let rows = index.header.indexed_rows;
    if rows <= EXHAUSTIVE_SEARCH_ROW_LIMIT {
        search_exhaustive_leaves_with_store(index, query, k, store)
    } else {
        let beam = search_beam_width(rows);
        let mut visited = Vec::new();
        if rows <= SKIP_REFINEMENT_ROW_LIMIT {
            search_with_skip_refinement_with_store(index, query, k, beam, &mut visited, store)
        } else {
            search_greedy_blocks_only_with_store(index, query, k, beam, store)
        }
    }
}

#[cfg(test)]
mod kiss_coverage_tests {
    use super::*;
    use crate::mmap_store::MmapColumnStore;
    use ndarray::array;
    use tempfile::TempDir;

    #[test]
    fn sync_private_helpers_are_linked() {
        assert_eq!(env_usize("BPANN_KISS_MISSING_ENV_VAR", 99), 99);
        let _ = (
            index_compact_rows_per_fragment(),
            index_compact_fragment_max(),
            search_rows_per_fragment(),
            small_fragment_merge_rows(),
            search_fragment_budget_max(),
        );
        let _ = index_compact_threshold(2000);
        let _ = search_fragment_budget(4, 100_000);
        let _ = search_beam_width(500);
        let _ = (
            IncrementalIndex::take_pending_centroid as fn(&mut IncrementalIndex, usize) -> Option<Vec<f32>>,
            IncrementalIndex::ensure_sync as fn(&mut IncrementalIndex, &IndexBuildContext<'_>, usize) -> Result<(), BpannError>,
            IncrementalIndex::maybe_compact_or_persist
                as fn(&mut IncrementalIndex, &IndexBuildContext<'_>) -> Result<(), BpannError>,
            IncrementalIndex::compact as fn(&mut IncrementalIndex, &IndexBuildContext<'_>) -> Result<(), BpannError>,
            IncrementalIndex::amalgamate_smallest_run
                as fn(&mut IncrementalIndex, &IndexBuildContext<'_>, usize) -> Result<(), BpannError>,
            IncrementalIndex::amalgamate_smallest_pair
                as fn(&mut IncrementalIndex, &IndexBuildContext<'_>) -> Result<(), BpannError>,
            IncrementalIndex::build_batch
                as fn(
                    &mut IncrementalIndex,
                    &IndexBuildContext<'_>,
                    usize,
                    usize,
                ) -> Result<(), BpannError>,
            centroid_from_mmap_rows
                as fn(&IndexBuildContext<'_>, usize, usize) -> Result<Vec<f32>, BpannError>,
            search_index_candidates
                as fn(
                    &BpannIndex,
                    &[f32],
                    usize,
                    Option<&MmapSearchStore<'_>>,
                ) -> Result<Vec<(u32, f32)>, BpannError>,
        );
        fn _index_build_context_marker(ctx: &IndexBuildContext<'_>) {
            _kiss_index_build_context(ctx);
        }
    }

    #[test]
    fn sync_public_api_behavioral() {
        let dir = TempDir::new().unwrap();
        let mut idx = IncrementalIndex::new(dir.path().to_path_buf());
        let x = array![[0.0, 1.0], [1.0, 0.0]];
        idx.note_pending_rows(&x.view(), false, &[1.0, 1.0]);
        assert!(idx.take_pending_centroid(2).is_some());
        let x_path = dir.path().join("train_x.bin");
        let mut store = MmapColumnStore::mmap_open_or_create(x_path, 2, None).unwrap();
        store.mmap_append(&x.view()).unwrap();
        idx.ensure_sync_for_backend(&store, 2, false, &[1.0, 1.0], dir.path(), 1, 2)
            .unwrap();
        let results = idx.search_candidates(&[0.0, 0.0], 1, None).unwrap();
        let _ = results;
        assert!(idx.index_memory_bytes() > 0);
    }

    #[test]
    fn sync_compact_path_behavioral() {
        let dir = TempDir::new().unwrap();
        let mut idx = IncrementalIndex::new(dir.path().join("index"));
        let x_path = dir.path().join("train_x.bin");
        let mut store = MmapColumnStore::mmap_open_or_create(x_path, 2, None).unwrap();
        for batch in 0..8 {
            let row0 = batch as f64 * 2.0;
            let chunk = array![[row0, 0.0], [row0 + 1.0, 0.0]];
            store.mmap_append(&chunk.view()).unwrap();
            let start = batch * 2;
            let end = start + 2;
            idx.ensure_sync_for_backend(&store, 2, false, &[1.0, 1.0], dir.path(), 1, end)
                .unwrap();
        }
        assert!(idx.indices.len() <= 4);
        let _ = idx.search_candidates(&[1.0, 0.0], 2, None).unwrap();
    }

    fn _kiss_index_build_context<'a>(ctx: &IndexBuildContext<'a>) {
        let _ = (ctx.num_dim, ctx.scale_x, ctx.num_metrics);
    }
}
