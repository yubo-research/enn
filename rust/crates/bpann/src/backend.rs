use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2, ArrayView2};

use crate::error::BpannError;
use crate::index::{BpannIndex, IncrementalIndex};
use crate::large_n_search::{search_indexed_and_pending, SearchPendingArgs};
use crate::mmap_store::MmapColumnStore;
use crate::observation::{
    self as obs, TrainRowsAt, INDEX_BACKEND, MAX_NUM_DIM, MAX_RECORD_STRIDE,
};
use crate::small_n_search::{
    score_queries_flat, ScoreQueriesFlat, SMALL_N_INCORE_SEARCH_LIMIT,
};

pub const PAPER_TEX_PATH: &str = "papers/bpann_2511.15557v1.tex";
pub use crate::tuning::{DEFAULT_PENDING_FLUSH_THRESHOLD, DEFAULT_PENDING_HARD_FLUSH_THRESHOLD};

pub struct BpannBackend {
    work_dir: PathBuf,
    pub(crate) train_x: MmapColumnStore,
    train_y: MmapColumnStore,
    train_yvar: Option<MmapColumnStore>,
    pub(crate) num_dim: usize,
    num_metrics: usize,
    pub(crate) scale_x: bool,
    pub(crate) x_scale: Array1<f64>,
    pub(crate) index: IncrementalIndex,
    pending_flush_threshold: usize,
    pending_hard_flush_threshold: usize,
    defer_append_indexing: bool,
    pending_unindexed: AtomicUsize,
    index_dirty: Mutex<bool>,
    num_obs_counter: obs::NumObsCounter,
    /// Resident flat `N·D` f32 train cache for the small-N search path.
    /// Invalidated on append. Arc so parallel queries share one buffer.
    pub(crate) small_n_x_cache: Mutex<Option<(usize, Arc<[f32]>)>>,
}

impl BpannBackend {
    pub fn new(
        work_dir: PathBuf,
        train_x: Array2<f64>,
        train_y: Array2<f64>,
        train_yvar: Option<Array2<f64>>,
        scale_x: bool,
        x_scale: Array1<f64>,
    ) -> Result<Self, BpannError> {
        obs::bpann_validate_dim_limits(train_x.ncols())?;
        fs::create_dir_all(&work_dir).map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        obs::bpann_validate_index_backend(&work_dir, INDEX_BACKEND)?;
        let num_obs_counter = obs::NumObsCounter::open(&work_dir)?;

        let num_dim = train_x.ncols();
        let num_metrics = train_y.ncols();
        let known_nrows = obs::bpann_load_num_obs(&work_dir);
        let mut train_x_store = MmapColumnStore::mmap_open_or_create(
            work_dir.join("train_x.bin"),
            num_dim,
            known_nrows,
        )?;
        let mut train_y_store = MmapColumnStore::mmap_open_or_create(
            work_dir.join("train_y.bin"),
            num_metrics,
            known_nrows,
        )?;
        if train_x_store.nrows > 0 && train_x.nrows() > 0 {
            return Err(BpannError::InvalidParameter(format!(
                "work_dir already has {} persisted rows; refusing to ignore nonempty train_x/train_y (nrows={}). Reopen with empty arrays to resume, or use a fresh work_dir",
                train_x_store.nrows,
                train_x.nrows()
            )));
        }
        if train_x_store.nrows == 0 && train_x.nrows() > 0 {
            train_x_store.mmap_append(&train_x.view())?;
            train_y_store.mmap_append(&train_y.view())?;
        }
        if train_x_store.nrows == 0 {
            // Pre-grow and pre-touch fresh stores so the first append pays no
            // file-resize, page-fault, or block-allocation cost.
            train_x_store.ensure_capacity(crate::mmap_store::MMAP_GROW_ROWS)?;
            train_y_store.ensure_capacity(crate::mmap_store::MMAP_GROW_ROWS)?;
            train_x_store.pretouch();
            train_y_store.pretouch();
        }
        let train_yvar_store =
            obs::bpann_open_or_append_yvar(&work_dir, num_metrics, train_yvar.as_ref())?;

        let n = train_x_store.nrows;
        let index_dir = work_dir.join("index");
        let indexed_rows = obs::bpann_load_indexed_rows(&work_dir).unwrap_or(0).min(n);
        let indices = if index_dir.join("header.json").exists() && indexed_rows > 0 {
            vec![BpannIndex::open(index_dir.clone())?]
        } else {
            Vec::new()
        };
        let persisted_rows = indices
            .first()
            .map(|i| i.header.indexed_rows)
            .unwrap_or(0);
        let mut index = IncrementalIndex::new(index_dir);
        index.indices = indices;
        index.indexed_rows = persisted_rows.min(indexed_rows);
        let mut backend = Self {
            work_dir,
            train_x: train_x_store,
            train_y: train_y_store,
            train_yvar: train_yvar_store,
            num_dim,
            num_metrics,
            scale_x,
            x_scale,
            index,
            pending_flush_threshold: DEFAULT_PENDING_FLUSH_THRESHOLD,
            pending_hard_flush_threshold: DEFAULT_PENDING_HARD_FLUSH_THRESHOLD,
            defer_append_indexing: true,
            pending_unindexed: AtomicUsize::new(n.saturating_sub(indexed_rows)),
            index_dirty: Mutex::new(indexed_rows < n),
            num_obs_counter,
            small_n_x_cache: Mutex::new(None),
        };
        if persisted_rows < indexed_rows {
            backend.index.ensure_sync_for_backend(
                &backend.train_x,
                backend.num_dim,
                backend.scale_x,
                backend.x_scale.as_slice().unwrap(),
                &backend.work_dir,
                backend.num_metrics,
                indexed_rows,
            )?;
        }
        backend.pending_unindexed
            .store(n.saturating_sub(indexed_rows), Ordering::Relaxed);
        backend.index.indexed_rows = indexed_rows;
        obs::bpann_write_metadata(
            &backend.work_dir,
            n,
            num_dim,
            num_metrics,
            scale_x,
            indexed_rows,
        )?;
        backend.num_obs_counter.set(n);
        Ok(backend)
    }

    pub fn new_empty(work_dir: PathBuf, num_dim: usize, num_metrics: usize) -> Result<Self, BpannError> {
        Self::new(
            work_dir,
            Array2::zeros((0, num_dim)),
            Array2::zeros((0, num_metrics)),
            None,
            false,
            Array1::ones(num_dim),
        )
    }

    pub fn with_defer_append_indexing(mut self, defer: bool) -> Self {
        self.defer_append_indexing = defer;
        self
    }

    pub fn defer_append_indexing(&self) -> bool {
        self.defer_append_indexing
    }

    pub fn pending_rows(&self) -> usize {
        self.pending_unindexed.load(Ordering::Relaxed)
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.train_x.nrows
    }

    pub fn num_dim(&self) -> usize {
        self.num_dim
    }

    pub fn num_metrics(&self) -> usize {
        self.num_metrics
    }

    pub fn mark_index_stale(&mut self) {
        self.reset_index();
    }

    pub fn ensure_index_sync_with_scale(
        &mut self,
        scale_x: bool,
        x_scale: &Array1<f64>,
    ) -> Result<(), BpannError> {
        if self.scale_x != scale_x || self.x_scale != *x_scale {
            self.scale_x = scale_x;
            self.x_scale = x_scale.to_owned();
            self.reset_index();
        }
        self.ensure_index_sync()
    }

    fn reset_index(&mut self) {
        self.index.reset();
        self.pending_unindexed
            .store(self.len(), Ordering::Relaxed);
        *self.index_dirty.lock().expect("index_dirty") = true;
    }

    pub fn indexed_rows(&self) -> usize {
        self.index.indexed_rows
    }

    pub fn append_row(
        &mut self,
        x: &Array1<f64>,
        y: &Array1<f64>,
        yvar: Option<&Array1<f64>>,
    ) -> Result<(), BpannError> {
        let x2 = x.clone().insert_axis(ndarray::Axis(0));
        let y2 = y.clone().insert_axis(ndarray::Axis(0));
        let yv2 = yvar.map(|v| v.clone().insert_axis(ndarray::Axis(0)));
        self.append_rows(
            &x2.view(),
            &y2.view(),
            yv2.as_ref().map(|a| a.view()).as_ref(),
        )
    }

    pub fn append_rows(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Result<(), BpannError> {
        if x.nrows() == 0 {
            return Ok(());
        }
        if x.ncols() != self.num_dim || y.ncols() != self.num_metrics || x.nrows() != y.nrows() {
            return Err(BpannError::InvalidShape {
                expected: vec![x.nrows(), self.num_dim],
                got: vec![x.nrows(), x.ncols()],
            });
        }
        obs::bpann_check_append_row_limit(self.len() + x.nrows())?;
        self.train_x.mmap_append(x)?;
        self.train_y.mmap_append(y)?;
        obs::bpann_append_yvar_on_add(
            &self.work_dir,
            self.num_metrics,
            &mut self.train_yvar,
            yvar,
        )?;
        self.index
            .note_pending_rows(x, self.scale_x, self.x_scale.as_slice().unwrap());
        self.pending_unindexed
            .fetch_add(x.nrows(), Ordering::Relaxed);
        *self.index_dirty.lock().expect("index_dirty") = true;
        *self.small_n_x_cache.lock().expect("small_n_x_cache") = None;
        self.num_obs_counter.set(self.len());
        let pending = self.pending_rows();
        // Hard cap: soft-sync on the caller when pending reaches the hard threshold
        // (deferred or not). Soft threshold syncs only on the non-deferred path.
        if pending >= self.pending_hard_flush_threshold
            || (!self.defer_append_indexing && pending >= self.pending_flush_threshold)
        {
            self.ensure_index_sync()?;
        }
        Ok(())
    }

    /// Soft sync: build/compact fragments in memory and update pending counters.
    /// May write `indexed_rows.bin`. Does not write `pages.bin` / `skip_edges.bin`
    /// and does not clear `index_dirty` (hard persist alone clears disk-dirty).
    ///
    /// Mutates the live index in place. Background flush still uses
    /// [`soft_sync_build`] / [`soft_sync_publish`] so readers can search the
    /// previous snapshot while a detached build runs.
    pub fn ensure_index_sync(&mut self) -> Result<(), BpannError> {
        let end = self.len();
        if self.index.indexed_rows >= end {
            self.pending_unindexed.store(0, Ordering::Relaxed);
            return Ok(());
        }
        self.index.ensure_sync_for_backend(
            &self.train_x,
            self.num_dim,
            self.scale_x,
            self.x_scale.as_slice().unwrap(),
            &self.work_dir,
            self.num_metrics,
            end,
        )?;
        self.pending_unindexed.store(0, Ordering::Relaxed);
        // Soft-sync and centroid builds fault train pages; drop them from RSS.
        self.release_observation_pages()?;
        Ok(())
    }

    /// Remap observation mmaps so faulted/dirty pages leave process RSS.
    pub fn release_observation_pages(&mut self) -> Result<(), BpannError> {
        self.train_x.release_resident_pages()?;
        self.train_y.release_resident_pages()?;
        if let Some(store) = self.train_yvar.as_mut() {
            store.release_resident_pages()?;
        }
        Ok(())
    }

    pub fn persist_index_to_disk(&mut self) -> Result<(), BpannError> {
        let index_dirty = *self.index_dirty.lock().expect("index_dirty");
        if !self
            .index
            .needs_disk_rewrite(index_dirty, self.train_x.nrows)
        {
            self.pending_unindexed.store(0, Ordering::Relaxed);
            *self.index_dirty.lock().expect("index_dirty") = false;
            return Ok(());
        }
        self.index.persist_to_disk_for_backend(
            &self.train_x,
            self.num_dim,
            self.scale_x,
            self.x_scale.as_slice().unwrap(),
            &self.work_dir,
            self.num_metrics,
        )?;
        self.pending_unindexed.store(0, Ordering::Relaxed);
        *self.index_dirty.lock().expect("index_dirty") = false;
        Ok(())
    }

    pub fn train_rows_at(&self, indices: &[usize]) -> Result<TrainRowsAt, BpannError> {
        obs::bpann_train_rows_at(
            self.len(),
            &self.train_x,
            &self.train_y,
            self.train_yvar.as_ref(),
            indices,
        )
    }

    pub fn search(
        &self,
        queries: &ArrayView2<f64>,
        search_k: usize,
        exclude_nearest: bool,
    ) -> Result<(Array2<f64>, Array2<i64>), BpannError> {
        let total = self.len();
        let n_query = queries.nrows();
        if total == 0 {
            return Ok((Array2::zeros((n_query, 0)), Array2::zeros((n_query, 0))));
        }
        let k_req = search_k.min(total);
        // Fetch/return up to k_req columns; exclude drops self when present.
        // Novel queries keep the true NN (including when k_req == total).
        let k_eff = k_req;
        let pool_k = if exclude_nearest {
            (k_eff + 1).min(total)
        } else {
            k_eff
        };
        if k_eff == 0 {
            return Ok((
                Array2::zeros((n_query, 0)),
                Array2::zeros((n_query, 0)),
            ));
        }
        // Unfilled slots must not look like a valid neighbor (idx 0, dist 0).
        let mut dist2s = Array2::from_elem((n_query, k_eff), f64::INFINITY);
        let mut indices = Array2::from_elem((n_query, k_eff), -1i64);
        let scale_x = self.scale_x;
        let x_scale_vec = self.x_scale.as_slice().unwrap().to_vec();
        let num_dim = self.num_dim;
        let query_rows: Vec<Vec<f64>> = (0..n_query)
            .map(|q| queries.row(q).to_vec())
            .collect();

        // Small-N: resident flat f32 cache + heap top-k (shared across queries).
        if total <= SMALL_N_INCORE_SEARCH_LIMIT {
            let flat = crate::small_n_search::load_or_build_small_n_cache(self, total)?;
            let per_query = score_queries_flat(
                &query_rows,
                &ScoreQueriesFlat {
                    flat: &flat,
                    total,
                    num_dim,
                    scale_x,
                    x_scale: &x_scale_vec,
                    k_eff,
                    pool_k,
                    exclude_nearest,
                },
            );
            for (q, (dist_row, idx_row)) in per_query.into_iter().enumerate() {
                for j in 0..k_eff {
                    dist2s[[q, j]] = dist_row[j];
                    indices[[q, j]] = idx_row[j];
                }
            }
            return Ok(trim_trailing_invalid_neighbor_cols(dist2s, indices));
        }

        search_indexed_and_pending(
            self,
            &query_rows,
            &mut dist2s,
            &mut indices,
            SearchPendingArgs {
                total,
                k_eff,
                pool_k,
                exclude_nearest,
                scale_x,
                x_scale: &x_scale_vec,
                num_dim,
            },
        )?;
        Ok(trim_trailing_invalid_neighbor_cols(dist2s, indices))
    }

    pub fn index_snapshot(&self) -> Option<&BpannIndex> {
        self.index.indices.first()
    }

    pub fn page_bytes(&self) -> Vec<u8> {
        self.index
            .indices
            .first()
            .map(|i| i.page_bytes())
            .unwrap_or_default()
    }

    pub fn mmap_row_slice(&self, i: usize) -> Result<&[f64], BpannError> {
        self.train_x.mmap_row_slice(i)
    }

    /// Y (and optional yvar) row slices without touching `train_x`.
    pub fn mmap_row_y_and_yvar(
        &self,
        i: usize,
    ) -> Result<(&[f64], Option<&[f64]>), BpannError> {
        let y = self.train_y.mmap_row_slice(i)?;
        let yvar = match self.train_yvar.as_ref() {
            None => None,
            Some(store) => Some(store.mmap_row_slice(i)?),
        };
        Ok((y, yvar))
    }

    pub fn index_memory_bytes(&self) -> usize {
        self.index.index_memory_bytes()
    }

    pub fn reopen(work_dir: PathBuf) -> Result<Self, BpannError> {
        let meta_path = work_dir.join("metadata.json");
        let text = fs::read_to_string(&meta_path)
            .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        let num_dim = crate::observation::parse_json_usize_field(&text, "num_dim")
            .ok_or_else(|| BpannError::InvalidParameter("missing num_dim".to_string()))?;
        let num_metrics = crate::observation::parse_json_usize_field(&text, "num_metrics")
            .ok_or_else(|| BpannError::InvalidParameter("missing num_metrics".to_string()))?;
        let scale_x = text.contains("\"scale_x\":true");
        Self::new(
            work_dir,
            Array2::zeros((0, num_dim)),
            Array2::zeros((0, num_metrics)),
            None,
            scale_x,
            Array1::ones(num_dim),
        )
    }
}

impl BpannBackend {
    pub fn with_pending_flush_threshold(mut self, threshold: usize) -> Self {
        self.pending_flush_threshold = threshold;
        if self.pending_hard_flush_threshold < threshold {
            self.pending_hard_flush_threshold = threshold;
        }
        self
    }

    pub fn with_pending_hard_flush_threshold(mut self, threshold: usize) -> Self {
        self.pending_hard_flush_threshold = threshold.max(self.pending_flush_threshold);
        self
    }

    pub fn pending_flush_threshold(&self) -> usize {
        self.pending_flush_threshold
    }

    pub fn pending_hard_flush_threshold(&self) -> usize {
        self.pending_hard_flush_threshold
    }

    /// Update soft/hard pending flush thresholds (keeps `hard >= soft`).
    pub fn reconfigure_flush_thresholds(&mut self, soft: usize, hard: usize) {
        let soft = soft.max(1);
        self.pending_flush_threshold = soft;
        self.pending_hard_flush_threshold = hard.max(soft);
    }
}

/// Drop trailing neighbor columns that are invalid (`idx < 0`) in every query row.
fn trim_trailing_invalid_neighbor_cols(
    dist2s: Array2<f64>,
    indices: Array2<i64>,
) -> (Array2<f64>, Array2<i64>) {
    let n_query = indices.nrows();
    let mut width = indices.ncols();
    while width > 0 && (0..n_query).all(|r| indices[[r, width - 1]] < 0) {
        width -= 1;
    }
    if width == indices.ncols() {
        (dist2s, indices)
    } else if width == 0 {
        (
            Array2::zeros((n_query, 0)),
            Array2::zeros((n_query, 0)),
        )
    } else {
        (
            dist2s
                .slice_axis(ndarray::Axis(1), ndarray::Slice::from(..width))
                .to_owned(),
            indices
                .slice_axis(ndarray::Axis(1), ndarray::Slice::from(..width))
                .to_owned(),
        )
    }
}

/// Build a soft-sync result under a shared borrow (no publish).
/// Readers may search the live index concurrently while this runs.
pub fn soft_sync_build(backend: &BpannBackend) -> Result<Option<IncrementalIndex>, BpannError> {
    let end = backend.len();
    if backend.index.indexed_rows >= end {
        return Ok(None);
    }
    let mut working = backend.index.clone();
    working.ensure_sync_for_backend(
        &backend.train_x,
        backend.num_dim,
        backend.scale_x,
        backend.x_scale.as_slice().unwrap(),
        &backend.work_dir,
        backend.num_metrics,
        end,
    )?;
    Ok(Some(working))
}

/// Publish a detached soft-sync result under exclusive borrow.
pub fn soft_sync_publish(backend: &mut BpannBackend, built: IncrementalIndex) {
    backend.index = built;
    backend
        .pending_unindexed
        .store(0, Ordering::Relaxed);
}

pub fn open_rejects_num_dim(num_dim: usize) -> Result<(), BpannError> {
    obs::bpann_validate_dim_limits(num_dim)
}

pub fn open_rejects_record_stride(num_dim: usize) -> Result<(), BpannError> {
    let record_stride = num_dim * std::mem::size_of::<f64>();
    if record_stride > MAX_RECORD_STRIDE {
        return Err(BpannError::InvalidParameter(format!(
            "record_stride {record_stride} exceeds maximum {MAX_RECORD_STRIDE}"
        )));
    }
    if num_dim > MAX_NUM_DIM {
        return Err(BpannError::InvalidParameter(format!(
            "num_dim {num_dim} exceeds maximum {MAX_NUM_DIM}"
        )));
    }
    Ok(())
}

