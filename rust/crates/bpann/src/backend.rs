use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use ndarray::{Array1, Array2, ArrayView2};
use rayon::prelude::*;

use crate::distance::bpann_row_to_f32;
use crate::error::BpannError;
use crate::index::{bpann_brute_force_topk_mmap, BpannIndex, IncrementalIndex, MmapSearchStore};
use crate::merge::{bpann_merge_topk_candidates, merge_topk_precomputed_dist};
use crate::mmap_store::MmapColumnStore;
use crate::observation::{
    self as obs, TrainRowsAt, INDEX_BACKEND, MAX_NUM_DIM, MAX_RECORD_STRIDE,
};

pub const PAPER_TEX_PATH: &str = "papers/bpann_2511.15557v1.tex";
pub const DEFAULT_PENDING_FLUSH_THRESHOLD: usize = 1000;

pub struct BpannBackend {
    work_dir: PathBuf,
    train_x: MmapColumnStore,
    train_y: MmapColumnStore,
    train_yvar: Option<MmapColumnStore>,
    num_dim: usize,
    num_metrics: usize,
    scale_x: bool,
    x_scale: Array1<f64>,
    index: IncrementalIndex,
    pending_flush_threshold: usize,
    defer_append_indexing: bool,
    pending_unindexed: AtomicUsize,
    index_dirty: Mutex<bool>,
    num_obs_counter: obs::NumObsCounter,
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
        if train_x_store.nrows == 0 && train_x.nrows() > 0 {
            train_x_store.mmap_append(&train_x.view())?;
            train_y_store.mmap_append(&train_y.view())?;
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
            defer_append_indexing: true,
            pending_unindexed: AtomicUsize::new(n.saturating_sub(indexed_rows)),
            index_dirty: Mutex::new(indexed_rows < n),
            num_obs_counter,
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

    pub fn with_pending_flush_threshold(mut self, threshold: usize) -> Self {
        self.pending_flush_threshold = threshold;
        self
    }

    pub fn with_defer_append_indexing(mut self, defer: bool) -> Self {
        self.defer_append_indexing = defer;
        self
    }

    pub fn pending_flush_threshold(&self) -> usize {
        self.pending_flush_threshold
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

    pub fn index_dir(&self) -> &Path {
        &self.index.index_dir
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
        self.num_obs_counter.set(self.len());
        if !self.defer_append_indexing
            && self.pending_rows() >= self.pending_flush_threshold
        {
            self.ensure_index_sync()?;
        }
        Ok(())
    }

    pub fn ensure_index_sync(&mut self) -> Result<(), BpannError> {
        let end = self.len();
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
        let indexed = self.index.indexed_rows;
        let k_eff = search_k.min(total);
        let pool_k = if exclude_nearest {
            (search_k + 1).min(total)
        } else {
            k_eff
        };
        let mut dist2s = Array2::zeros((n_query, k_eff));
        let mut indices = Array2::zeros((n_query, k_eff));
        let scale_x = self.scale_x;
        let x_scale_vec = self.x_scale.as_slice().unwrap().to_vec();
        let pending_start = indexed;
        let has_pending = pending_start < total;
        let index_k = if exclude_nearest {
            pool_k.max(k_eff * 2)
        } else {
            k_eff
        };
        let num_dim = self.num_dim;
        let train_x = &self.train_x;
        let query_rows: Vec<Vec<f64>> = (0..n_query)
            .map(|q| queries.row(q).to_vec())
            .collect();
        let per_query: Vec<(Vec<f64>, Vec<i64>)> = query_rows
            .par_iter()
            .map(|query_buf| {
                let mut query_f32 = Vec::with_capacity(num_dim);
                bpann_row_to_f32(
                    query_buf,
                    scale_x,
                    &x_scale_vec,
                    &mut query_f32,
                );
                let store = MmapSearchStore {
                    train_x,
                    scale_x,
                    x_scale: &x_scale_vec,
                };

                let leg_a = if indexed > 0 && !self.index.indices.is_empty() {
                    self.index
                        .search_candidates(&query_f32, index_k.max(1), Some(&store))
                        .expect("search_candidates")
                } else {
                    Vec::new()
                };

                if !has_pending {
                    let merged = if scale_x {
                        bpann_merge_topk_candidates(
                            &self.train_x,
                            query_buf,
                            &leg_a,
                            &[],
                            k_eff,
                            pool_k,
                            exclude_nearest,
                            scale_x,
                            &x_scale_vec,
                        )
                        .expect("bpann_merge_topk_candidates")
                    } else {
                        merge_topk_precomputed_dist(
                            &leg_a,
                            &[],
                            k_eff,
                            pool_k,
                            exclude_nearest,
                        )
                    };
                    let mut dist_row = vec![0.0; k_eff];
                    let mut idx_row = vec![0; k_eff];
                    for (j, (id, dist)) in merged.into_iter().enumerate() {
                        dist_row[j] = dist;
                        idx_row[j] = id as i64;
                    }
                    return (dist_row, idx_row);
                }

                let leg_b = bpann_brute_force_topk_mmap(
                    &self.train_x,
                    pending_start,
                    total,
                    query_buf,
                    k_eff,
                    scale_x,
                    &x_scale_vec,
                )
                .expect("bpann_brute_force_topk_mmap");

                let merged = if scale_x {
                    bpann_merge_topk_candidates(
                        &self.train_x,
                        query_buf,
                        &leg_a,
                        &leg_b,
                        k_eff,
                        pool_k,
                        exclude_nearest,
                        scale_x,
                        &x_scale_vec,
                    )
                    .expect("bpann_merge_topk_candidates")
                } else {
                    merge_topk_precomputed_dist(
                        &leg_a,
                        &leg_b,
                        k_eff,
                        pool_k,
                        exclude_nearest,
                    )
                };
                let mut dist_row = vec![0.0; k_eff];
                let mut idx_row = vec![0; k_eff];
                for (j, (id, dist)) in merged.into_iter().enumerate() {
                    dist_row[j] = dist;
                    idx_row[j] = id as i64;
                }
                (dist_row, idx_row)
            })
            .collect();
        for (q, (dist_row, idx_row)) in per_query.into_iter().enumerate() {
            for j in 0..k_eff {
                dist2s[[q, j]] = dist_row[j];
                indices[[q, j]] = idx_row[j];
            }
        }
        Ok((dist2s, indices))
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
