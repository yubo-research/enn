//! Disk-backed ENN backend (mmap observations + in-tree HNSW graph).

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2, ArrayView2};
use rand::{rngs::StdRng, SeedableRng};

use crate::backend::disk_observation as disk_obs;
use super::{
    flush::{try_schedule_background_flush, wait_for_background_flush, BackgroundFlushState},
    graph_header::GraphHeader,
    graph_mut::GraphMut,
    hnsw,
    node_layout::NodeLayout,
    params::ef_search_for_k,
    store::MmapGraph,
    HnswHeader,
};
use crate::backend::DiskEnnBackend;
use crate::error::ENNError;
use crate::index::IndexDriver;
use crate::knn::MmapColumnStore;

const INDEX_BACKEND: &str = "hnsw_disk";
pub const DEFAULT_PENDING_FLUSH_THRESHOLD: usize = 1000;

pub struct DiskHnswEnnBackend {
    work_dir: PathBuf,
    train_x: MmapColumnStore,
    train_y: MmapColumnStore,
    train_yvar: Option<MmapColumnStore>,
    num_dim: usize,
    num_metrics: usize,
    driver: IndexDriver,
    scale_x: bool,
    x_scale: Array1<f64>,
    graph_dir: PathBuf,
    graph: MmapGraph,
    graph_header: GraphHeader,
    hnsw_header: HnswHeader,
    indexed_rows: usize,
    pending_flush_threshold: usize,
    index_dirty: Mutex<bool>,
    index_stale: Mutex<bool>,
    pub(crate) flush: Arc<Mutex<BackgroundFlushState>>,
}

impl DiskHnswEnnBackend {
    pub fn new(
        work_dir: PathBuf,
        train_x: Array2<f64>,
        train_y: Array2<f64>,
        train_yvar: Option<Array2<f64>>,
        scale_x: bool,
        x_scale: Array1<f64>,
        driver: IndexDriver,
    ) -> Result<Self, ENNError> {
        if driver != IndexDriver::HNSWDisk {
            return Err(ENNError::InvalidParameter(
                "DiskHnswEnnBackend requires IndexDriver::HNSWDisk".to_string(),
            ));
        }
        let layout = NodeLayout::new(train_x.ncols());
        disk_obs::validate_dim_limits(train_x.ncols(), layout.record_stride)?;

        fs::create_dir_all(&work_dir).map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
        disk_obs::validate_index_backend(&work_dir, INDEX_BACKEND)?;

        let num_dim = train_x.ncols();
        let num_metrics = train_y.ncols();
        let x_path = work_dir.join("train_x.bin");
        let y_path = work_dir.join("train_y.bin");
        // Row counts come from train file bytes; disk_obs::load_num_obs is for metadata introspection.
        let mut train_x_store = MmapColumnStore::mmap_open_or_create(x_path, num_dim, None)?;
        let mut train_y_store =
            MmapColumnStore::mmap_open_or_create(y_path, num_metrics, None)?;
        if train_x_store.nrows == 0 && train_x.nrows() > 0 {
            train_x_store.mmap_append(&train_x.view())?;
            train_y_store.mmap_append(&train_y.view())?;
        }
        let train_yvar_store =
            disk_obs::open_or_append_yvar(&work_dir, num_metrics, train_yvar.as_ref())?;

        let n = train_x_store.nrows;
        let graph_dir = work_dir.join("graph");
        let (graph, graph_header, hnsw_header, indexed_rows) =
            open_or_create_graph(&graph_dir, num_dim, &work_dir, n)?;

        disk_obs::write_metadata(
            &work_dir,
            n,
            num_dim,
            num_metrics,
            scale_x,
            indexed_rows,
            INDEX_BACKEND,
        )?;

        Ok(Self {
            work_dir,
            train_x: train_x_store,
            train_y: train_y_store,
            train_yvar: train_yvar_store,
            num_dim,
            num_metrics,
            driver,
            scale_x,
            x_scale,
            graph_dir,
            graph,
            graph_header,
            hnsw_header,
            indexed_rows,
            pending_flush_threshold: DEFAULT_PENDING_FLUSH_THRESHOLD,
            index_dirty: Mutex::new(indexed_rows < n),
            index_stale: Mutex::new(false),
            flush: Arc::new(Mutex::new(BackgroundFlushState::default())),
        })
    }

    pub fn with_pending_flush_threshold(mut self, threshold: usize) -> Self {
        self.pending_flush_threshold = threshold;
        self
    }

    #[doc(hidden)]
    pub fn set_pending_flush_threshold(&mut self, threshold: usize) {
        self.pending_flush_threshold = threshold;
    }

    pub fn pending_flush_threshold(&self) -> usize {
        self.pending_flush_threshold
    }

    pub fn pending_rows(&self) -> usize {
        self.len().saturating_sub(self.indexed_rows)
    }

    pub fn is_index_stale(&self) -> bool {
        *self
            .index_stale
            .lock()
            .expect("index_stale mutex poisoned")
    }

    /// True when search may skip `ensure_index_sync` (pending tier is searchable).
    pub fn defer_index_sync_for_search(&self) -> bool {
        !self.is_index_stale()
    }

    pub fn schedule_background_flush(
        &self,
        disk_arc: Arc<Mutex<DiskEnnBackend>>,
    ) -> Result<(), ENNError> {
        if self.is_index_stale() || self.pending_rows() < self.pending_flush_threshold {
            return Ok(());
        }
        try_schedule_background_flush(&self.flush, disk_arc)
    }

    pub fn wait_for_flush(&self) -> Result<(), ENNError> {
        wait_for_background_flush(&self.flush)
    }

    #[doc(hidden)]
    pub fn flush_arc(&self) -> Arc<Mutex<BackgroundFlushState>> {
        Arc::clone(&self.flush)
    }

    #[doc(hidden)]
    pub fn flush_test_barrier_hold(&self, hold: bool) {
        self.flush
            .lock()
            .expect("flush state mutex poisoned")
            .barrier
            .set_hold(hold);
    }

    #[doc(hidden)]
    pub fn inject_next_flush_failure(&self) {
        self.flush
            .lock()
            .expect("flush state mutex poisoned")
            .inject_failure();
    }

    pub fn new_empty(work_dir: PathBuf, num_dim: usize, num_metrics: usize) -> Result<Self, ENNError> {
        Self::new(
            work_dir,
            Array2::zeros((0, num_dim)),
            Array2::zeros((0, num_metrics)),
            None,
            false,
            Array1::ones(num_dim),
            IndexDriver::HNSWDisk,
        )
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

    pub fn driver(&self) -> IndexDriver {
        self.driver
    }

    pub fn indexed_rows(&self) -> usize {
        self.indexed_rows
    }

    pub fn mark_index_stale(&self) {
        disk_obs::set_index_stale(&self.index_stale);
    }
}

fn row_to_f32(backend: &DiskHnswEnnBackend, row: &[f64], out: &mut Vec<f32>) {
    out.clear();
    if backend.scale_x {
        out.extend(
            row.iter()
                .zip(backend.x_scale.iter())
                .map(|(&v, &s)| (v / s) as f32),
        );
    } else {
        out.extend(row.iter().map(|&v| v as f32));
    }
}

fn index_row_range(backend: &mut DiskHnswEnnBackend, start: usize, end: usize) -> Result<(), ENNError> {
    if start >= end {
        return Ok(());
    }
    let seed = std::env::var("ENN_HNSW_DISK_BUILD_SEED")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(start as u64);
    let mut rng = StdRng::seed_from_u64(seed);
    let mut vec_buf = Vec::with_capacity(backend.num_dim);

    for i in start..end {
        let row = backend.train_x.mmap_row_slice(i)?;
        row_to_f32(backend, row, &mut vec_buf);
        hnsw::insert(
            &mut backend.graph,
            &mut backend.hnsw_header,
            i as u32,
            &vec_buf,
            &mut rng,
        );
    }
    backend.graph_header.entry_point = backend.hnsw_header.entry_point;
    backend.graph_header.max_level = backend.hnsw_header.max_level;
    backend.graph_header
        .write_json(&backend.graph_dir.join("header.json"))
        .map_err(ENNError::InvalidParameter)?;
    backend.indexed_rows = end;
    disk_obs::write_metadata(
        &backend.work_dir,
        backend.len(),
        backend.num_dim,
        backend.num_metrics,
        backend.scale_x,
        backend.indexed_rows,
        INDEX_BACKEND,
    )?;
    backend.graph.fsync().map_err(ENNError::InvalidParameter)?;
    Ok(())
}

impl DiskHnswEnnBackend {
    pub(crate) fn flush_pending_index_rows(&mut self) -> Result<(), ENNError> {
        let start = self.indexed_rows;
        let end = self.len();
        index_row_range(self, start, end)?;
        *self
            .index_dirty
            .lock()
            .expect("index_dirty mutex poisoned") = false;
        Ok(())
    }

    pub fn ensure_index_sync(
        &mut self,
        scale_x: bool,
        x_scale: &Array1<f64>,
    ) -> Result<(), ENNError> {
        self.scale_x = scale_x;
        self.x_scale = x_scale.to_owned();
        if scale_x {
            let rebuild = {
                let stale = self
                    .index_stale
                    .lock()
                    .expect("index_stale mutex poisoned");
                *stale
            };
            if !rebuild && self.indexed_rows >= self.len() {
                return Ok(());
            }
            if rebuild {
                self.indexed_rows = 0;
                self.hnsw_header = HnswHeader {
                    entry_point: 0,
                    max_level: 0,
                    num_dim: self.num_dim,
                };
                index_row_range(self, 0, self.len())?;
                *self
                    .index_stale
                    .lock()
                    .expect("index_stale mutex poisoned") = false;
            } else if self.indexed_rows < self.len() {
                index_row_range(self, self.indexed_rows, self.len())?;
            }
            *self
                .index_dirty
                .lock()
                .expect("index_dirty mutex poisoned") = false;
            return Ok(());
        }
        let n = self.len();
        if self.indexed_rows < n {
            index_row_range(self, self.indexed_rows, n)?;
        }
        *self
            .index_dirty
            .lock()
            .expect("index_dirty mutex poisoned") = false;
        Ok(())
    }

    pub fn row_x(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        Ok(Array1::from(self.train_x.mmap_row_slice(i)?.to_vec()))
    }

    pub fn row_y(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        Ok(Array1::from(self.train_y.mmap_row_slice(i)?.to_vec()))
    }

    pub fn row_yvar(&self, i: usize) -> Result<Option<Array1<f64>>, ENNError> {
        disk_obs::mmap_row_yvar(self.train_yvar.as_ref(), i)
    }

    pub fn search(
        &self,
        x: &ArrayView2<f64>,
        search_k: i32,
        exclude_nearest: bool,
    ) -> Result<(Array2<f64>, Array2<i64>), ENNError> {
        let total = self.len();
        if total == 0 {
            return Ok((Array2::zeros((x.nrows(), 0)), Array2::zeros((x.nrows(), 0))));
        }
        let indexed = self.indexed_rows;
        let k_eff = (search_k as usize).min(total);
        let pool_k = if exclude_nearest {
            ((search_k + 1) as usize).min(total)
        } else {
            k_eff
        };
        let hnsw_k = (2 * k_eff).min(indexed);
        let hnsw_ef = ef_search_for_k(hnsw_k.max(1));
        let pending_k = k_eff;

        let mut dist2s = Array2::zeros((x.nrows(), k_eff));
        let mut indices = Array2::zeros((x.nrows(), k_eff));
        let scale_x = self.scale_x;
        let x_scale = self.x_scale.view();
        let train_x = &self.train_x;

        for q in 0..x.nrows() {
            let query_row = x.slice(ndarray::s![q, ..]);
            let query: Vec<f64> = query_row.iter().copied().collect();

            let leg_a: Vec<(u32, f32)> = if indexed > 0 && hnsw_k > 0 {
                let mut query_f32 = Vec::with_capacity(self.num_dim);
                row_to_f32(self, &query, &mut query_f32);
                hnsw::search(
                    &self.graph,
                    &self.hnsw_header,
                    &query_f32,
                    hnsw_k,
                    hnsw_ef,
                    indexed as u32,
                )
            } else {
                Vec::new()
            };

            let pending_start = if indexed == 0 { 0 } else { indexed };
            let leg_b = hnsw::brute_force_topk_mmap(
                train_x,
                pending_start,
                total,
                &query,
                pending_k,
                scale_x,
                x_scale.as_slice().unwrap(),
            )?;

            let merged = hnsw::merge_topk_candidates(
                train_x,
                &query,
                &leg_a,
                &leg_b,
                k_eff,
                pool_k,
                exclude_nearest,
                scale_x,
                x_scale.as_slice().unwrap(),
            )?;

            for (j, (id, dist)) in merged.into_iter().enumerate() {
                dist2s[[q, j]] = dist;
                indices[[q, j]] = id as i64;
            }
        }

        Ok((dist2s, indices))
    }

    pub fn index_memory_bytes(&self) -> Result<usize, ENNError> {
        let nodes_path = self.graph_dir.join("nodes.bin");
        let header_path = self.graph_dir.join("header.json");
        let mut total = 0usize;
        for p in [nodes_path, header_path] {
            if p.exists() {
                total += p.metadata().map(|m| m.len() as usize).unwrap_or(0);
            }
        }
        Ok(total)
    }
}

fn open_or_create_graph(
    graph_dir: &Path,
    num_dim: usize,
    work_dir: &Path,
    num_obs: usize,
) -> Result<(MmapGraph, GraphHeader, HnswHeader, usize), ENNError> {
    let indexed_rows = disk_obs::load_indexed_rows(work_dir).unwrap_or(0).min(num_obs);
    if graph_dir.join("header.json").exists() {
        let (graph, hdr) = MmapGraph::open(graph_dir).map_err(ENNError::InvalidParameter)?;
        if hdr.num_dim != num_dim {
            return Err(ENNError::InvalidParameter(format!(
                "graph num_dim {} != model num_dim {num_dim}",
                hdr.num_dim
            )));
        }
        let hnsw_header = HnswHeader {
            entry_point: hdr.entry_point,
            max_level: hdr.max_level,
            num_dim,
        };
        Ok((graph, hdr, hnsw_header, indexed_rows))
    } else {
        let (graph, hdr) =
            MmapGraph::create(graph_dir, num_dim).map_err(ENNError::InvalidParameter)?;
        let hnsw_header = HnswHeader {
            entry_point: 0,
            max_level: 0,
            num_dim,
        };
        Ok((graph, hdr, hnsw_header, 0))
    }
}

impl DiskHnswEnnBackend {
    pub fn append_rows(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Result<(), ENNError> {
        let current_len = self.train_x.nrows;
        disk_obs::append_disk_observation_rows(
            &mut disk_obs::DiskAppendContext {
                work_dir: &self.work_dir,
                num_metrics: self.num_metrics,
                train_x: &mut self.train_x,
                train_y: &mut self.train_y,
                train_yvar: &mut self.train_yvar,
                index_dirty: &self.index_dirty,
                current_len,
            },
            x,
            y,
            yvar,
        )?;
        Ok(())
    }

    pub fn train_rows_at(
        &self,
        indices: &[usize],
    ) -> Result<crate::backend::TrainRowsAtResult, ENNError> {
        disk_obs::train_rows_for_disk_backend(
            self.train_x.nrows,
            &self.train_x,
            &self.train_y,
            self.train_yvar.as_ref(),
            indices,
        )
    }
}
