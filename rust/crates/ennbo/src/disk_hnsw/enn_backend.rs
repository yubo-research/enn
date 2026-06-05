//! Disk-backed ENN backend (mmap observations + in-tree HNSW graph).

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ndarray::{Array1, Array2, ArrayView2};
use rand::{rngs::StdRng, SeedableRng};

use crate::backend::disk_observation as disk_obs;
use super::{
    graph_header::GraphHeader,
    graph_mut::GraphMut,
    hnsw,
    node_layout::NodeLayout,
    params::ef_search_for_k,
    store::MmapGraph,
    HnswHeader,
};
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
        })
    }

    pub fn with_pending_flush_threshold(mut self, threshold: usize) -> Self {
        self.pending_flush_threshold = threshold;
        self
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
        !self.is_index_stale() && self.pending_rows() < self.pending_flush_threshold
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

    pub fn len(&self) -> usize {
        self.train_x.nrows
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
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

    fn row_to_f32(&self, row: &[f64], out: &mut Vec<f32>) {
        out.clear();
        if self.scale_x {
            out.extend(row.iter().zip(self.x_scale.iter()).map(|(&v, &s)| (v / s) as f32));
        } else {
            out.extend(row.iter().map(|&v| v as f32));
        }
    }

    fn index_row_range(&mut self, start: usize, end: usize) -> Result<(), ENNError> {
        if start >= end {
            return Ok(());
        }
        let seed = std::env::var("ENN_HNSW_DISK_BUILD_SEED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(start as u64);
        let mut rng = StdRng::seed_from_u64(seed);
        let mut vec_buf = Vec::with_capacity(self.num_dim);

        for i in start..end {
            let row = self.train_x.mmap_row_slice(i)?;
            self.row_to_f32(row, &mut vec_buf);
            hnsw::insert(
                &mut self.graph,
                &mut self.hnsw_header,
                i as u32,
                &vec_buf,
                &mut rng,
            );
        }
        self.graph_header.entry_point = self.hnsw_header.entry_point;
        self.graph_header.max_level = self.hnsw_header.max_level;
        self.graph_header
            .write_json(&self.graph_dir.join("header.json"))
            .map_err(ENNError::InvalidParameter)?;
        self.indexed_rows = end;
        disk_obs::write_metadata(
            &self.work_dir,
            self.len(),
            self.num_dim,
            self.num_metrics,
            self.scale_x,
            self.indexed_rows,
            INDEX_BACKEND,
        )?;
        self.graph.fsync().map_err(ENNError::InvalidParameter)?;
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
                self.index_row_range(0, self.len())?;
                *self
                    .index_stale
                    .lock()
                    .expect("index_stale mutex poisoned") = false;
            } else if self.indexed_rows < self.len() {
                self.index_row_range(self.indexed_rows, self.len())?;
            }
            *self
                .index_dirty
                .lock()
                .expect("index_dirty mutex poisoned") = false;
            return Ok(());
        }
        let n = self.len();
        if self.indexed_rows < n {
            self.index_row_range(self.indexed_rows, n)?;
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
                self.row_to_f32(&query, &mut query_f32);
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
        if self.len().saturating_sub(self.indexed_rows) >= self.pending_flush_threshold {
            self.index_row_range(self.indexed_rows, self.len())?;
            *self
                .index_dirty
                .lock()
                .expect("index_dirty mutex poisoned") = false;
        }
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


#[cfg(test)]
mod disk_hnsw_unit_tests {
    use super::*;
    use crate::backend::disk_observation as disk_obs;
    use crate::disk_hnsw::EF_CONSTRUCTION;
    use super::hnsw::brute_force_topk;
    use ndarray::array;
    use tempfile::TempDir;

    #[test]
    fn disk_hnsw_new_empty_without_work_dir_errors() {
        let err = DiskHnswEnnBackend::new_empty(PathBuf::from("/nonexistent/path/for/test"), 2, 1);
        // new_empty with valid path works; test via EnnBackend in mod tests
        let dir = TempDir::new().expect("tempdir");
        let b = DiskHnswEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1).unwrap();
        assert_eq!(b.driver(), IndexDriver::HNSWDisk);
        let _ = err;
    }

    #[test]
    fn disk_hnsw_incremental_add_search() {
        let dir = TempDir::new().expect("tempdir");
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let train_y = array![[0.0], [1.0], [2.0]];
        let mut backend = DiskHnswEnnBackend::new(
            dir.path().to_path_buf(),
            train_x,
            train_y,
            None,
            false,
            Array1::ones(2),
            IndexDriver::HNSWDisk,
        )
        .unwrap();
        backend.ensure_index_sync(false, &Array1::ones(2)).unwrap();
        backend
            .append_rows(
                &array![[2.0, 2.0]].view(),
                &array![[3.0]].view(),
                None,
            )
            .unwrap();
        let query = array![[1.95, 1.95]];
        let (_, idx_before) = backend.search(&query.view(), 1, false).unwrap();
        assert_eq!(idx_before[[0, 0]], 3);
        backend.ensure_index_sync(false, &Array1::ones(2)).unwrap();
        let (_, idx_after) = backend.search(&query.view(), 1, false).unwrap();
        assert_eq!(idx_after[[0, 0]], 3);
    }


    #[test]
    fn disk_hnsw_reopen_existing_mmap_files() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().to_path_buf();
        let mut b1 = DiskHnswEnnBackend::new(
            path.clone(),
            array![[0.0, 0.0]],
            array![[0.0]],
            None,
            false,
            Array1::ones(2),
            IndexDriver::HNSWDisk,
        )
        .unwrap();
        b1.ensure_index_sync(false, &Array1::ones(2)).unwrap();
        let b2 = DiskHnswEnnBackend::new(
            path,
            array![[0.0, 0.0]],
            array![[0.0]],
            None,
            false,
            Array1::ones(2),
            IndexDriver::HNSWDisk,
        )
        .unwrap();
        assert_eq!(b2.len(), 1);
        let (_, idx) = b2.search(&array![[0.0, 0.0]].view(), 1, false).unwrap();
        assert_eq!(idx[[0, 0]], 0);
    }

    #[test]
    fn disk_hnsw_reopen_resumes_partial_sync() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().to_path_buf();
        let scale = Array1::ones(2);
        let mut backend = DiskHnswEnnBackend::new(
            path.clone(),
            array![[0.0, 0.0], [1.0, 0.0]],
            array![[0.0], [1.0]],
            None,
            false,
            scale.clone(),
            IndexDriver::HNSWDisk,
        )
        .unwrap();
        backend.ensure_index_sync(false, &scale).unwrap();
        backend
            .append_rows(
                &array![[2.0, 2.0], [3.0, 3.0]].view(),
                &array![[2.0], [3.0]].view(),
                None,
            )
            .unwrap();
        assert_eq!(backend.indexed_rows(), 2);
        assert_eq!(backend.len(), 4);
        drop(backend);
        let mut reopened = DiskHnswEnnBackend::new(
            path,
            Array2::zeros((0, 2)),
            Array2::zeros((0, 1)),
            None,
            false,
            scale.clone(),
            IndexDriver::HNSWDisk,
        )
        .unwrap();
        assert_eq!(reopened.indexed_rows(), 2);
        assert_eq!(reopened.len(), 4);
        reopened.ensure_index_sync(false, &scale).unwrap();
        assert_eq!(reopened.indexed_rows(), 4);
        reopened
            .search(&array![[0.1, 0.1]].view(), 2, false)
            .unwrap();
    }

    #[test]
    fn disk_hnsw_train_rows_at_parity() {
        let dir = TempDir::new().expect("tempdir");
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let train_y = array![[0.0], [1.0], [2.0]];
        let backend = DiskHnswEnnBackend::new(
            dir.path().to_path_buf(),
            train_x.clone(),
            train_y.clone(),
            None,
            false,
            Array1::ones(2),
            IndexDriver::HNSWDisk,
        )
        .unwrap();
        let (x, y, _) = backend.train_rows_at(&[0, 2]).unwrap();
        assert_eq!(x[[0, 0]], train_x[[0, 0]]);
        assert_eq!(y[[1, 0]], train_y[[2, 0]]);
    }

    #[test]
    fn disk_hnsw_persists_observation_files() {
        let dir = TempDir::new().expect("tempdir");
        let mut backend = DiskHnswEnnBackend::new(
            dir.path().to_path_buf(),
            array![[0.0, 0.0]],
            array![[0.0]],
            None,
            false,
            Array1::ones(2),
            IndexDriver::HNSWDisk,
        )
        .unwrap();
        backend
            .append_rows(
                &array![[1.0, 1.0]].view(),
                &array![[1.0]].view(),
                None,
            )
            .unwrap();
        assert!(dir.path().join("train_x.bin").exists());
    }

    #[test]
    fn disk_hnsw_append_exceeds_u32_max_errors() {
        let err = disk_obs::check_append_row_limit(u32::MAX as usize).unwrap_err();
        assert!(err.to_string().contains("u32::MAX"));
    }

    #[test]
    fn disk_hnsw_header_params_match_defaults() {
        let dir = TempDir::new().expect("tempdir");
        let mut backend = DiskHnswEnnBackend::new(
            dir.path().to_path_buf(),
            array![[0.0, 0.0]],
            array![[0.0]],
            None,
            false,
            Array1::ones(2),
            IndexDriver::HNSWDisk,
        )
        .unwrap();
        backend.ensure_index_sync(false, &Array1::ones(2)).unwrap();
        let text = fs::read_to_string(dir.path().join("graph/header.json")).unwrap();
        assert!(text.contains("\"M\":16"));
        assert!(text.contains("\"M0\":32"));
        assert!(text.contains("\"LMAX\":16"));
        assert!(text.contains(&format!("\"ef_construction\":{}", EF_CONSTRUCTION)));
    }

    #[test]
    fn disk_hnsw_rejects_mismatched_index_backend() {
        let dir = TempDir::new().expect("tempdir");
        disk_obs::write_metadata(dir.path(), 1, 2, 1, false, 0, "flat").unwrap();
        match DiskHnswEnnBackend::new(
            dir.path().to_path_buf(),
            array![[0.0, 0.0]],
            array![[0.0]],
            None,
            false,
            Array1::ones(2),
            IndexDriver::HNSWDisk,
        ) {
            Err(e) => assert!(e.to_string().contains("hnsw_disk")),
            Ok(_) => panic!("expected backend mismatch error"),
        }
    }

    #[test]
    fn disk_hnsw_open_with_yvar() {
        let dir = TempDir::new().expect("tempdir");
        let yv = array![[0.1], [0.2]];
        let backend = DiskHnswEnnBackend::new(
            dir.path().to_path_buf(),
            array![[0.0, 0.0], [1.0, 0.0]],
            array![[0.0], [1.0]],
            Some(yv),
            false,
            Array1::ones(2),
            IndexDriver::HNSWDisk,
        )
        .unwrap();
        assert!(backend.train_yvar.is_some());
        assert!(dir.path().join("train_yvar.bin").exists());
    }

    #[test]
    fn disk_hnsw_open_reopen_graph_dim_mismatch() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().to_path_buf();
        let mut b1 = DiskHnswEnnBackend::new(
            path.clone(),
            array![[0.0, 0.0]],
            array![[0.0]],
            None,
            false,
            Array1::ones(2),
            IndexDriver::HNSWDisk,
        )
        .unwrap();
        b1.ensure_index_sync(false, &Array1::ones(2)).unwrap();
        match DiskHnswEnnBackend::new(
            path,
            array![[0.0, 0.0, 0.0]],
            array![[0.0]],
            None,
            false,
            Array1::ones(3),
            IndexDriver::HNSWDisk,
        ) {
            Err(e) => assert!(e.to_string().contains("num_dim")),
            Ok(_) => panic!("expected dim mismatch"),
        }
    }

    #[test]
    fn disk_hnsw_reopen_skips_yvar_reappend() {
        let dir = TempDir::new().expect("tempdir");
        let yv = array![[0.1], [0.2]];
        let path = dir.path().to_path_buf();
        DiskHnswEnnBackend::new(
            path.clone(),
            array![[0.0, 0.0], [1.0, 0.0]],
            array![[0.0], [1.0]],
            Some(yv.clone()),
            false,
            Array1::ones(2),
            IndexDriver::HNSWDisk,
        )
        .unwrap();
        let reopened = DiskHnswEnnBackend::new(
            path,
            array![[0.0, 0.0], [1.0, 0.0]],
            array![[0.0], [1.0]],
            Some(yv),
            false,
            Array1::ones(2),
            IndexDriver::HNSWDisk,
        )
        .unwrap();
        assert_eq!(reopened.len(), 2);
    }

    #[test]
    fn disk_hnsw_append_adds_yvar_late() {
        let dir = TempDir::new().expect("tempdir");
        let mut backend =
            DiskHnswEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1).unwrap();
        backend
            .append_rows(
                &array![[0.0, 0.0]].view(),
                &array![[0.0]].view(),
                Some(&array![[0.5]].view()),
            )
            .unwrap();
        assert!(backend.train_yvar.is_some());
    }

    #[test]
    fn disk_hnsw_new_on_empty_work_dir_ok() {
        let dir = TempDir::new().expect("tempdir");
        let backend = DiskHnswEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1).unwrap();
        assert!(backend.is_empty());
    }

    #[test]
    fn disk_hnsw_multi_chunk_sync() {
        let dir = TempDir::new().expect("tempdir");
        let mut backend = DiskHnswEnnBackend::new_empty(dir.path().to_path_buf(), 4, 1).unwrap();
        let rows = 9000usize;
        let x = Array2::from_shape_fn((rows, 4), |(i, j)| (i + j) as f64);
        let y = Array2::zeros((rows, 1));
        backend.append_rows(&x.view(), &y.view(), None).unwrap();
        backend.ensure_index_sync(false, &Array1::ones(4)).unwrap();
        assert_eq!(backend.indexed_rows(), rows);
        assert_eq!(backend.indexed_rows(), backend.len());
    }

    #[test]
    fn disk_hnsw_scale_x_rebuild() {
        let dir = TempDir::new().expect("tempdir");
        let mut backend = DiskHnswEnnBackend::new(
            dir.path().to_path_buf(),
            array![[2.0, 4.0], [4.0, 8.0]],
            array![[0.0], [1.0]],
            None,
            false,
            Array1::from_elem(2, 2.0),
            IndexDriver::HNSWDisk,
        )
        .unwrap();
        backend.ensure_index_sync(false, &Array1::ones(2)).unwrap();
        backend.mark_index_stale();
        backend
            .ensure_index_sync(true, &Array1::from_elem(2, 2.0))
            .unwrap();
        assert_eq!(backend.indexed_rows(), backend.len());
    }

    #[test]
    fn disk_hnsw_search_exclude_nearest_and_row_yvar() {
        let dir = TempDir::new().expect("tempdir");
        let backend = DiskHnswEnnBackend::new(
            dir.path().to_path_buf(),
            array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            array![[0.0], [1.0], [2.0]],
            Some(array![[0.1], [0.2], [0.3]]),
            false,
            Array1::ones(2),
            IndexDriver::HNSWDisk,
        )
        .unwrap();
        let mut backend = backend;
        backend.ensure_index_sync(false, &Array1::ones(2)).unwrap();
        let (_, idx) = backend
            .search(&array![[0.0, 0.0]].view(), 2, true)
            .unwrap();
        assert_ne!(idx[[0, 0]], 0);
        assert!(backend.row_yvar(1).unwrap().is_some());
    }

    #[test]
    fn disk_hnsw_helper_functions_direct_coverage() {
        let dir = TempDir::new().expect("tempdir");
        assert!(disk_obs::open_or_append_yvar(dir.path(), 1, None).unwrap().is_none());
        assert!(disk_obs::validate_index_backend(dir.path(), INDEX_BACKEND).is_ok());
        disk_obs::write_metadata(dir.path(), 0, 2, 1, false, 0, INDEX_BACKEND).unwrap();
        assert!(disk_obs::validate_index_backend(dir.path(), INDEX_BACKEND).is_ok());
        let graph_dir = dir.path().join("graph");
        let (_g, _h, _hh, indexed) =
            open_or_create_graph(&graph_dir, 2, dir.path(), 0).expect("create graph");
        assert_eq!(indexed, 0);
        disk_obs::write_metadata(dir.path(), 3, 2, 1, false, 99, "hnsw_disk").unwrap();
        let (_g2, _h2, _hh2, indexed2) =
            open_or_create_graph(&graph_dir, 2, dir.path(), 3).expect("reopen graph");
        assert_eq!(indexed2, 3);
    }

    #[test]
    fn disk_hnsw_valid_metadata_tree_new_empty() {
        let dir = TempDir::new().expect("tempdir");
        disk_obs::write_metadata(dir.path(), 0, 2, 1, false, 0, "hnsw_disk").unwrap();
        let backend = DiskHnswEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1).unwrap();
        assert!(backend.is_empty());
    }

    #[test]
    fn disk_hnsw_scale_x_first_sync_without_mark_stale() {
        let dir = TempDir::new().expect("tempdir");
        let mut backend = DiskHnswEnnBackend::new(
            dir.path().to_path_buf(),
            array![[2.0, 4.0], [4.0, 8.0], [6.0, 2.0]],
            array![[0.0], [1.0], [2.0]],
            None,
            true,
            Array1::from_elem(2, 2.0),
            IndexDriver::HNSWDisk,
        )
        .unwrap();
        assert_eq!(backend.indexed_rows(), 0);
        backend
            .ensure_index_sync(true, &Array1::from_elem(2, 2.0))
            .unwrap();
        assert_eq!(backend.indexed_rows(), backend.len());
        let (_, idx) = backend
            .search(&array![[2.0, 2.0]].view(), 1, false)
            .unwrap();
        assert_eq!(idx.shape(), [1, 1]);
    }

    #[test]
    fn disk_hnsw_scale_x_brute_force_match() {
        let dir = TempDir::new().expect("tempdir");
        let train_x = array![[2.0, 4.0], [4.0, 8.0], [6.0, 2.0]];
        let mut backend = DiskHnswEnnBackend::new(
            dir.path().to_path_buf(),
            train_x,
            array![[0.0], [1.0], [2.0]],
            None,
            true,
            Array1::from_elem(2, 2.0),
            IndexDriver::HNSWDisk,
        )
        .unwrap();
        backend.mark_index_stale();
        backend
            .ensure_index_sync(true, &Array1::from_elem(2, 2.0))
            .unwrap();
        let query = array![[2.0, 2.0]];
        let (_, idx) = backend.search(&query.view(), 1, false).unwrap();
        let vecs: Vec<Vec<f32>> = (0..3)
            .map(|i| {
                backend
                    .train_x
                    .mmap_row_slice(i)
                    .unwrap()
                    .iter()
                    .map(|&v| (v / 2.0) as f32)
                    .collect()
            })
            .collect();
        let q: Vec<f32> = query.row(0).iter().map(|&v| (v / 2.0) as f32).collect();
        let bf = brute_force_topk(&vecs, &q, 1);
        assert_eq!(idx[[0, 0]] as u32, bf[0].0);
    }
}
