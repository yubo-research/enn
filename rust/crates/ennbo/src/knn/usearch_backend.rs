use std::path::{Path, PathBuf};

use ndarray::ArrayView2;
use usearch::{Index, IndexOptions, MetricKind, ScalarKind, new_index};

use super::{arr2_rows_to_f32, pad_neighbor_cols_to_search_k, unpack_batch_search};
use crate::index::IndexError;

pub(crate) struct USearchBackend {
    index: Index,
    num_dim: usize,
    index_path: Option<PathBuf>,
    view_only: bool,
    /// In-memory tail adds since last successful persist to `index_path`.
    dirty: bool,
}

fn usearch_options(num_dim: usize) -> IndexOptions {
    IndexOptions {
        dimensions: num_dim,
        metric: MetricKind::L2sq,
        quantization: ScalarKind::F32,
        connectivity: 32,
        expansion_add: 0,
        expansion_search: 0,
        multi: false,
    }
}

fn usearch_map_err<E: std::fmt::Display>(e: E) -> IndexError {
    IndexError::InvalidParameter(e.to_string())
}

fn validate_metadata(meta: &usearch::IndexMetadata, num_dim: usize) -> Result<(), IndexError> {
    if meta.dimensions as usize != num_dim {
        return Err(IndexError::InvalidShape {
            expected: num_dim,
            got: meta.dimensions as usize,
        });
    }
    if meta.metric != MetricKind::L2sq {
        return Err(IndexError::InvalidParameter(format!(
            "index metric mismatch: expected L2sq, got {:?}",
            meta.metric
        )));
    }
    if meta.quantization != ScalarKind::F32 {
        return Err(IndexError::InvalidParameter(format!(
            "index scalar mismatch: expected F32, got {:?}",
            meta.quantization
        )));
    }
    Ok(())
}

fn atomic_save(index: &Index, path: &Path) -> Result<(), IndexError> {
    let tmp = path.with_extension("usearch.tmp");
    index
        .save(tmp.to_str().expect("temp path utf-8"))
        .map_err(usearch_map_err)?;
    std::fs::rename(&tmp, path).map_err(|e| IndexError::InvalidParameter(e.to_string()))?;
    Ok(())
}

impl USearchBackend {
    pub(crate) fn new(
        num_dim: usize,
        train_scaled: &ArrayView2<f64>,
        index_path: Option<PathBuf>,
    ) -> Result<Self, IndexError> {
        if let Some(ref path) = index_path {
            if path.exists() {
                if train_scaled.nrows() == 0 {
                    return Self::open_view_only(path, num_dim);
                }
                return Self::open_mutable(path, num_dim);
            }
        }
        let mut backend = Self::build_in_memory(num_dim)?;
        backend.index_path = index_path;
        backend.bulk_add_then_save(train_scaled, 0)?;
        Ok(backend)
    }

    pub(crate) fn open_or_build(
        num_dim: usize,
        train_scaled: &ArrayView2<f64>,
        path: &Path,
    ) -> Result<Self, IndexError> {
        Self::new(num_dim, train_scaled, Some(path.to_path_buf()))
    }

    pub(crate) fn open_view_only(path: &Path, num_dim: usize) -> Result<Self, IndexError> {
        let meta = Index::metadata(path.to_str().expect("path utf-8")).map_err(usearch_map_err)?;
        validate_metadata(&meta, num_dim)?;
        let index = Index::restore_view(path.to_str().expect("path utf-8")).map_err(usearch_map_err)?;
        Ok(Self {
            index,
            num_dim,
            index_path: Some(path.to_path_buf()),
            view_only: true,
            dirty: false,
        })
    }

    fn open_mutable(path: &Path, num_dim: usize) -> Result<Self, IndexError> {
        let meta = Index::metadata(path.to_str().expect("path utf-8")).map_err(usearch_map_err)?;
        validate_metadata(&meta, num_dim)?;
        let index = Index::restore(path.to_str().expect("path utf-8")).map_err(usearch_map_err)?;
        Ok(Self {
            index,
            num_dim,
            index_path: Some(path.to_path_buf()),
            view_only: false,
            dirty: false,
        })
    }

    fn build_in_memory(num_dim: usize) -> Result<Self, IndexError> {
        let index = new_index(&usearch_options(num_dim)).map_err(usearch_map_err)?;
        Ok(Self {
            index,
            num_dim,
            index_path: None,
            view_only: false,
            dirty: false,
        })
    }

    /// Materialize a mutable in-memory index from a view-only mmap handle without
    /// re-reading the checkpoint from disk (`Index::restore`).
    fn materialize_from_view(view_index: &Index, num_dim: usize) -> Result<Index, IndexError> {
        let len = view_index.serialized_length();
        let mut buf = vec![0u8; len];
        view_index
            .save_to_buffer(&mut buf)
            .map_err(usearch_map_err)?;
        let meta = Index::metadata_from_buffer(&buf).map_err(usearch_map_err)?;
        validate_metadata(&meta, num_dim)?;
        Index::restore_from_buffer(&buf).map_err(usearch_map_err)
    }

    fn ensure_mutable(&mut self) -> Result<(), IndexError> {
        if !self.view_only {
            return Ok(());
        }
        let num_dim = self.num_dim;
        let index_path = self
            .index_path
            .clone()
            .ok_or_else(|| IndexError::InvalidParameter("view-only index has no path".to_string()))?;
        let index = Self::materialize_from_view(&self.index, num_dim)?;
        self.index = index;
        self.view_only = false;
        self.index_path = Some(index_path);
        Ok(())
    }

    fn bulk_add(&mut self, rows: &ArrayView2<f64>, start_key: u64) -> Result<(), IndexError> {
        let dim = rows.ncols();
        if dim != self.num_dim {
            return Err(IndexError::InvalidShape {
                expected: self.num_dim,
                got: dim,
            });
        }
        let n = rows.nrows();
        if n == 0 {
            return Ok(());
        }
        self.ensure_mutable()?;
        self.index
            .reserve(self.index.size() + n)
            .map_err(usearch_map_err)?;
        let data = arr2_rows_to_f32(rows);
        for (i, chunk) in data.chunks(dim).enumerate() {
            let key = start_key + i as u64;
            self.index.add(key, chunk).map_err(usearch_map_err)?;
        }
        Ok(())
    }

    fn save_if_path(&self) -> Result<(), IndexError> {
        if let Some(ref path) = self.index_path {
            if self.view_only {
                return Ok(());
            }
            atomic_save(&self.index, path)
        } else {
            Ok(())
        }
    }

    /// Restore in-memory state from the on-disk checkpoint after a failed persist.
    /// Falls back to an empty in-memory index when the checkpoint is missing or unreadable.
    fn reload_from_disk_checkpoint(&mut self) -> Result<(), IndexError> {
        let num_dim = self.num_dim;
        let path = self.index_path.clone();
        if let Some(ref path) = path {
            if path.exists() {
                if let Ok(restored) = Self::open_mutable(path, num_dim) {
                    *self = restored;
                    return Ok(());
                }
            }
        }
        *self = Self::build_in_memory(num_dim)?;
        self.index_path = path;
        Ok(())
    }

    fn persist_to_disk(&mut self) -> Result<(), IndexError> {
        if let Err(e) = self.save_if_path() {
            let _ = self.reload_from_disk_checkpoint();
            return Err(e);
        }
        self.dirty = false;
        Ok(())
    }

    fn bulk_add_then_save(
        &mut self,
        rows: &ArrayView2<f64>,
        start_key: u64,
    ) -> Result<(), IndexError> {
        self.bulk_add(rows, start_key)?;
        self.persist_to_disk()
    }

    pub(crate) fn len(&self) -> usize {
        self.index.size()
    }

    pub(crate) fn rebuild(
        &mut self,
        train_scaled: &ArrayView2<f64>,
        index_path: Option<&Path>,
    ) -> Result<(), IndexError> {
        let path = index_path
            .map(|p| p.to_path_buf())
            .or_else(|| self.index_path.clone());
        self.view_only = false;
        if train_scaled.nrows() == 0 {
            if let Some(ref p) = path {
                if p.exists() {
                    *self = Self::open_view_only(p, self.num_dim)?;
                    self.index_path = path;
                    return Ok(());
                }
            }
            *self = Self::build_in_memory(self.num_dim)?;
            self.index_path = path;
            return Ok(());
        }
        *self = Self::build_in_memory(self.num_dim)?;
        self.index_path = path;
        self.bulk_add_then_save(train_scaled, 0)
    }

    pub(crate) fn add(
        &mut self,
        rows_scaled: &ArrayView2<f64>,
        start_key: u64,
    ) -> Result<(), IndexError> {
        self.bulk_add(rows_scaled, start_key)?;
        self.dirty = true;
        Ok(())
    }

    /// Atomically persist in-memory tail adds to `index_path` when dirty.
    pub(crate) fn checkpoint(&mut self) -> Result<(), IndexError> {
        if !self.dirty {
            return Ok(());
        }
        self.persist_to_disk()
    }

    pub(crate) fn is_dirty(&self) -> bool {
        self.dirty
    }

    pub(crate) fn search(
        &self,
        queries_scaled: &ArrayView2<f64>,
        k_eff: usize,
        search_k: usize,
    ) -> Result<(ndarray::Array2<f64>, ndarray::Array2<i64>), IndexError> {
        let n_query = queries_scaled.nrows();
        let dim = queries_scaled.ncols();
        if dim != self.num_dim {
            return Err(IndexError::InvalidShape {
                expected: self.num_dim,
                got: dim,
            });
        }
        let q = arr2_rows_to_f32(queries_scaled);
        let mut distances = Vec::with_capacity(n_query * k_eff);
        let mut labels = Vec::with_capacity(n_query * k_eff);
        for i in 0..n_query {
            let query = &q[i * dim..(i + 1) * dim];
            let matches = self.index.search(query, k_eff).map_err(usearch_map_err)?;
            let got = matches.keys.len();
            for j in 0..k_eff {
                if j < got {
                    labels.push(matches.keys[j] as i64);
                    distances.push(matches.distances[j]);
                } else {
                    labels.push(-1);
                    distances.push(f32::INFINITY);
                }
            }
        }
        let (d, idx) = unpack_batch_search(n_query, k_eff, &distances, &labels);
        Ok(pad_neighbor_cols_to_search_k(d, idx, search_k))
    }

    pub(crate) fn save_atomic(&mut self, path: &Path) -> Result<(), IndexError> {
        if self.view_only {
            return Ok(());
        }
        atomic_save(&self.index, path)?;
        if self.index_path.as_deref() == Some(path) {
            self.dirty = false;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    /// Unique directory per test so nextest parallel runs do not share index files.
    fn temp_index_dir() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("index.usearch");
        (dir, path)
    }

    #[test]
    fn usearch_rebuild_and_incremental_add() {
        let (_dir, path) = temp_index_dir();
        let _ = std::fs::remove_file(&path);
        let train = array![[0.0, 0.0], [1.0, 0.0]];
        let mut backend = USearchBackend::new(2, &train.view(), None).unwrap();
        backend
            .add(&array![[0.0, 1.0]].view(), 2)
            .unwrap();
        assert_eq!(backend.len(), 3);
        backend.rebuild(&train.view(), Some(&path)).unwrap();
        assert!(path.exists());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn usearch_options_and_map_err() {
        let opts = usearch_options(10);
        assert_eq!(opts.dimensions, 10);
        assert_eq!(opts.metric, MetricKind::L2sq);
        assert_eq!(opts.quantization, ScalarKind::F32);
        assert_eq!(opts.connectivity, 32);
        let err = usearch_map_err("usearch failure");
        assert!(matches!(
            err,
            IndexError::InvalidParameter(ref s) if s == "usearch failure"
        ));
    }

    #[test]
    fn validate_metadata_rejects_wrong_dim() {
        let (_dir, path) = temp_index_dir();
        let _ = std::fs::remove_file(&path);
        let train = array![[0.0, 0.0], [1.0, 0.0]];
        USearchBackend::new(2, &train.view(), Some(path.clone())).unwrap();
        let meta = Index::metadata(path.to_str().unwrap()).unwrap();
        assert!(validate_metadata(&meta, 2).is_ok());
        assert!(matches!(
            validate_metadata(&meta, 3),
            Err(IndexError::InvalidShape { .. })
        ));
        assert!(matches!(
            USearchBackend::open_view_only(&path, 3),
            Err(IndexError::InvalidShape { .. })
        ));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn atomic_save_roundtrip() {
        let (_dir, path) = temp_index_dir();
        let _ = std::fs::remove_file(&path);
        let train = array![[0.0, 0.0], [1.0, 0.0]];
        let backend = USearchBackend::new(2, &train.view(), None).unwrap();
        atomic_save(&backend.index, &path).unwrap();
        assert!(path.exists());
        assert!(Index::metadata(path.to_str().unwrap()).is_ok());
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn usearch_save_restore_view_search() {
        let (_dir, path) = temp_index_dir();
        let _ = std::fs::remove_file(&path);
        let train = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let built = USearchBackend::new(2, &train.view(), Some(path.clone())).unwrap();
        assert_eq!(built.len(), 4);
        let viewed = USearchBackend::open_view_only(&path, 2).unwrap();
        let (d, i) = viewed.search(&array![[0.0, 0.0]].view(), 2, 2).unwrap();
        assert_eq!(i[[0, 0]], 0);
        assert!(d[[0, 0]] < 1e-5);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn usearch_add_rolls_back_memory_when_checkpoint_fails() {
        let (dir, path) = temp_index_dir();
        let _ = std::fs::remove_file(&path);
        let mut backend =
            USearchBackend::new(2, &array![[0.0, 0.0]].view(), Some(path.clone())).unwrap();
        assert_eq!(backend.len(), 1);
        backend
            .add(&array![[1.0, 1.0]].view(), 1)
            .unwrap();
        assert_eq!(backend.len(), 2);
        assert!(backend.is_dirty());

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(dir.path())
                .expect("dir metadata")
                .permissions();
            perms.set_mode(0o555);
            std::fs::set_permissions(dir.path(), perms).expect("chmod read-only");

            let result = backend.checkpoint();
            assert!(result.is_err(), "checkpoint should fail on read-only directory");
            assert_eq!(
                backend.len(),
                1,
                "in-memory index must roll back when checkpoint fails"
            );

            let empty = ndarray::Array2::<f64>::zeros((0, 2));
            let reopened = USearchBackend::new(2, &empty.view(), Some(path.clone())).unwrap();
            assert_eq!(
                reopened.len(),
                1,
                "on-disk checkpoint must not include uncommitted add"
            );

            let mut restore = std::fs::metadata(dir.path())
                .expect("dir metadata")
                .permissions();
            restore.set_mode(0o755);
            std::fs::set_permissions(dir.path(), restore).expect("restore write permission");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn usearch_empty_rebuild_preserves_disk_checkpoint() {
        let (_dir, path) = temp_index_dir();
        let _ = std::fs::remove_file(&path);
        let train = array![[0.0, 0.0], [1.0, 0.0]];
        let mut backend = USearchBackend::new(2, &train.view(), Some(path.clone())).unwrap();
        assert_eq!(backend.len(), 2);

        let empty = ndarray::Array2::<f64>::zeros((0, 2));
        backend.rebuild(&empty.view(), Some(&path)).unwrap();
        assert_eq!(
            backend.len(),
            2,
            "empty rebuild must reload existing checkpoint, not wipe memory"
        );

        let reopened = USearchBackend::new(2, &empty.view(), Some(path.clone())).unwrap();
        assert_eq!(
            reopened.len(),
            2,
            "empty rebuild must not overwrite on-disk checkpoint"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn usearch_checkpoint_clears_memory_when_reload_fails() {
        let (dir, path) = temp_index_dir();
        let _ = std::fs::remove_file(&path);
        let mut backend =
            USearchBackend::new(2, &array![[0.0, 0.0]].view(), Some(path.clone())).unwrap();
        assert_eq!(backend.len(), 1);
        backend
            .add(&array![[1.0, 1.0]].view(), 1)
            .unwrap();
        std::fs::write(&path, b"corrupt").expect("corrupt checkpoint");

        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let mut perms = std::fs::metadata(dir.path())
                .expect("dir metadata")
                .permissions();
            perms.set_mode(0o555);
            std::fs::set_permissions(dir.path(), perms).expect("chmod read-only");

            let result = backend.checkpoint();
            assert!(result.is_err(), "checkpoint should fail on read-only directory");
            assert_ne!(
                backend.len(),
                2,
                "must not retain partial add when reload fails"
            );

            let mut restore = std::fs::metadata(dir.path())
                .expect("dir metadata")
                .permissions();
            restore.set_mode(0o755);
            std::fs::set_permissions(dir.path(), restore).expect("restore write permission");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn usearch_incremental_add_defers_disk_until_checkpoint() {
        let (_dir, path) = temp_index_dir();
        let _ = std::fs::remove_file(&path);
        let mut backend =
            USearchBackend::new(2, &array![[0.0, 0.0]].view(), Some(path.clone())).unwrap();
        let size_after_build = std::fs::metadata(&path).expect("checkpoint exists").len();

        backend
            .add(&array![[1.0, 1.0]].view(), 1)
            .unwrap();
        assert_eq!(backend.len(), 2);
        assert!(backend.is_dirty());

        let empty = ndarray::Array2::<f64>::zeros((0, 2));
        let reopened = USearchBackend::new(2, &empty.view(), Some(path.clone())).unwrap();
        assert_eq!(
            reopened.len(),
            1,
            "incremental add must not persist until checkpoint"
        );
        assert_eq!(
            std::fs::metadata(&path).expect("checkpoint exists").len(),
            size_after_build,
            "checkpoint file size must be unchanged before explicit persist"
        );

        backend.checkpoint().unwrap();
        assert!(!backend.is_dirty());
        let reopened = USearchBackend::new(2, &empty.view(), Some(path.clone())).unwrap();
        assert_eq!(reopened.len(), 2);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn usearch_cold_open_empty_train_uses_mmap_view() {
        let (_dir, path) = temp_index_dir();
        let _ = std::fs::remove_file(&path);
        let train = array![[0.0, 0.0], [1.0, 0.0]];
        USearchBackend::new(2, &train.view(), Some(path.clone())).unwrap();
        let empty = ndarray::Array2::<f64>::zeros((0, 2));
        let reopened = USearchBackend::new(2, &empty.view(), Some(path.clone())).unwrap();
        assert!(
            reopened.view_only,
            "search-first reopen must mmap checkpoint, not full restore"
        );
        assert_eq!(reopened.len(), 2);
        let (d, i) = reopened.search(&array![[0.0, 0.0]].view(), 1, 1).unwrap();
        assert_eq!(i[[0, 0]], 0);
        assert!(d[[0, 0]] < 1e-5);

        let mut writable = reopened;
        writable
            .add(&array![[1.0, 1.0]].view(), 2)
            .unwrap();
        assert!(
            !writable.view_only,
            "first tail add must materialize a mutable index"
        );
        assert_eq!(writable.len(), 3);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn usearch_incremental_add_search_matches_after_checkpoint_reopen() {
        let (_dir, path) = temp_index_dir();
        let _ = std::fs::remove_file(&path);
        let train = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let mut backend = USearchBackend::new(2, &train.view(), Some(path.clone())).unwrap();
        backend
            .add(&array![[1.0, 1.0]].view(), 3)
            .unwrap();

        let query = array![[0.0, 0.0]];
        let (d_before, i_before) = backend.search(&query.view(), 2, 2).unwrap();

        backend.checkpoint().unwrap();

        let viewed = USearchBackend::open_view_only(&path, 2).unwrap();
        let (d_after, i_after) = viewed.search(&query.view(), 2, 2).unwrap();
        assert_eq!(i_before[[0, 0]], i_after[[0, 0]]);
        assert!((d_before[[0, 0]] - d_after[[0, 0]]).abs() < 1e-4);
        let _ = std::fs::remove_file(&path);
    }

    const COLD_OPEN_DIM: usize = 10;
    const COLD_OPEN_N_LARGE: usize = 50_000;
    const COLD_OPEN_N_SMALL: usize = 10_000;
    /// Conservative vs measured 5.2× at N=50k (prior KPop session).
    const MIN_RESTORE_VS_MMAP_RATIO: f64 = 2.0;
    const COLD_OPEN_WARMUP: usize = 3;
    const COLD_OPEN_REPS: usize = 7;
    const COLD_OPEN_MAX_ATTEMPTS: usize = 10;

    fn deterministic_train(n: usize, dim: usize) -> ndarray::Array2<f64> {
        ndarray::Array2::from_shape_fn((n, dim), |(i, j)| {
            let fi = i as f64;
            let fj = j as f64;
            (fi * 0.001 + fj * 0.017).sin()
        })
    }

    fn build_golden_checkpoint(path: &Path, n: usize, dim: usize) {
        let train = deterministic_train(n, dim);
        let _ = USearchBackend::new(dim, &train.view(), Some(path.to_path_buf()))
            .unwrap_or_else(|e| panic!("build golden checkpoint n={n} dim={dim}: {e}"));
    }

    fn time_mmap_cold_open(path: &Path, dim: usize) -> Duration {
        let empty = ndarray::Array2::<f64>::zeros((0, dim));
        let start = Instant::now();
        let backend =
            USearchBackend::new(dim, &empty.view(), Some(path.to_path_buf())).unwrap();
        assert!(
            backend.view_only,
            "empty-train cold open must use mmap view, not full restore"
        );
        black_box(backend.len());
        start.elapsed()
    }

    fn time_full_restore_open(path: &Path, dim: usize) -> Duration {
        let start = Instant::now();
        let backend = USearchBackend::open_mutable(path, dim).unwrap();
        assert!(
            !backend.view_only,
            "open_mutable must materialize a writable in-memory graph"
        );
        black_box(backend.len());
        start.elapsed()
    }

    fn time_view_then_disk_restore(path: &Path, dim: usize) -> Duration {
        let empty = ndarray::Array2::<f64>::zeros((0, dim));
        let start = Instant::now();
        let backend =
            USearchBackend::new(dim, &empty.view(), Some(path.to_path_buf())).unwrap();
        assert!(backend.view_only);
        let upgraded = USearchBackend::open_mutable(path, dim).unwrap();
        black_box(upgraded.len());
        start.elapsed()
    }

    fn time_view_then_materialize(path: &Path, dim: usize) -> Duration {
        let empty = ndarray::Array2::<f64>::zeros((0, dim));
        let start = Instant::now();
        let mut backend =
            USearchBackend::new(dim, &empty.view(), Some(path.to_path_buf())).unwrap();
        assert!(backend.view_only);
        let row = ndarray::Array2::zeros((1, dim));
        backend.add(&row.view(), backend.len() as u64).unwrap();
        assert!(!backend.view_only);
        black_box(backend.len());
        start.elapsed()
    }

    fn median_disk_restore_vs_materialize_ratio(path: &Path, dim: usize) -> f64 {
        for i in 0..COLD_OPEN_WARMUP {
            if i % 2 == 0 {
                black_box(time_view_then_disk_restore(path, dim));
                black_box(time_view_then_materialize(path, dim));
            } else {
                black_box(time_view_then_materialize(path, dim));
                black_box(time_view_then_disk_restore(path, dim));
            }
        }
        let mut ratios = Vec::with_capacity(COLD_OPEN_REPS);
        for i in 0..COLD_OPEN_REPS {
            let (t_disk, t_mat) = if i % 2 == 0 {
                let t_disk = time_view_then_disk_restore(path, dim);
                let t_mat = time_view_then_materialize(path, dim);
                (t_disk, t_mat)
            } else {
                let t_mat = time_view_then_materialize(path, dim);
                let t_disk = time_view_then_disk_restore(path, dim);
                (t_disk, t_mat)
            };
            let denom = t_mat.as_secs_f64().max(1e-9);
            ratios.push(t_disk.as_secs_f64() / denom);
        }
        let mut sorted = ratios;
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("timing ratios must be ordered"));
        sorted[sorted.len() / 2]
    }

    fn assert_disk_restore_slower_than_materialize(path: &Path, n: usize, dim: usize) {
        /// After mmap cold open, upgrading via disk `restore` should cost more than
        /// materializing from the live view handle (regression guard for ensure_mutable).
        const MIN_DISK_VS_MATERIALIZE_RATIO: f64 = 1.15;
        let mut last_ratio = 0.0_f64;
        for attempt in 0..COLD_OPEN_MAX_ATTEMPTS {
            let ratio = median_disk_restore_vs_materialize_ratio(path, dim);
            last_ratio = ratio;
            if ratio >= MIN_DISK_VS_MATERIALIZE_RATIO {
                eprintln!(
                    "usearch ensure_mutable speedup n={n} dim={dim}: disk-restore/materialize={ratio:.2}x \
                     (min {MIN_DISK_VS_MATERIALIZE_RATIO}x, attempt {})",
                    attempt + 1
                );
                return;
            }
            assert!(
                attempt + 1 != COLD_OPEN_MAX_ATTEMPTS,
                "disk-restore/materialize ratio {ratio:.2}x below min {MIN_DISK_VS_MATERIALIZE_RATIO}x \
                 at n={n} dim={dim} after {COLD_OPEN_MAX_ATTEMPTS} attempts",
            );
        }
        panic!(
            "disk-restore/materialize ratio {last_ratio:.2}x below min {MIN_DISK_VS_MATERIALIZE_RATIO}x \
             at n={n} dim={dim}",
        );
    }

    fn median_restore_vs_mmap_ratio(path: &Path, dim: usize) -> f64 {
        for i in 0..COLD_OPEN_WARMUP {
            if i % 2 == 0 {
                black_box(time_full_restore_open(path, dim));
                black_box(time_mmap_cold_open(path, dim));
            } else {
                black_box(time_mmap_cold_open(path, dim));
                black_box(time_full_restore_open(path, dim));
            }
        }
        let mut ratios = Vec::with_capacity(COLD_OPEN_REPS);
        for i in 0..COLD_OPEN_REPS {
            let (t_restore, t_mmap) = if i % 2 == 0 {
                let t_restore = time_full_restore_open(path, dim);
                let t_mmap = time_mmap_cold_open(path, dim);
                (t_restore, t_mmap)
            } else {
                let t_mmap = time_mmap_cold_open(path, dim);
                let t_restore = time_full_restore_open(path, dim);
                (t_restore, t_mmap)
            };
            let denom = t_mmap.as_secs_f64().max(1e-9);
            ratios.push(t_restore.as_secs_f64() / denom);
        }
        let mut sorted = ratios;
        sorted.sort_by(|a, b| a.partial_cmp(b).expect("timing ratios must be ordered"));
        sorted[sorted.len() / 2]
    }

    fn assert_restore_slower_than_mmap(path: &Path, n: usize, dim: usize) {
        let mut last_ratio = 0.0_f64;
        for attempt in 0..COLD_OPEN_MAX_ATTEMPTS {
            let ratio = median_restore_vs_mmap_ratio(path, dim);
            last_ratio = ratio;
            if ratio >= MIN_RESTORE_VS_MMAP_RATIO {
                eprintln!(
                    "usearch cold-open speedup n={n} dim={dim}: restore/mmap={ratio:.2}x \
                     (min {MIN_RESTORE_VS_MMAP_RATIO}x, attempt {})",
                    attempt + 1
                );
                return;
            }
            assert!(
                attempt + 1 != COLD_OPEN_MAX_ATTEMPTS,
                "restore/mmap ratio {ratio:.2}x below min {MIN_RESTORE_VS_MMAP_RATIO}x \
                 at n={n} dim={dim} after {COLD_OPEN_MAX_ATTEMPTS} attempts",
            );
        }
        panic!(
            "restore/mmap ratio {last_ratio:.2}x below min {MIN_RESTORE_VS_MMAP_RATIO}x \
             at n={n} dim={dim}",
        );
    }

    #[test]
    fn usearch_cold_open_mmap_faster_than_full_restore() {
        let (_dir, path) = temp_index_dir();
        let _ = std::fs::remove_file(&path);
        build_golden_checkpoint(&path, COLD_OPEN_N_LARGE, COLD_OPEN_DIM);
        assert_restore_slower_than_mmap(&path, COLD_OPEN_N_LARGE, COLD_OPEN_DIM);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn usearch_cold_open_mmap_speedup_metamorphic_across_n() {
        for n in [COLD_OPEN_N_SMALL, COLD_OPEN_N_LARGE] {
            let (_dir, path) = temp_index_dir();
            let _ = std::fs::remove_file(&path);
            build_golden_checkpoint(&path, n, COLD_OPEN_DIM);
            assert_restore_slower_than_mmap(&path, n, COLD_OPEN_DIM);
            let _ = std::fs::remove_file(&path);
        }
    }

    #[test]
    fn usearch_cold_open_mmap_speedup_fuzz() {
        use rand::RngCore;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42);
        eprintln!("usearch_cold_open_mmap_speedup_fuzz seed={seed}");
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let n = 15_000 + (rng.next_u64() % 10_000) as usize;
        let (_dir, path) = temp_index_dir();
        let _ = std::fs::remove_file(&path);
        build_golden_checkpoint(&path, n, COLD_OPEN_DIM);
        assert_restore_slower_than_mmap(&path, n, COLD_OPEN_DIM);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn usearch_ensure_mutable_materialize_faster_than_disk_restore() {
        let (_dir, path) = temp_index_dir();
        let _ = std::fs::remove_file(&path);
        build_golden_checkpoint(&path, COLD_OPEN_N_LARGE, COLD_OPEN_DIM);
        assert_disk_restore_slower_than_materialize(&path, COLD_OPEN_N_LARGE, COLD_OPEN_DIM);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn usearch_ensure_mutable_speedup_metamorphic_across_n() {
        for n in [COLD_OPEN_N_SMALL, COLD_OPEN_N_LARGE] {
            let (_dir, path) = temp_index_dir();
            let _ = std::fs::remove_file(&path);
            build_golden_checkpoint(&path, n, COLD_OPEN_DIM);
            assert_disk_restore_slower_than_materialize(&path, n, COLD_OPEN_DIM);
            let _ = std::fs::remove_file(&path);
        }
    }

    #[test]
    fn usearch_ensure_mutable_speedup_fuzz() {
        use rand::RngCore;
        use rand::SeedableRng;
        use rand_chacha::ChaCha8Rng;

        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(42);
        eprintln!("usearch_ensure_mutable_speedup_fuzz seed={seed}");
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let n = 15_000 + (rng.next_u64() % 10_000) as usize;
        let (_dir, path) = temp_index_dir();
        let _ = std::fs::remove_file(&path);
        build_golden_checkpoint(&path, n, COLD_OPEN_DIM);
        assert_disk_restore_slower_than_materialize(&path, n, COLD_OPEN_DIM);
        let _ = std::fs::remove_file(&path);
    }
}
