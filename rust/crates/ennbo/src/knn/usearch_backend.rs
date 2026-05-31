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
                return Self::open_mutable(path, num_dim);
            }
        }
        let mut backend = Self::build_in_memory(num_dim)?;
        backend.bulk_add(train_scaled, 0)?;
        backend.index_path = index_path;
        backend.save_if_path()?;
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
        })
    }

    fn build_in_memory(num_dim: usize) -> Result<Self, IndexError> {
        let index = new_index(&usearch_options(num_dim)).map_err(usearch_map_err)?;
        Ok(Self {
            index,
            num_dim,
            index_path: None,
            view_only: false,
        })
    }

    fn ensure_mutable(&mut self) -> Result<(), IndexError> {
        if !self.view_only {
            return Ok(());
        }
        let path = self
            .index_path
            .clone()
            .ok_or_else(|| IndexError::InvalidParameter("view-only index has no path".to_string()))?;
        *self = Self::open_mutable(&path, self.num_dim)?;
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
        *self = Self::build_in_memory(self.num_dim)?;
        self.index_path = path;
        self.bulk_add(train_scaled, 0)?;
        self.save_if_path()
    }

    pub(crate) fn add(
        &mut self,
        rows_scaled: &ArrayView2<f64>,
        start_key: u64,
    ) -> Result<(), IndexError> {
        self.bulk_add(rows_scaled, start_key)?;
        self.save_if_path()
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

    pub(crate) fn save_atomic(&self, path: &Path) -> Result<(), IndexError> {
        if self.view_only {
            return Ok(());
        }
        atomic_save(&self.index, path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

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
    fn usearch_incremental_add_persists_on_disk() {
        let (_dir, path) = temp_index_dir();
        let _ = std::fs::remove_file(&path);
        let mut backend =
            USearchBackend::new(2, &array![[0.0, 0.0]].view(), Some(path.clone())).unwrap();
        backend
            .add(&array![[1.0, 1.0]].view(), 1)
            .unwrap();
        let empty = ndarray::Array2::<f64>::zeros((0, 2));
        let reopened = USearchBackend::new(2, &empty.view(), Some(path.clone())).unwrap();
        assert_eq!(reopened.len(), 2);
        let _ = std::fs::remove_file(&path);
    }
}
