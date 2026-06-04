//! KNN backends behind [`crate::index::ENNIndex`] (Faiss in-memory only).

pub(crate) mod faiss_backend;
pub(crate) use faiss_backend::MmapColumnStore;

use ndarray::{Array2, ArrayView2};
use std::sync::Mutex;

use crate::index::{IndexDriver, IndexError};

pub(crate) use faiss_backend::FaissBackend;

/// In-memory Faiss KNN storage (Exact / HNSW).
pub(crate) enum KnnBackend {
    Faiss(Mutex<FaissBackend>),
}

impl KnnBackend {
    pub(crate) fn new(
        num_dim: usize,
        driver: IndexDriver,
        train_scaled: &ArrayView2<f64>,
    ) -> Result<Self, IndexError> {
        match driver {
            IndexDriver::Exact | IndexDriver::HNSW => Ok(Self::Faiss(Mutex::new(FaissBackend::new(
                num_dim,
                driver,
                train_scaled,
            )?))),
            IndexDriver::HNSWDisk => Err(IndexError::InvalidParameter(
                "IndexDriver::HNSWDisk is disk-only; use DiskHnswEnnBackend".to_string(),
            )),
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Faiss(inner) => inner.lock().expect("knn mutex poisoned").len(),
        }
    }

    pub(crate) fn memory_usage_bytes(&self) -> usize {
        match self {
            Self::Faiss(inner) => inner
                .lock()
                .expect("knn mutex poisoned")
                .memory_usage_bytes(),
        }
    }

    pub(crate) fn rebuild(&self, train_scaled: &ArrayView2<f64>) -> Result<(), IndexError> {
        match self {
            Self::Faiss(inner) => inner
                .lock()
                .expect("knn mutex poisoned")
                .rebuild(train_scaled),
        }
    }

    pub(crate) fn add(&self, rows_scaled: &ArrayView2<f64>, start_key: u64) -> Result<(), IndexError> {
        match self {
            Self::Faiss(inner) => inner
                .lock()
                .expect("knn mutex poisoned")
                .add(rows_scaled, start_key),
        }
    }

    pub(crate) fn search(
        &self,
        queries_scaled: &ArrayView2<f64>,
        k_eff: usize,
        search_k: usize,
    ) -> Result<(Array2<f64>, Array2<i64>), IndexError> {
        match self {
            Self::Faiss(inner) => inner
                .lock()
                .expect("knn mutex poisoned")
                .search(queries_scaled, k_eff, search_k),
        }
    }
}

pub(crate) fn arr2_rows_to_f32(a: &ArrayView2<f64>) -> Vec<f32> {
    a.iter().map(|v| *v as f32).collect()
}

pub(crate) fn pad_neighbor_cols_to_search_k(
    dist2s: Array2<f64>,
    idx: Array2<i64>,
    search_k: usize,
) -> (Array2<f64>, Array2<i64>) {
    use ndarray::{concatenate, Axis};
    let k_eff = dist2s.ncols();
    if k_eff >= search_k {
        return (dist2s, idx);
    }
    let n_query = dist2s.nrows();
    if k_eff == 0 {
        return (
            Array2::from_elem((n_query, search_k), f64::INFINITY),
            Array2::zeros((n_query, search_k)),
        );
    }
    let pad_w = search_k - k_eff;
    let pad_dist = Array2::from_elem((n_query, pad_w), f64::INFINITY);
    let far = idx.slice(ndarray::s![.., k_eff - 1..k_eff]).to_owned();
    let mut pad_idx = Array2::zeros((n_query, pad_w));
    for j in 0..pad_w {
        pad_idx.column_mut(j).assign(&far.column(0));
    }
    (
        concatenate![Axis(1), dist2s.view(), pad_dist.view()],
        concatenate![Axis(1), idx.view(), pad_idx.view()],
    )
}

pub(crate) fn unpack_batch_search(
    n_query: usize,
    k: usize,
    distances: &[f32],
    labels: &[i64],
) -> (Array2<f64>, Array2<i64>) {
    let mut dist2s = Array2::zeros((n_query, k));
    let mut indices = Array2::zeros((n_query, k));
    for i in 0..n_query {
        for j in 0..k {
            let o = i * k + j;
            dist2s[[i, j]] = f64::from(distances[o]);
            indices[[i, j]] = labels[o];
        }
    }
    (dist2s, indices)
}

#[cfg(test)]
mod knn_backend_tests {
    use super::*;
    use crate::index::IndexDriver;
    use ndarray::array;

    #[test]
    fn knn_backend_faiss_exact() {
        let train = array![[0.0, 0.0], [1.0, 1.0]];
        let backend = KnnBackend::new(2, IndexDriver::Exact, &train.view()).unwrap();
        assert_eq!(backend.len(), 2);
        backend.add(&array![[2.0, 2.0]].view(), 2).unwrap();
        assert_eq!(backend.len(), 3);
        let (_d, i) = backend
            .search(&array![[0.0, 0.0]].view(), 2, 2)
            .unwrap();
        assert_eq!(i[[0, 0]], 0);
        backend.rebuild(&train.view()).unwrap();
    }

    #[test]
    fn knn_backend_faiss_hnsw() {
        let train = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let backend = KnnBackend::new(2, IndexDriver::HNSW, &train.view()).unwrap();
        assert_eq!(backend.len(), 3);
    }

    #[test]
    fn knn_backend_hnsw_disk_driver_errors() {
        let train = array![[0.0, 0.0], [1.0, 0.0]];
        match KnnBackend::new(2, IndexDriver::HNSWDisk, &train.view()) {
            Err(e) => assert!(e.to_string().contains("disk-only")),
            Ok(_) => panic!("expected HNSWDisk on KnnBackend to error"),
        }
    }

    #[test]
    fn pad_and_unpack_helpers() {
        let (d, i) = pad_neighbor_cols_to_search_k(array![[1.0]], array![[0i64]], 3);
        assert_eq!(d.ncols(), 3);
        assert_eq!(i.ncols(), 3);
        let (d2, i2) = unpack_batch_search(1, 2, &[0.5, 1.5], &[0, 1]);
        assert_eq!(d2[[0, 1]], 1.5);
        assert_eq!(i2[[0, 1]], 1);
    }
}
