use faiss::error::Error as FaissError;
use faiss::index::IndexImpl;
use faiss::{index_factory, Index, MetricType};
use ndarray::ArrayView2;

use super::{arr2_rows_to_f32, pad_neighbor_cols_to_search_k, unpack_batch_search};
use crate::index::{IndexDriver, IndexError};

pub(crate) struct FaissBackend {
    inner: IndexImpl,
    num_dim: usize,
    driver: IndexDriver,
}

fn faiss_spec(driver: IndexDriver) -> &'static str {
    match driver {
        IndexDriver::Exact => "Flat",
        IndexDriver::HNSW => "HNSW32",
    }
}

fn faiss_map_err(e: FaissError) -> IndexError {
    IndexError::InvalidParameter(e.to_string())
}

impl FaissBackend {
    pub(crate) fn new(
        num_dim: usize,
        driver: IndexDriver,
        train_scaled: &ArrayView2<f64>,
    ) -> Result<Self, IndexError> {
        let inner = Self::make_index(num_dim, driver, train_scaled)?;
        Ok(Self {
            inner,
            num_dim,
            driver,
        })
    }

    fn make_index(
        num_dim: usize,
        driver: IndexDriver,
        train_scaled: &ArrayView2<f64>,
    ) -> Result<IndexImpl, IndexError> {
        let mut index =
            index_factory(num_dim as u32, faiss_spec(driver), MetricType::L2).map_err(faiss_map_err)?;
        if train_scaled.nrows() > 0 {
            let data = arr2_rows_to_f32(train_scaled);
            index.add(&data).map_err(faiss_map_err)?;
        }
        Ok(index)
    }

    pub(crate) fn len(&self) -> usize {
        self.inner.ntotal() as usize
    }

    pub(crate) fn rebuild(&mut self, train_scaled: &ArrayView2<f64>) -> Result<(), IndexError> {
        self.inner = Self::make_index(self.num_dim, self.driver, train_scaled)?;
        Ok(())
    }

    pub(crate) fn add(
        &mut self,
        rows_scaled: &ArrayView2<f64>,
        _start_key: u64,
    ) -> Result<(), IndexError> {
        let data = arr2_rows_to_f32(rows_scaled);
        self.inner.add(&data).map_err(faiss_map_err)
    }

    pub(crate) fn search(
        &mut self,
        queries_scaled: &ArrayView2<f64>,
        k_eff: usize,
        search_k: usize,
    ) -> Result<(ndarray::Array2<f64>, ndarray::Array2<i64>), IndexError> {
        let n_query = queries_scaled.nrows();
        let q = arr2_rows_to_f32(queries_scaled);
        let res = self.inner.search(&q, k_eff).map_err(faiss_map_err)?;
        let labels: Vec<i64> = res.labels.iter().map(|l| l.to_native()).collect();
        let (d, i) = unpack_batch_search(n_query, k_eff, &res.distances, &labels);
        Ok(pad_neighbor_cols_to_search_k(d, i, search_k))
    }
}

#[cfg(test)]
mod faiss_backend_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn faiss_backend_exact_roundtrip() {
        let train = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let mut backend = FaissBackend::new(2, IndexDriver::Exact, &train.view()).unwrap();
        assert_eq!(backend.len(), 3);
        backend
            .add(&array![[1.0, 1.0]].view(), 3)
            .unwrap();
        assert_eq!(backend.len(), 4);
        let (d, i) = backend
            .search(&array![[0.0, 0.0]].view(), 2, 2)
            .unwrap();
        assert_eq!(i[[0, 0]], 0);
        assert!(d[[0, 0]] < 1e-5);
        backend.rebuild(&train.view()).unwrap();
        assert_eq!(backend.len(), 3);
    }

    #[test]
    fn faiss_spec_and_map_err() {
        assert_eq!(faiss_spec(IndexDriver::HNSW), "HNSW32");
        let err = faiss_map_err(faiss::error::Error::IndexDescription);
        assert!(matches!(err, IndexError::InvalidParameter(_)));
    }

    #[test]
    fn faiss_backend_hnsw_search() {
        let train = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let mut backend = FaissBackend::new(2, IndexDriver::HNSW, &train.view()).unwrap();
        let (_d, i) = backend
            .search(&array![[0.0, 0.0]].view(), 2, 2)
            .unwrap();
        assert_eq!(i[[0, 0]], 0);
    }
}

#[cfg(test)]
pub(crate) fn faiss_spec_for_test(driver: IndexDriver) -> &'static str {
    faiss_spec(driver)
}

#[cfg(test)]
pub(crate) fn faiss_map_err_for_test(e: FaissError) -> IndexError {
    faiss_map_err(e)
}

#[cfg(test)]
pub(crate) fn make_faiss_for_test(
    num_dim: usize,
    driver: IndexDriver,
    train_scaled: &ArrayView2<f64>,
) -> Result<IndexImpl, IndexError> {
    FaissBackend::make_index(num_dim, driver, train_scaled)
}
