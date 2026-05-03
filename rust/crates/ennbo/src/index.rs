use faiss::error::Error as FaissError;
use faiss::index::IndexImpl;
use faiss::{index_factory, Idx, Index, MetricType};
use ndarray::{s, Array1, Array2, ArrayView2, Axis};
use std::sync::Mutex;
use thiserror::Error;

#[derive(Error, Debug, Clone, PartialEq)]
pub enum IndexError {
    #[error("Invalid shape: expected {expected} dims, got {got}")]
    InvalidShape { expected: usize, got: usize },
    #[error("Invalid search parameter: {0}")]
    InvalidParameter(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum IndexDriver {
    #[default]
    Exact,
    HNSW,
}

pub struct ENNIndex {
    inner: Mutex<IndexImpl>,
    num_dim: usize,
    x_scale: Array1<f64>,
    scale_x: bool,
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

fn arr2_rows_to_f32(a: &ArrayView2<f64>) -> Vec<f32> {
    a.iter().map(|v| *v as f32).collect()
}

fn make_faiss(
    num_dim: usize,
    driver: IndexDriver,
    train_scaled: &ArrayView2<f64>,
) -> Result<IndexImpl, IndexError> {
    let mut index = index_factory(num_dim as u32, faiss_spec(driver), MetricType::L2)
        .map_err(faiss_map_err)?;
    if train_scaled.nrows() > 0 {
        let data = arr2_rows_to_f32(train_scaled);
        index.add(&data).map_err(faiss_map_err)?;
    }
    Ok(index)
}

fn pad_neighbor_cols_to_search_k(
    dist2s: Array2<f64>,
    idx: Array2<i64>,
    search_k: usize,
) -> (Array2<f64>, Array2<i64>) {
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
    let far = idx.slice(s![.., k_eff - 1..k_eff]).to_owned();
    let mut pad_idx = Array2::zeros((n_query, pad_w));
    for j in 0..pad_w {
        pad_idx.column_mut(j).assign(&far.column(0));
    }
    (
        ndarray::concatenate![Axis(1), dist2s.view(), pad_dist.view()],
        ndarray::concatenate![Axis(1), idx.view(), pad_idx.view()],
    )
}

fn unpack_faiss_search(
    n_query: usize,
    k: usize,
    distances: &[f32],
    labels: &[Idx],
) -> (Array2<f64>, Array2<i64>) {
    let mut dist2s = Array2::zeros((n_query, k));
    let mut indices = Array2::zeros((n_query, k));
    for i in 0..n_query {
        for j in 0..k {
            let o = i * k + j;
            dist2s[[i, j]] = f64::from(distances[o]);
            indices[[i, j]] = labels[o].to_native();
        }
    }
    (dist2s, indices)
}

impl ENNIndex {
    pub fn new(
        train_x_scaled: Array2<f64>,
        num_dim: usize,
        x_scale: Array1<f64>,
        scale_x: bool,
        driver: IndexDriver,
    ) -> Result<Self, IndexError> {
        let inner = Mutex::new(make_faiss(num_dim, driver, &train_x_scaled.view())?);
        Ok(Self {
            inner,
            num_dim,
            x_scale,
            scale_x,
            driver,
        })
    }

    pub fn driver(&self) -> IndexDriver {
        self.driver
    }

    pub fn add(&mut self, x: &ArrayView2<f64>) -> Result<(), IndexError> {
        if x.ncols() != self.num_dim {
            return Err(IndexError::InvalidShape {
                expected: self.num_dim,
                got: x.ncols(),
            });
        }
        let x_scaled: Array2<f64> = if self.scale_x {
            x / &self.x_scale.view().insert_axis(Axis(0))
        } else {
            x.to_owned()
        };
        let data = arr2_rows_to_f32(&x_scaled.view());
        self.inner
            .lock()
            .expect("faiss index mutex poisoned")
            .add(&data)
            .map_err(faiss_map_err)
    }

    pub fn search(
        &self,
        x: &ArrayView2<f64>,
        search_k: i32,
        exclude_nearest: bool,
    ) -> Result<(Array2<f64>, Array2<i64>), IndexError> {
        if search_k <= 0 {
            return Err(IndexError::InvalidParameter(format!(
                "search_k must be > 0, got {search_k}"
            )));
        }
        if x.ncols() != self.num_dim {
            return Err(IndexError::InvalidShape {
                expected: self.num_dim,
                got: x.ncols(),
            });
        }

        let n_train = self
            .inner
            .lock()
            .expect("faiss index mutex poisoned")
            .ntotal() as usize;
        let n_query = x.nrows();
        let search_k = search_k as usize;

        let x_scaled: Array2<f64> = if self.scale_x {
            x / &self.x_scale.view().insert_axis(Axis(0))
        } else {
            x.to_owned()
        };
        let q = arr2_rows_to_f32(&x_scaled.view());

        let (mut dist2s, mut indices) = if n_train == 0 {
            (
                Array2::from_elem((n_query, search_k), f64::INFINITY),
                Array2::zeros((n_query, search_k)),
            )
        } else {
            let k_eff = search_k.min(n_train);
            let res = self
                .inner
                .lock()
                .expect("faiss index mutex poisoned")
                .search(&q, k_eff)
                .map_err(faiss_map_err)?;
            let (d, i) = unpack_faiss_search(n_query, k_eff, &res.distances, &res.labels);
            pad_neighbor_cols_to_search_k(d, i, search_k)
        };

        if exclude_nearest {
            let nc = dist2s.ncols();
            if nc <= 1 {
                dist2s = Array2::zeros((n_query, nc.saturating_sub(1)));
                indices = Array2::zeros((n_query, nc.saturating_sub(1)));
            } else {
                dist2s = dist2s.slice_axis(Axis(1), ndarray::Slice::from(1..)).to_owned();
                indices = indices.slice_axis(Axis(1), ndarray::Slice::from(1..)).to_owned();
            }
        }

        Ok((dist2s, indices))
    }

    pub fn len(&self) -> usize {
        self.inner
            .lock()
            .expect("faiss index mutex poisoned")
            .ntotal() as usize
    }

    pub fn is_empty(&self) -> bool {
        self.inner
            .lock()
            .expect("faiss index mutex poisoned")
            .ntotal()
            == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use ndarray::Array2;

    fn index_unit(train_x: Array2<f64>, driver: IndexDriver) -> ENNIndex {
        let x_scale = array![1.0, 1.0];
        ENNIndex::new(train_x, 2, x_scale, false, driver).unwrap()
    }

    #[test]
    fn test_index_creation() {
        let train_x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let index = index_unit(train_x, IndexDriver::Exact);

        assert_eq!(index.len(), 3);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_index_search() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let index = index_unit(train_x, IndexDriver::Exact);

        let query = array![[0.0, 0.0]];
        let (dist2s, indices) = index.search(&query.view(), 2, false).unwrap();

        assert_eq!(indices[[0, 0]], 0);
        assert!(dist2s[[0, 0]] < 1e-6);

        assert!(dist2s[[0, 1]] > 0.0);
    }

    #[test]
    fn test_index_search_exclude_nearest() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let index = index_unit(train_x, IndexDriver::Exact);

        let query = array![[0.0, 0.0]];
        let (dist2s, indices) = index.search(&query.view(), 2, true).unwrap();

        assert_eq!(dist2s.ncols(), 1);
        assert_eq!(indices.ncols(), 1);

        assert_ne!(indices[[0, 0]], 0);
    }

    #[test]
    fn test_index_add() {
        let mut index = index_unit(array![[0.0, 0.0]], IndexDriver::Exact);

        let new_point = array![[1.0, 1.0]];
        index.add(&new_point.view()).unwrap();

        assert_eq!(index.len(), 2);
    }

    #[test]
    fn test_invalid_search_k() {
        let index = index_unit(array![[0.0, 0.0]], IndexDriver::Exact);

        let query = array![[0.0, 0.0]];
        let result = index.search(&query.view(), 0, false);

        assert!(matches!(result, Err(IndexError::InvalidParameter(_))));
    }

    #[test]
    fn test_invalid_dimensions() {
        let index = index_unit(array![[0.0, 0.0]], IndexDriver::Exact);

        let query = array![[0.0, 0.0, 0.0]];
        let result = index.search(&query.view(), 1, false);

        assert!(matches!(result, Err(IndexError::InvalidShape { expected: 2, got: 3 })));
    }

    #[test]
    fn test_hnsw_search() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let index = index_unit(train_x, IndexDriver::HNSW);

        let query = array![[0.0, 0.0]];
        let (dist2s, indices) = index.search(&query.view(), 2, false).unwrap();

        assert_eq!(indices[[0, 0]], 0);
        assert!(dist2s[[0, 0]] < 0.001);
        assert!(dist2s[[0, 1]] > 0.0);
    }

    #[test]
    fn test_hnsw_search_regression_all_indices_valid_for_k_equals_n() {
        let n = 10usize;
        let train_x = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 {
                i as f64
            } else {
                (i * 13) as f64 % 5.0
            }
        });
        let index = index_unit(train_x, IndexDriver::HNSW);
        let query = array![[0.25_f64, 0.25]];
        let k = n as i32;
        let (_dist2s, indices) = index.search(&query.view(), k, false).unwrap();
        assert_eq!(indices.ncols(), n);
        for j in 0..n {
            let id = indices[[0, j]];
            assert!(
                id >= 0 && (id as usize) < n,
                "neighbor slot j={j} must be a valid train row index, got id={id}"
            );
        }
    }

    #[test]
    fn test_scaled_search() {
        let x_scale = array![2.0, 2.0];
        let index = ENNIndex::new(
            array![[0.0, 0.0], [1.0, 1.0]],
            2,
            x_scale,
            true,
            IndexDriver::Exact,
        )
        .unwrap();

        let query = array![[2.0, 2.0]];
        let (dist2s, indices) = index.search(&query.view(), 1, false).unwrap();

        assert_eq!(indices[[0, 0]], 1);
        assert!(dist2s[[0, 0]] < 0.0001);
    }
}
