//! In-memory ENN backend (dev / small N).

use ndarray::{Array1, Array2, ArrayView2, Axis};
use std::sync::Mutex;

use crate::error::ENNError;
use crate::index::{ENNIndex, IndexDriver};

use super::disk_observation as disk_obs;
use super::row_storage::RowStorage;

pub struct InMemoryEnnBackend {
    train_x_rows: RowStorage,
    train_y_rows: RowStorage,
    train_yvar_rows: Option<RowStorage>,
    num_dim: usize,
    num_metrics: usize,
    index: ENNIndex,
    index_stale: Mutex<bool>,
    index_synced_obs: Mutex<usize>,
}

impl InMemoryEnnBackend {
    pub fn new(
        train_x: Array2<f64>,
        train_y: Array2<f64>,
        train_yvar: Option<Array2<f64>>,
        scale_x: bool,
        x_scale: Array1<f64>,
        driver: IndexDriver,
    ) -> Result<Self, ENNError> {
        let num_obs = train_x.nrows();
        let num_dim = train_x.ncols();
        let num_metrics = train_y.ncols();

        let train_x_scaled = if scale_x {
            (&train_x / &x_scale.view().insert_axis(Axis(0))).to_owned()
        } else {
            train_x.clone()
        };

        let index = ENNIndex::new(
            train_x_scaled,
            num_dim,
            x_scale,
            scale_x,
            driver,
        )
        .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
        let index_synced_obs = index.len().min(num_obs);

        Ok(Self {
            train_x_rows: RowStorage::from_array2(train_x),
            train_y_rows: RowStorage::from_array2(train_y),
            train_yvar_rows: train_yvar.map(RowStorage::from_array2),
            num_dim,
            num_metrics,
            index,
            index_stale: Mutex::new(false),
            index_synced_obs: Mutex::new(index_synced_obs),
        })
    }

    pub fn new_empty(num_dim: usize, num_metrics: usize, driver: IndexDriver) -> Result<Self, ENNError> {
        let empty_x = Array2::<f64>::zeros((0, num_dim));
        let empty_y = Array2::<f64>::zeros((0, num_metrics));
        Self::new(empty_x, empty_y, None, false, Array1::ones(num_dim), driver)
    }

    pub fn len(&self) -> usize {
        self.train_x_rows.nrows()
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
        self.index.driver()
    }

    pub fn index_len(&self) -> usize {
        self.index.len()
    }

    pub fn train_x_view(&self) -> ndarray::ArrayView2<'_, f64> {
        self.train_x_rows.view()
    }

    pub fn train_y_view(&self) -> ndarray::ArrayView2<'_, f64> {
        self.train_y_rows.view()
    }

    pub fn index(&self) -> &ENNIndex {
        &self.index
    }

    pub fn append_rows(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Result<(), ENNError> {
        if x.nrows() == 0 {
            return Ok(());
        }
        self.train_x_rows.push_rows(x)?;
        self.train_y_rows.push_rows(y)?;
        match (&mut self.train_yvar_rows, yvar) {
            (Some(yvar_rows), Some(yv)) => yvar_rows.push_rows(yv)?,
            (None, Some(yv)) => {
                self.train_yvar_rows = Some(RowStorage::from_array2(yv.to_owned()));
            }
            (Some(_), None) | (None, None) => {}
        }
        Ok(())
    }

    pub fn mark_index_stale(&self) {
        disk_obs::set_index_stale(&self.index_stale);
    }

    pub fn is_index_stale(&self) -> bool {
        disk_obs::read_index_stale(&self.index_stale)
    }

    pub fn ensure_index_sync(
        &self,
        scale_x: bool,
        x_scale: &Array1<f64>,
    ) -> Result<(), ENNError> {
        if scale_x {
            let mut stale = self
                .index_stale
                .lock()
                .expect("index_stale mutex poisoned");
            if !*stale {
                return Ok(());
            }
            let train_x_scaled = (&self.train_x_rows.view()
                / &x_scale.view().insert_axis(Axis(0)))
                .to_owned();
            self.index
                .rebuild_from_scaled(train_x_scaled, x_scale.clone())?;
            *stale = false;
            *self
                .index_synced_obs
                .lock()
                .expect("index_synced_obs mutex poisoned") = self.len();
            return Ok(());
        }
        let num_obs = self.len();
        let mut synced = self
            .index_synced_obs
            .lock()
            .expect("index_synced_obs mutex poisoned");
        if *synced > self.index.len() {
            *synced = self.index.len();
        }
        if num_obs > 0
            && (self.index.len() != num_obs || (*synced == 0 && !self.index.is_empty()))
        {
            let train_x_scaled = self.train_x_rows.view().to_owned();
            self.index
                .rebuild_from_scaled(train_x_scaled, x_scale.clone())?;
            *synced = num_obs;
            return Ok(());
        }
        if *synced >= num_obs && self.index.len() >= num_obs {
            return Ok(());
        }
        let train_view = self.train_x_rows.view();
        let pending = train_view.slice(ndarray::s![*synced.., ..]);
        if pending.nrows() > 0 {
            self.index
                .add(&pending)
                .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
        }
        *synced = num_obs;
        Ok(())
    }

    pub fn train_rows_at(
        &self,
        indices: &[usize],
    ) -> Result<super::TrainRowsAtResult, ENNError> {
        let n = self.len();
        for &i in indices {
            if i >= n {
                return Err(ENNError::InvalidParameter(format!(
                    "train_rows_at index {i} out of range [0, {n})"
                )));
            }
        }
        let x = self.train_x_rows.gather_rows(indices);
        let y = self.train_y_rows.gather_rows(indices);
        let yvar = self
            .train_yvar_rows
            .as_ref()
            .map(|r| r.gather_rows(indices));
        Ok((x, y, yvar))
    }

    pub fn row_x(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        if i >= self.len() {
            return Err(ENNError::InvalidParameter(format!(
                "row_x index {i} out of range [0, {})",
                self.len()
            )));
        }
        Ok(self.train_x_rows.row_vec(i))
    }

    pub fn row_y(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        if i >= self.len() {
            return Err(ENNError::InvalidParameter(format!(
                "row_y index {i} out of range [0, {})",
                self.len()
            )));
        }
        Ok(self.train_y_rows.row_vec(i))
    }

    pub fn row_yvar(&self, i: usize) -> Result<Option<Array1<f64>>, ENNError> {
        if i >= self.len() {
            return Err(ENNError::InvalidParameter(format!(
                "row_yvar index {i} out of range [0, {})",
                self.len()
            )));
        }
        Ok(self
            .train_yvar_rows
            .as_ref()
            .map(|r| r.row_vec(i)))
    }

    pub fn search(
        &self,
        x: &ArrayView2<f64>,
        search_k: i32,
        exclude_nearest: bool,
    ) -> Result<(Array2<f64>, Array2<i64>), ENNError> {
        self.index
            .search(x, search_k, exclude_nearest)
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))
    }

    pub fn index_memory_bytes(&self) -> Result<usize, ENNError> {
        Ok(self.index.memory_usage_bytes())
    }
}

#[cfg(test)]
mod in_memory_unit_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn covers_row_and_search_helpers() {
        let backend = InMemoryEnnBackend::new(
            array![[0.0, 0.0], [1.0, 0.0]],
            array![[0.0], [1.0]],
            None,
            false,
            Array1::ones(2),
            IndexDriver::Exact,
        )
        .unwrap();
        let (x, y, yv) = backend.train_rows_at(&[1]).unwrap();
        assert_eq!(x.nrows(), 1);
        assert_eq!(y.nrows(), 1);
        assert!(yv.is_none());
        assert_eq!(backend.row_x(0).unwrap()[0], 0.0);
        assert_eq!(backend.row_y(1).unwrap()[0], 1.0);
        assert!(backend.row_yvar(0).unwrap().is_none());
        backend.search(&array![[0.1, 0.2]].view(), 1, false).unwrap();
        assert!(backend.index_memory_bytes().unwrap() > 0);
    }
}
