//! Epistemic Nearest Neighbors model implementation.

use ndarray::{Array1, Array2, ArrayView2, Axis};
use std::path::PathBuf;
use std::sync::Mutex;

use crate::error::ENNError;
use crate::index::{ENNIndex, IndexDriver};

#[derive(Debug)]
pub(crate) struct RowStorage {
    buf: Vec<f64>,
    nrows: usize,
    ncols: usize,
}

impl RowStorage {
    fn from_array2(a: Array2<f64>) -> Self {
        let (nrows, ncols) = a.dim();
        let a = a.as_standard_layout().into_owned();
        let mut buf = Vec::with_capacity(nrows.saturating_mul(ncols));
        buf.extend(a.iter());
        let cur_elems = nrows.saturating_mul(ncols);
        buf.reserve(cur_elems.max(ncols.saturating_mul(4096)));
        Self { buf, nrows, ncols }
    }

    fn nrows(&self) -> usize {
        self.nrows
    }

    fn view(&self) -> ArrayView2<'_, f64> {
        ArrayView2::from_shape((self.nrows, self.ncols), &self.buf[..self.nrows * self.ncols])
            .expect("row-major view")
    }

    fn gather_rows(&self, indices: &[usize]) -> Array2<f64> {
        let mut out = Array2::zeros((indices.len(), self.ncols));
        for (new_i, &old_i) in indices.iter().enumerate() {
            out.row_mut(new_i).assign(&self.view().row(old_i));
        }
        out
    }

    fn row_owned(&self, i: usize) -> Array1<f64> {
        self.view().row(i).to_owned()
    }

    fn push_rows(&mut self, extra: &ArrayView2<f64>) -> Result<(), ENNError> {
        if extra.ncols() != self.ncols {
            return Err(ENNError::InvalidShape {
                expected: vec![self.nrows, self.ncols],
                got: vec![extra.nrows(), extra.ncols()],
            });
        }
        let n1 = extra.nrows();
        if n1 == 0 {
            return Ok(());
        }
        let cur_elems = self.nrows * self.ncols;
        let add_elems = n1 * self.ncols;
        let new_elems = cur_elems + add_elems;
        if self.buf.capacity() < new_elems {
            let growth = new_elems.saturating_sub(self.buf.capacity());
            let slack = cur_elems.max(self.ncols.saturating_mul(4096));
            self.buf.reserve(growth + slack);
        }
        for row in extra.axis_iter(Axis(0)) {
            self.buf.extend(row.iter().copied());
        }
        self.nrows += n1;
        Ok(())
    }
}

fn build_enn_index(
    train_x_scaled: Array2<f64>,
    num_dim: usize,
    x_scale: Array1<f64>,
    scale_x: bool,
    driver: IndexDriver,
    index_path: Option<PathBuf>,
) -> Result<ENNIndex, ENNError> {
    #[cfg(feature = "usearch")]
    if let Some(path) = index_path {
        return ENNIndex::with_index_path(
            train_x_scaled,
            num_dim,
            x_scale,
            scale_x,
            driver,
            path,
        )
        .map_err(|e| ENNError::InvalidParameter(e.to_string()));
    }
    #[cfg(not(feature = "usearch"))]
    if index_path.is_some() {
        return Err(ENNError::InvalidParameter(
            "index_path requires the `usearch` feature".to_string(),
        ));
    }
    ENNIndex::new(train_x_scaled, num_dim, x_scale, scale_x, driver)
        .map_err(|e| ENNError::InvalidParameter(e.to_string()))
}

/// Epistemic Nearest Neighbors model.
///
/// This is the main ENN surrogate model that provides uncertainty-aware
/// predictions using k-nearest neighbors with epistemic variance modeling.
pub struct EpistemicNearestNeighbors {
    /// Training inputs.
    pub(crate) train_x_rows: RowStorage,
    /// Training targets.
    pub(crate) train_y_rows: RowStorage,
    /// Observation noise variance (optional).
    pub(crate) train_yvar_rows: Option<RowStorage>,
    /// Number of observations.
    pub(crate) num_obs: usize,
    /// Number of input dimensions.
    pub(crate) num_dim: usize,
    /// Number of output metrics.
    pub(crate) num_metrics: usize,
    pub(crate) scale_x: bool,
    pub(crate) x_scale: Array1<f64>,
    /// Scale factors for outputs.
    pub(crate) y_scale: Array1<f64>,
    /// KNN index.
    pub(crate) index: ENNIndex,
    y_sum: Array1<f64>,
    y_sumsq: Array1<f64>,
    x_sum: Array1<f64>,
    x_sumsq: Array1<f64>,
    index_stale: Mutex<bool>,
    index_synced_obs: Mutex<usize>,
}

impl EpistemicNearestNeighbors {
    /// Create a new ENN model.
    pub fn new(
        train_x: Array2<f64>,
        train_y: Array2<f64>,
        train_yvar: Option<Array2<f64>>,
        scale_x: bool,
        driver: IndexDriver,
    ) -> Result<Self, ENNError> {
        Self::new_with_index_path(train_x, train_y, train_yvar, scale_x, driver, None)
    }

    /// Create a new ENN model with an optional USearch index file path.
    pub fn new_with_index_path(
        train_x: Array2<f64>,
        train_y: Array2<f64>,
        train_yvar: Option<Array2<f64>>,
        scale_x: bool,
        driver: IndexDriver,
        index_path: Option<PathBuf>,
    ) -> Result<Self, ENNError> {
        if train_x.nrows() != train_y.nrows() {
            return Err(ENNError::InvalidShape {
                expected: vec![train_y.nrows(), train_x.ncols()],
                got: vec![train_x.nrows(), train_x.ncols()],
            });
        }

        if let Some(ref yvar) = train_yvar {
            if yvar.shape() != train_y.shape() {
                return Err(ENNError::InvalidShape {
                    expected: train_y.shape().to_vec(),
                    got: yvar.shape().to_vec(),
                });
            }
        }

        let num_obs = train_x.nrows();
        let num_dim = train_x.ncols();
        let num_metrics = train_y.ncols();

        let (y_sum, y_sumsq) = column_sums_and_sumsq(train_y.view());
        let y_scale = scale_from_moments(num_obs, num_metrics, &y_sum, &y_sumsq, 0.0);

        let (x_scale, x_sum, x_sumsq) = if scale_x {
            let (xs, xsq) = column_sums_and_sumsq(train_x.view());
            let xscl = scale_from_moments(num_obs, num_dim, &xs, &xsq, 1e-12);
            (xscl, xs, xsq)
        } else {
            (
                Array1::ones(num_dim),
                Array1::zeros(num_dim),
                Array1::zeros(num_dim),
            )
        };

        let train_x_scaled = if scale_x {
            (&train_x / &x_scale.view().insert_axis(Axis(0))).to_owned()
        } else {
            train_x.clone()
        };

        let index = build_enn_index(
            train_x_scaled,
            num_dim,
            x_scale.clone(),
            scale_x,
            driver,
            index_path,
        )?;
        let index_synced_obs = index.len().min(num_obs);

        let train_x_rows = RowStorage::from_array2(train_x);
        let train_y_rows = RowStorage::from_array2(train_y);
        let train_yvar_rows = train_yvar.map(RowStorage::from_array2);

        Ok(Self {
            train_x_rows,
            train_y_rows,
            train_yvar_rows,
            num_obs,
            num_dim,
            num_metrics,
            scale_x,
            x_scale,
            y_scale,
            index,
            y_sum,
            y_sumsq,
            x_sum,
            x_sumsq,
            index_stale: Mutex::new(false),
            index_synced_obs: Mutex::new(index_synced_obs),
        })
    }

    pub(crate) fn ensure_index_sync(&self) -> Result<(), ENNError> {
        if self.scale_x {
            let mut stale = self
                .index_stale
                .lock()
                .expect("index_stale mutex poisoned");
            if !*stale {
                return Ok(());
            }
            let train_x_scaled =
                (&self.train_x_rows.view() / &self.x_scale.view().insert_axis(Axis(0))).to_owned();
            self.index
                .rebuild_from_scaled(train_x_scaled, self.x_scale.clone())?;
            *stale = false;
            *self
                .index_synced_obs
                .lock()
                .expect("index_synced_obs mutex poisoned") = self.num_obs;
            return Ok(());
        }
        let mut synced = self
            .index_synced_obs
            .lock()
            .expect("index_synced_obs mutex poisoned");
        if *synced > self.index.len() {
            *synced = self.index.len();
        }
        if self.num_obs > 0 && self.index.len() != self.num_obs {
            let train_x_scaled = self.train_x_rows.view().to_owned();
            self.index
                .rebuild_from_scaled(train_x_scaled, self.x_scale.clone())?;
            *synced = self.num_obs;
            return Ok(());
        }
        if *synced >= self.num_obs && self.index.len() >= self.num_obs {
            return Ok(());
        }
        let train_view = self.train_x_rows.view();
        let pending = train_view.slice(ndarray::s![*synced.., ..]);
        if pending.nrows() > 0 {
            self.index
                .add(&pending)
                .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
        }
        *synced = self.num_obs;
        Ok(())
    }

    pub fn sync_index(&self) -> Result<(), ENNError> {
        self.ensure_index_sync()?;
        #[cfg(feature = "usearch")]
        self.index
            .checkpoint()
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
        Ok(())
    }

    /// RAM used by the KNN index after the last successful sync (not disk checkpoints).
    pub fn index_memory_bytes(&self) -> Result<usize, ENNError> {
        self.ensure_index_sync()?;
        Ok(self.index.memory_usage_bytes())
    }

    /// Whether the FAISS index is marked stale (full rebuild required on next sync).
    pub fn is_index_stale(&self) -> bool {
        *self
            .index_stale
            .lock()
            .expect("index_stale mutex poisoned")
    }

    fn validate_add(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Option<ENNError> {
        if self.num_obs > 0 && self.scale_x {
            return Some(ENNError::InvalidParameter(
                "scale_x must be false when adding to a non-empty model".to_string(),
            ));
        }
        if x.nrows() != y.nrows() || x.ncols() != self.num_dim || y.ncols() != self.num_metrics {
            return Some(ENNError::InvalidShape {
                expected: vec![y.nrows(), self.num_metrics],
                got: vec![x.nrows(), x.ncols()],
            });
        }
        match (yvar, self.train_yvar_rows.is_some()) {
            (Some(yv), _) if yv.shape() != y.shape() => Some(ENNError::InvalidShape {
                expected: y.shape().to_vec(),
                got: yv.shape().to_vec(),
            }),
            (Some(_), false) if self.num_obs > 0 => Some(ENNError::InvalidParameter(
                "yvar provided but model has no existing yvar".to_string(),
            )),
            (None, true) => Some(ENNError::InvalidParameter(
                "yvar must be provided if model has existing yvar".to_string(),
            )),
            _ => None,
        }
    }

    pub fn add(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Result<(), ENNError> {
        if let Some(err) = self.validate_add(x, y, yvar) {
            return Err(err);
        }
        if x.nrows() > 0 {
            self.train_x_rows.push_rows(x)?;
            self.train_y_rows.push_rows(y)?;
            match (&mut self.train_yvar_rows, yvar) {
                (Some(yvar_rows), Some(yv)) => yvar_rows.push_rows(yv)?,
                (None, Some(yv)) => {
                    self.train_yvar_rows = Some(RowStorage::from_array2(yv.to_owned()));
                }
                (Some(_), None) | (None, None) => {}
            }

            accumulate_columns(&mut self.y_sum, &mut self.y_sumsq, y.view());
            let n = self.train_y_rows.nrows();
            self.y_scale = scale_from_moments(n, self.num_metrics, &self.y_sum, &self.y_sumsq, 0.0);

            if self.scale_x {
                accumulate_columns(&mut self.x_sum, &mut self.x_sumsq, x.view());
                self.x_scale = scale_from_moments(n, self.num_dim, &self.x_sum, &self.x_sumsq, 1e-12);
                *self
                    .index_stale
                    .lock()
                    .expect("index_stale mutex poisoned") = true;
            }

            self.num_obs = self.train_x_rows.nrows();
        }

        Ok(())
    }

    /// Get the number of observations.
    pub fn len(&self) -> usize {
        self.num_obs
    }

    /// Get number of outputs.
    pub fn num_outputs(&self) -> usize {
        self.num_metrics
    }

    /// Get k nearest neighbors for query points.
    pub fn neighbors(
        &self,
        x: &ArrayView2<f64>,
        k: i32,
        exclude_nearest: bool,
    ) -> Result<Array2<usize>, ENNError> {
        if x.ncols() != self.num_dim {
            return Err(ENNError::InvalidShape {
                expected: vec![x.nrows(), self.num_dim],
                got: vec![x.nrows(), x.ncols()],
            });
        }

        if k < 0 {
            return Err(ENNError::InvalidParameter(format!(
                "k must be non-negative, got {}",
                k
            )));
        }

        if self.num_obs == 0 {
            return Ok(Array2::zeros((x.nrows(), 0)));
        }

        if exclude_nearest && self.num_obs <= 1 {
            return Err(ENNError::InvalidParameter(format!(
                "exclude_nearest=true requires at least 2 observations, got {}",
                self.num_obs
            )));
        }

        let search_k = if exclude_nearest {
            ((k + 1) as usize).min(self.num_obs)
        } else {
            (k as usize).min(self.num_obs)
        };

        if search_k == 0 {
            return Ok(Array2::zeros((x.nrows(), 0)));
        }

        self.ensure_index_sync()?;
        let (_, idx_full) = self.index.search(x, search_k as i32, exclude_nearest)?;

        let k_out = (k as usize).min(idx_full.ncols());
        let mut result = Array2::zeros((x.nrows(), k_out));
        for i in 0..x.nrows() {
            for j in 0..k_out {
                result[[i, j]] = idx_full[[i, j]] as usize;
            }
        }

        Ok(result)
    }

    /// Gather training rows by global index.
    pub fn train_rows_at(
        &self,
        indices: &[usize],
    ) -> Result<(Array2<f64>, Array2<f64>, Option<Array2<f64>>), ENNError> {
        for &i in indices {
            if i >= self.num_obs {
                return Err(ENNError::InvalidParameter(format!(
                    "train_rows_at index {i} out of range [0, {})",
                    self.num_obs
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

    /// Training inputs (deprecated for large N; use `train_rows_at`).
    pub fn train_x(&self) -> ArrayView2<'_, f64> {
        self.train_x_rows.view()
    }

    pub fn train_y(&self) -> ArrayView2<'_, f64> {
        self.train_y_rows.view()
    }

    pub fn train_yvar(&self) -> Option<ArrayView2<'_, f64>> {
        self.train_yvar_rows.as_ref().map(|r| r.view())
    }

    pub(crate) fn y_scale(&self) -> &Array1<f64> {
        &self.y_scale
    }

    /// Input scale as a single row `(1, num_dim)` for NumPy-style broadcasting.
    pub fn x_scale_row(&self) -> Array2<f64> {
        self.x_scale.clone().insert_axis(Axis(0))
    }

    /// Output scale as a single row `(1, num_metrics)` for NumPy-style broadcasting.
    pub fn y_scale_row(&self) -> Array2<f64> {
        self.y_scale.clone().insert_axis(Axis(0))
    }

    /// Whether training inputs are divided by per-dimension std scales.
    pub fn scale_x_enabled(&self) -> bool {
        self.scale_x
    }

    pub(crate) fn index(&self) -> &ENNIndex {
        &self.index
    }

    pub fn neighbor_distances_and_indices(
        &self,
        x: &ArrayView2<f64>,
        search_k: i32,
        exclude_nearest: bool,
    ) -> Result<(Array2<f64>, Array2<i64>), ENNError> {
        self.ensure_index_sync()?;
        Ok(self.index.search(x, search_k, exclude_nearest)?)
    }

    /// Neighbor lookup via `index_search` (f64 exact ranking for Exact driver).
    pub fn index_neighbor_distances_and_indices(
        &self,
        x: &ArrayView2<f64>,
        search_k: i32,
        exclude_nearest: bool,
        tie_break_neighbors: bool,
    ) -> Result<(Array2<f64>, Array2<i64>), ENNError> {
        crate::posterior::index_search(self, x, search_k, exclude_nearest, tie_break_neighbors)
    }

    pub(crate) fn num_obs(&self) -> usize {
        self.num_obs
    }

    pub fn num_dim(&self) -> usize {
        self.num_dim
    }

    pub fn num_metrics(&self) -> usize {
        self.num_metrics
    }
}

fn column_sums_and_sumsq(a: ArrayView2<f64>) -> (Array1<f64>, Array1<f64>) {
    let ncol = a.ncols();
    let mut sum = Array1::zeros(ncol);
    let mut sumsq = Array1::zeros(ncol);
    for row in a.axis_iter(Axis(0)) {
        for j in 0..ncol {
            let v = row[j];
            sum[j] += v;
            sumsq[j] += v * v;
        }
    }
    (sum, sumsq)
}

fn accumulate_columns(sum: &mut Array1<f64>, sumsq: &mut Array1<f64>, extra: ArrayView2<f64>) {
    let ncol = extra.ncols();
    for row in extra.axis_iter(Axis(0)) {
        for j in 0..ncol {
            let v = row[j];
            sum[j] += v;
            sumsq[j] += v * v;
        }
    }
}

fn scale_from_moments(
    n: usize,
    ncol: usize,
    sum: &Array1<f64>,
    sumsq: &Array1<f64>,
    min_std: f64,
) -> Array1<f64> {
    if n < 2 {
        return Array1::ones(ncol);
    }
    let nf = n as f64;
    Array1::from_iter((0..ncol).map(|j| {
        let mean = sum[j] / nf;
        let var = (sumsq[j] / nf - mean * mean).max(0.0);
        let std = var.sqrt();
        if std.is_finite() && std > min_std {
            std
        } else {
            1.0
        }
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_enn_creation() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let train_y = array![[0.0], [1.0], [1.0], [2.0]];

        let model =
            EpistemicNearestNeighbors::new(train_x, train_y, None, false, IndexDriver::Exact)
                .unwrap();

        assert_eq!(model.len(), 4);
        assert_eq!(model.num_outputs(), 1);
    }

    #[test]
    fn test_enn_add() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0]];
        let train_y = array![[0.0], [1.0]];

        let mut model =
            EpistemicNearestNeighbors::new(train_x, train_y, None, false, IndexDriver::Exact)
                .unwrap();

        let new_x = array![[0.0, 1.0]];
        let new_y = array![[1.0]];

        model.add(&new_x.view(), &new_y.view(), None).unwrap();

        assert_eq!(model.len(), 3);
    }

    #[test]
    fn kiss_row_storage_and_scale_helpers() {
        let rows = array![[1.0, 2.0], [3.0, 4.0]];
        let mut storage = RowStorage::from_array2(rows.clone());
        assert_eq!(storage.nrows(), 2);
        storage
            .push_rows(&array![[5.0, 6.0]].view())
            .unwrap();
        assert_eq!(storage.nrows(), 3);
        let (sum, sumsq) = column_sums_and_sumsq(rows.view());
        let mut sum2 = sum.clone();
        let mut sumsq2 = sumsq.clone();
        accumulate_columns(&mut sum2, &mut sumsq2, array![[0.0, 0.0]].view());
        let scale = scale_from_moments(2, 2, &sum, &sumsq, 1e-9);
        assert_eq!(scale.len(), 2);
    }

    #[test]
    fn test_sync_index_idempotent() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0]];
        let train_y = array![[0.0], [1.0]];

        let mut model =
            EpistemicNearestNeighbors::new(train_x, train_y, None, false, IndexDriver::Exact)
                .unwrap();

        let new_x = array![[0.0, 1.0]];
        let new_y = array![[1.0]];
        model.add(&new_x.view(), &new_y.view(), None).unwrap();

        model.sync_index().unwrap();
        model.sync_index().unwrap();
    }

    #[cfg(feature = "usearch")]
    #[test]
    fn stale_smaller_checkpoint_with_full_train_reconciles_before_search() {
        use tempfile::TempDir;

        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("index.usearch");
        let train2 = array![[0.0, 0.0], [1.0, 0.0]];
        let train_y2 = array![[0.0], [1.0]];
        EpistemicNearestNeighbors::new_with_index_path(
            train2,
            train_y2,
            None,
            false,
            IndexDriver::HNSWUSearch,
            Some(path.clone()),
        )
        .expect("seed checkpoint");

        let train5 = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [5.0, 5.0],
        ];
        let train_y5 = array![[0.0], [1.0], [2.0], [3.0], [4.0]];
        let model = EpistemicNearestNeighbors::new_with_index_path(
            train5.clone(),
            train_y5,
            None,
            false,
            IndexDriver::HNSWUSearch,
            Some(path),
        )
        .expect("reopen with full train");
        assert_eq!(model.len(), 5);
        assert_eq!(model.index.len(), 5);

        let idx = model
            .neighbors(&array![[4.9, 4.9]].view(), 1, false)
            .expect("search");
        assert_eq!(idx[[0, 0]], 4);

        let exact = EpistemicNearestNeighbors::new(
            train5,
            array![[0.0], [1.0], [2.0], [3.0], [4.0]],
            None,
            false,
            IndexDriver::Exact,
        )
        .expect("exact reference");
        let exact_idx = exact
            .neighbors(&array![[4.9, 4.9]].view(), 1, false)
            .expect("exact search");
        assert_eq!(idx[[0, 0]], exact_idx[[0, 0]]);
    }

    #[cfg(feature = "usearch")]
    #[test]
    fn view_only_reopen_then_add_neighbors_use_row_ordinals() {
        use crate::params::{ENNParams, PosteriorFlags};
        use crate::traits::PosteriorComputation;
        use tempfile::TempDir;

        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("index.usearch");
        let n_seed = 100usize;
        let mut train_x =
            ndarray::Array2::zeros((n_seed, 2));
        let mut train_y = ndarray::Array2::zeros((n_seed, 1));
        for i in 0..n_seed {
            train_x[[i, 0]] = i as f64 * 0.01;
            train_x[[i, 1]] = (i as f64 * 0.02).sin();
            train_y[[i, 0]] = i as f64;
        }
        EpistemicNearestNeighbors::new_with_index_path(
            train_x,
            train_y,
            None,
            false,
            IndexDriver::HNSWUSearch,
            Some(path.clone()),
        )
        .expect("seed checkpoint");

        let empty_x = ndarray::Array2::<f64>::zeros((0, 2));
        let empty_y = ndarray::Array2::<f64>::zeros((0, 1));
        let mut model = EpistemicNearestNeighbors::new_with_index_path(
            empty_x,
            empty_y,
            None,
            false,
            IndexDriver::HNSWUSearch,
            Some(path),
        )
        .expect("view-only reopen");
        assert_eq!(model.len(), 0);
        assert!(model.index.len() >= n_seed);

        model
            .add(&array![[0.1, 0.2]].view(), &array![[1.0]].view(), None)
            .unwrap();
        assert_eq!(model.len(), 1);

        let idx = model
            .neighbors(&array![[0.1, 0.2]].view(), 1, false)
            .expect("neighbors");
        assert_eq!(
            idx[[0, 0]], 0,
            "neighbor index must be a row ordinal, not a stale USearch key"
        );

        let params = ENNParams {
            k_num_neighbors: 1,
            epistemic_variance_scale: 1.0,
            aleatoric_variance_scale: 0.0,
        };
        model
            .posterior(
                &array![[0.1, 0.2]].view(),
                &params,
                &PosteriorFlags::default(),
            )
            .expect("posterior must not panic on row gather");
    }

    #[cfg(feature = "usearch")]
    #[test]
    fn reopen_with_wrong_checkpoint_prefix_rebuilds_before_search() {
        use tempfile::TempDir;

        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("index.usearch");
        let train2 = array![[0.0, 0.0], [1.0, 0.0]];
        let train_y2 = array![[0.0], [1.0]];
        EpistemicNearestNeighbors::new_with_index_path(
            train2,
            train_y2,
            None,
            false,
            IndexDriver::HNSWUSearch,
            Some(path.clone()),
        )
        .expect("seed checkpoint");

        let train2_wrong = array![[9.0, 9.0], [8.0, 8.0]];
        let train_y_wrong = array![[0.0], [1.0]];
        let model = EpistemicNearestNeighbors::new_with_index_path(
            train2_wrong.clone(),
            train_y_wrong,
            None,
            false,
            IndexDriver::HNSWUSearch,
            Some(path),
        )
        .expect("reopen with mismatched prefix");

        let idx = model
            .neighbors(&array![[9.05, 9.05]].view(), 1, false)
            .expect("search");
        assert_eq!(idx[[0, 0]], 0);

        let exact = EpistemicNearestNeighbors::new(
            train2_wrong,
            array![[0.0], [1.0]],
            None,
            false,
            IndexDriver::Exact,
        )
        .expect("exact reference");
        let exact_idx = exact
            .neighbors(&array![[9.05, 9.05]].view(), 1, false)
            .expect("exact search");
        assert_eq!(idx[[0, 0]], exact_idx[[0, 0]]);
    }
}
