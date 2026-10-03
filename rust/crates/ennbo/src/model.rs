//! Epistemic Nearest Neighbors model implementation.

use ndarray::{Array1, Array2, ArrayView2};
use std::path::PathBuf;
use std::sync::atomic::AtomicBool;

use crate::backend::{EnnBackend, EnnStorage};
use crate::error::ENNError;
use crate::index::IndexDriver;
use crate::layout::EnnLayout;
use crate::metric_auto::AutoMetric;
use crate::y_bounds::resolve_y_bounds;

/// Rows read at a time when recomputing statistics from a reopened disk store, so
/// the pass needs memory independent of the number of rows.
const STATS_CHUNK_ROWS: usize = 1 << 16;

type InitStats = (
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
);

mod access;
mod auto_api;
mod metric;
mod y_bounds_api;
pub use access::{EnnIndexAccess, EnnRowAccess};

/// Epistemic Nearest Neighbors model.
pub struct EpistemicNearestNeighbors {
    pub(crate) layout: EnnLayout,
    pub(crate) backend: EnnBackend,
    pub(crate) num_obs: usize,
    pub(crate) num_dim: usize,
    pub(crate) num_metrics: usize,
    pub(crate) scale_x: bool,
    pub(crate) x_scale: Array1<f64>,
    /// True once a caller-set metric owns `x_scale` (BPANN_DISK with metric learning).
    pub(crate) metric_fixed: bool,
    /// Disk `scale_x`: the `x_scale` the BPANN partition was last built under.
    pub(crate) built_x_scale: Array1<f64>,
    /// Dimensions `scale_x` leaves at scale 1 (e.g. one-hot categories).
    pub(crate) unscaled_dims: Vec<usize>,
    pub(crate) y_scale: Array1<f64>,
    /// Per-metric `(lo, hi)` in natural units; open sides are `±∞`.
    pub(crate) y_bounds: Array2<f64>,
    y_sum: Array1<f64>,
    y_sumsq: Array1<f64>,
    x_sum: Array1<f64>,
    x_sumsq: Array1<f64>,
    /// Disk work directory when using disk storage (for metadata patches).
    work_dir: Option<PathBuf>,
    /// Whether `y_bounds` is in `metadata.json` (it never changes, and metadata
    /// rewrites keep it, so it is written once).
    y_bounds_persisted: AtomicBool,
    pub(crate) auto_metric: Option<AutoMetric>,
    tied_groups: Vec<Vec<usize>>,
}

impl EpistemicNearestNeighbors {
    fn validate_shapes(
        train_x: &Array2<f64>,
        train_y: &Array2<f64>,
        train_yvar: Option<&Array2<f64>>,
    ) -> Result<(), ENNError> {
        if train_x.nrows() != train_y.nrows() {
            return Err(ENNError::InvalidShape {
                expected: vec![train_y.nrows(), train_x.ncols()],
                got: vec![train_x.nrows(), train_x.ncols()],
            });
        }
        if let Some(yvar) = train_yvar {
            if yvar.shape() != train_y.shape() {
                return Err(ENNError::InvalidShape {
                    expected: train_y.shape().to_vec(),
                    got: yvar.shape().to_vec(),
                });
            }
        }
        let yvar_view = train_yvar.map(|v| v.view());
        validate_finite_xy(&train_x.view(), &train_y.view(), yvar_view.as_ref())?;
        Ok(())
    }

    fn init_stats(train_x: &Array2<f64>, train_y: &Array2<f64>, scale_x: bool) -> InitStats {
        let num_obs = train_x.nrows();
        let num_dim = train_x.ncols();
        let num_metrics = train_y.ncols();
        let (y_sum, y_sumsq) = column_sums_and_sumsq(train_y.view());
        let y_scale = scale_from_moments(num_obs, num_metrics, &y_sum, &y_sumsq, 0.0);
        if scale_x {
            let (x_sum, x_sumsq) = column_sums_and_sumsq(train_x.view());
            let x_scale = scale_from_moments(num_obs, num_dim, &x_sum, &x_sumsq, 1e-12);
            (y_scale, y_sum, y_sumsq, x_scale, x_sum, x_sumsq)
        } else {
            (
                y_scale,
                y_sum,
                y_sumsq,
                Array1::ones(num_dim),
                Array1::zeros(num_dim),
                Array1::zeros(num_dim),
            )
        }
    }

    /// Create a new ENN model (in-memory backend).
    pub fn new(
        train_x: Array2<f64>,
        train_y: Array2<f64>,
        train_yvar: Option<Array2<f64>>,
        scale_x: bool,
        driver: IndexDriver,
    ) -> Result<Self, ENNError> {
        Self::new_with_storage(
            train_x,
            train_y,
            train_yvar,
            EnnLayout::memory(driver, scale_x),
            None,
        )
    }

    /// Create a new ENN model with explicit storage backend.
    ///
    /// `y_bounds`: optional `(num_metrics, 2)` natural-unit open intervals.
    /// `None` → all `(−∞,+∞)` for new models; on disk reopen, load from metadata.
    /// `Some` on reopen must match persisted bounds.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_storage(
        train_x: Array2<f64>,
        train_y: Array2<f64>,
        train_yvar: Option<Array2<f64>>,
        layout: EnnLayout,
        y_bounds: Option<Array2<f64>>,
    ) -> Result<Self, ENNError> {
        let scale_x = layout.scale_x();
        let storage = layout.storage();
        let work_dir = layout.work_dir().map(std::path::Path::to_path_buf);
        Self::validate_shapes(&train_x, &train_y, train_yvar.as_ref())?;
        let num_dim = train_x.ncols();
        let mut num_metrics = train_y.ncols();
        let disk_work_dir = work_dir.clone().or_else(EnnStorage::work_dir_from_env);
        let disk_reopen = matches!(storage, EnnStorage::Disk)
            && train_x.nrows() == 0
            && train_y.nrows() == 0
            && disk_work_dir
                .as_ref()
                .is_some_and(|p| p.join("metadata.json").exists());

        let meta_text = if disk_reopen {
            disk_work_dir
                .as_ref()
                .and_then(|p| std::fs::read_to_string(p.join("metadata.json")).ok())
        } else {
            None
        };


        if disk_reopen {
            if let Some(text) = meta_text.as_deref() {
                if let Some(persisted) =
                    crate::backend::disk_observation::parse_json_usize_field(text, "num_metrics")
                {
                    num_metrics = persisted;
                }
            }
        }
        let y_bounds = resolve_y_bounds(
            y_bounds.as_ref(),
            num_metrics,
            meta_text.as_deref(),
        )?;


        let (train_y, train_yvar) = if disk_reopen {
            (train_y, train_yvar)
        } else {
            Self::ingress_warp_owned(train_y, train_yvar, &y_bounds)?
        };

        let num_obs = train_x.nrows();
        let (y_scale, y_sum, y_sumsq, x_scale, x_sum, x_sumsq) =
            Self::init_stats(&train_x, &train_y, scale_x);

        let stored_work_dir = disk_work_dir.clone();
        let backend = EnnBackend::from_layout(
            &layout,
            train_x,
            train_y,
            train_yvar,
            x_scale.clone(),
        )?;

        let mut model = Self {
            layout,
            backend,
            num_obs,
            num_dim,
            num_metrics,
            scale_x,
            built_x_scale: x_scale.clone(),
            x_scale,
            metric_fixed: false,
            unscaled_dims: Vec::new(),
            y_scale,
            y_bounds,
            y_sum,
            y_sumsq,
            x_sum,
            x_sumsq,
            work_dir: stored_work_dir,
            y_bounds_persisted: AtomicBool::new(false),
            auto_metric: None,
            tied_groups: Vec::new(),
        };
        if disk_reopen || model.num_obs != model.backend.len() {
            sync_obs_stats_from_backend(&mut model)?;
        }
        model.persist_y_bounds_metadata()?;
        Ok(model)
    }

    pub fn new_empty(
        num_dim: usize,
        num_metrics: usize,
        driver: IndexDriver,
        storage: EnnStorage,
        work_dir: Option<PathBuf>,
        pending_flush_threshold: Option<usize>,
    ) -> Result<Self, ENNError> {
        Self::new_empty_with_y_bounds(
            num_dim,
            num_metrics,
            driver,
            storage,
            work_dir,
            pending_flush_threshold,
            None,
        )
    }

    fn validate_add(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Option<ENNError> {
        if x.nrows() != y.nrows() || x.ncols() != self.num_dim || y.ncols() != self.num_metrics {
            return Some(ENNError::InvalidShape {
                expected: vec![y.nrows(), self.num_metrics],
                got: vec![x.nrows(), x.ncols()],
            });
        }
        if y.ncols() != self.num_metrics {
            return Some(ENNError::InvalidParameter(format!(
                "y has {} metric columns but model expects {}",
                y.ncols(),
                self.num_metrics
            )));
        }
        if let Err(e) = validate_finite_xy(x, y, yvar) {
            return Some(e);
        }
        match (yvar, self.rows().row_yvar(0).ok().flatten().is_some()) {
            (Some(yv), _) if yv.shape() != y.shape() => Some(ENNError::InvalidShape {
                expected: y.shape().to_vec(),
                got: yv.shape().to_vec(),
            }),
            (Some(_), false) if self.num_obs > 0 => Some(ENNError::InvalidParameter(
                "yvar provided but model has no existing yvar".to_string(),
            )),
            (None, true) if self.num_obs > 0 => Some(ENNError::InvalidParameter(
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
            let (y_z, yvar_z) = self.warp_observations(y, yvar)?;
            let yvar_view = yvar_z.as_ref().map(|v| v.view());
            self.backend.wait_for_flush()?;
            self.backend
                .append_rows(x, &y_z.view(), yvar_view.as_ref())?;
            accumulate_columns(&mut self.y_sum, &mut self.y_sumsq, y_z.view());
            let n = self.backend.len();
            self.y_scale = scale_from_moments(n, self.num_metrics, &self.y_sum, &self.y_sumsq, 0.0);

            if self.scale_x && !self.metric_fixed {
                accumulate_columns(&mut self.x_sum, &mut self.x_sumsq, x.view());
                self.refresh_data_x_scale(n)?;
            }

            self.num_obs = n;
            self.persist_y_bounds_metadata()?;
            self.observe_auto_metric(x, y)?;
        }
        Ok(())
    }

    /// Schedule a background index flush when pending rows exceed the disk threshold.
    pub fn schedule_background_flush(&self) -> Result<(), ENNError> {
        self.backend.schedule_background_flush()
    }

    /// Merge in-memory index fragments and persist a single on-disk BPANN index.
    pub fn persist_index_to_disk(&self) -> Result<(), ENNError> {
        crate::backend::persist_enn_backend_index(&self.backend)?;
        self.persist_y_bounds_metadata()
    }

    pub fn len(&self) -> usize {
        self.num_obs
    }

    pub fn is_empty(&self) -> bool {
        self.num_obs == 0
    }

    pub fn num_outputs(&self) -> usize {
        self.num_metrics
    }

    pub fn is_scale_x(&self) -> bool {
        self.scale_x
    }

    /// Python `neighbors`: one query row, then the same checks as [`Self::neighbors`].
    pub fn neighbors_one(
        &self,
        x: &ArrayView2<f64>,
        k: i32,
        exclude_nearest: bool,
    ) -> Result<Array2<usize>, ENNError> {
        if x.nrows() != 1 {
            return Err(ENNError::InvalidParameter(format!(
                "x must be single point with {} dims, got {:?}",
                self.num_dim,
                x.shape()
            )));
        }
        self.neighbors(x, k, exclude_nearest)
    }

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
                "k must be non-negative, got {k}"
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
        if !self.backend.defer_index_sync_for_search() {
            self.ensure_index_sync()?;
        }
        let (_, idx_full) = self.backend.search(x, search_k as i32, exclude_nearest)?;
        let k_out = (k as usize).min(idx_full.ncols());
        let mut result = Array2::zeros((x.nrows(), k_out));
        for i in 0..x.nrows() {
            for j in 0..k_out {
                result[[i, j]] = idx_full[[i, j]] as usize;
            }
        }
        Ok(result)
    }

    pub(crate) fn y_scale(&self) -> &Array1<f64> {
        &self.y_scale
    }

    pub fn x_scale_row(&self) -> Array2<f64> {
        self.x_scale.clone().insert_axis(ndarray::Axis(0))
    }

    /// Row view of the fit/posterior `y_scale` (warped storage units under y_bounds).
    pub fn y_scale_row(&self) -> Array2<f64> {
        self.y_scale.clone().insert_axis(ndarray::Axis(0))
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

    pub fn has_yvar(&self) -> bool {
        self.num_obs > 0 && self.rows().row_yvar(0).ok().flatten().is_some()
    }

    pub(crate) fn backend_driver(&self) -> IndexDriver {
        self.backend.driver()
    }

    pub(crate) fn train_y_view_opt(&self) -> Option<ndarray::ArrayView2<'_, f64>> {
        self.backend.in_memory_train_y_view()
    }

    pub(crate) fn train_x_view_opt(&self) -> Option<ndarray::ArrayView2<'_, f64>> {
        self.backend.in_memory_train_x_view()
    }

    pub(crate) fn backend_search(
        &self,
        x: &ArrayView2<f64>,
        search_k: i32,
        exclude_nearest: bool,
    ) -> Result<(Array2<f64>, Array2<i64>), ENNError> {
        if !self.backend.defer_index_sync_for_search() {
            self.ensure_index_sync()?;
        }
        self.backend.search(x, search_k, exclude_nearest)
    }
}

fn validate_finite_xy(
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
) -> Result<(), ENNError> {
    if x.iter().any(|v| !v.is_finite()) {
        return Err(ENNError::InvalidParameter(
            "x must contain only finite values".to_string(),
        ));
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(ENNError::InvalidParameter(
            "y must contain only finite values".to_string(),
        ));
    }
    if let Some(yv) = yvar {
        if yv.iter().any(|v| !v.is_finite()) {
            return Err(ENNError::InvalidParameter(
                "yvar must contain only finite values".to_string(),
            ));
        }
        if yv.iter().any(|&v| v < 0.0) {
            return Err(ENNError::InvalidParameter(
                "yvar must be non-negative".to_string(),
            ));
        }
    }
    Ok(())
}

/// Rebuild observation count and scale moments from persisted backend rows (disk reopen).
fn sync_obs_stats_from_backend(model: &mut EpistemicNearestNeighbors) -> Result<(), ENNError> {
    model.num_dim = model.backend.num_dim();
    model.num_metrics = model.backend.num_metrics();
    let n = model.backend.len();
    model.num_obs = n;
    if n == 0 {
        return Ok(());
    }
    let (mut y_sum, mut y_sumsq) = (Array1::zeros(model.num_metrics), Array1::zeros(model.num_metrics));
    let (mut x_sum, mut x_sumsq) = (Array1::zeros(model.num_dim), Array1::zeros(model.num_dim));
    for start in (0..n).step_by(STATS_CHUNK_ROWS) {
        let indices: Vec<usize> = (start..(start + STATS_CHUNK_ROWS).min(n)).collect();
        let (x, y, _) = model.backend.train_rows_at(&indices)?;
        accumulate_columns(&mut y_sum, &mut y_sumsq, y.view());
        if model.scale_x {
            accumulate_columns(&mut x_sum, &mut x_sumsq, x.view());
        }
    }
    model.y_sum = y_sum;
    model.y_sumsq = y_sumsq;
    model.y_scale = scale_from_moments(n, model.num_metrics, &model.y_sum, &model.y_sumsq, 0.0);
    if model.scale_x {
        model.x_sum = x_sum;
        model.x_sumsq = x_sumsq;
        model.x_scale = model.data_x_scale(n);
        model.built_x_scale = model.x_scale.clone();
    }
    model.backend
        .ensure_index_sync(model.scale_x, &model.x_scale)?;
    Ok(())
}

fn column_sums_and_sumsq(a: ArrayView2<f64>) -> (Array1<f64>, Array1<f64>) {
    let ncol = a.ncols();
    let mut sum = Array1::zeros(ncol);
    let mut sumsq = Array1::zeros(ncol);
    for row in a.axis_iter(ndarray::Axis(0)) {
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
    for row in extra.axis_iter(ndarray::Axis(0)) {
        for j in 0..ncol {
            let v = row[j];
            sum[j] += v;
            sumsq[j] += v * v;
        }
    }
}

pub(crate) fn scale_from_moments(
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
    use crate::backend::row_storage::RowStorage;
    use ndarray::array;

    #[test]
    fn test_enn_creation() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let train_y = array![[0.0], [1.0], [1.0], [2.0]];
        let model =
            EpistemicNearestNeighbors::new(train_x, train_y, None, false, IndexDriver::Flat)
                .unwrap();
        assert_eq!(model.len(), 4);
        assert_eq!(model.num_outputs(), 1);
    }

    #[test]
    fn test_enn_add() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0]];
        let train_y = array![[0.0], [1.0]];
        let mut model =
            EpistemicNearestNeighbors::new(train_x, train_y, None, false, IndexDriver::Flat)
                .unwrap();
        model
            .add(&array![[0.0, 1.0]].view(), &array![[1.0]].view(), None)
            .unwrap();
        assert_eq!(model.len(), 3);
    }

    #[test]
    fn kiss_row_storage_and_scale_helpers() {
        let rows = array![[1.0, 2.0], [3.0, 4.0]];
        let mut storage = RowStorage::from_array2(rows.clone());
        assert_eq!(storage.nrows(), 2);
        storage.push_rows(&array![[5.0, 6.0]].view()).unwrap();
        assert_eq!(storage.nrows(), 3);
        let (sum, sumsq) = column_sums_and_sumsq(rows.view());
        let mut sum2 = sum.clone();
        let mut sumsq2 = sumsq.clone();
        accumulate_columns(&mut sum2, &mut sumsq2, array![[0.0, 0.0]].view());
        let scale = scale_from_moments(2, 2, &sum, &sumsq, 1e-9);
        assert_eq!(scale.len(), 2);
    }

    #[test]
    fn test_new_empty_and_row_accessors() {
        let mut model = EpistemicNearestNeighbors::new_empty(
            2,
            1,
            IndexDriver::Flat,
            EnnStorage::InMemory,
            None,
            None,
        )
        .unwrap();
        model
            .add(&array![[1.0, 2.0]].view(), &array![[3.0]].view(), None)
            .unwrap();
        let x = model.rows().row_x(0).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12);
        let y = model.rows().row_y(0).unwrap();
        assert!((y[0] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn internal_in_memory_views_and_index() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0]];
        let train_y = array![[0.0], [1.0]];
        let model =
            EpistemicNearestNeighbors::new(train_x, train_y, None, false, IndexDriver::Flat)
                .unwrap();
        assert!(model.train_x_view_opt().is_some());
        assert!(model.train_y_view_opt().is_some());
        assert_eq!(model.index_access().len(), 2);
    }

    #[test]
    fn kiss_model_accessor_helpers() {
        let mut model = EpistemicNearestNeighbors::new(
            array![[0.0, 0.0], [1.0, 0.0]],
            array![[0.0], [1.0]],
            None,
            false,
            IndexDriver::Flat,
        )
        .unwrap();
        assert!(!model.is_scale_x());
        assert_eq!(model.backend_driver(), IndexDriver::Flat);
        let _ = model.x_scale_row();
        model
            .add(&array![[0.5, 0.5]].view(), &array![[0.5]].view(), None)
            .unwrap();
    }

    #[test]
    fn disk_reopen_recomputes_stats_across_chunks() {
        let dir = tempfile::TempDir::new().unwrap();
        let n = STATS_CHUNK_ROWS + 1234;
        let x = Array2::from_shape_fn((n, 2), |(i, j)| ((i * 7 + j * 3) % 101) as f64 * 0.01);
        let y = Array2::from_shape_fn((n, 1), |(i, _)| ((i * 13) % 97) as f64 * 0.1);
        let open = |x: Array2<f64>, y: Array2<f64>| {
            EpistemicNearestNeighbors::new_with_storage(
                x,
                y,
                None,
                EnnLayout::disk(dir.path().to_path_buf(), true),
                None,
            )
            .unwrap()
        };
        let built = open(x, y);
        let stats = |m: &EpistemicNearestNeighbors| {
            (m.num_obs, m.y_sum.clone(), m.y_sumsq.clone(), m.x_sum.clone(), m.x_sumsq.clone())
        };
        let expected = stats(&built);
        drop(built);
        let reopened = open(Array2::zeros((0, 2)), Array2::zeros((0, 1)));
        assert_eq!(stats(&reopened), expected);
        assert_eq!(reopened.x_scale, reopened.built_x_scale);
    }
}
