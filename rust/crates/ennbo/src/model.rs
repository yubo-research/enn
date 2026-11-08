use ndarray::{Array1, Array2, ArrayView2};
use std::path::PathBuf;

use crate::backend::EnnBackend;
use crate::dup_table::DupTable;
use crate::error::ENNError;
use crate::input_history::InputHistory;
use crate::layout::EnnLayout;
use crate::metric_auto::AutoMetric;
use crate::store_identity::{claim_store_dir, StoreIdentity};
use crate::y_bounds::resolve_y_bounds;

const STATS_CHUNK_ROWS: usize = 1 << 16;

mod access;
mod auto_api;
mod metric;
mod neighbor_rows;
mod outputs_api;
mod row_validation;
mod y_bounds_api;
pub use access::{EnnIndexAccess, EnnRowAccess};

struct ModelRows {
    train_x: Array2<f64>,
    train_y: Array2<f64>,
    train_yvar: Option<Array2<f64>>,
}

pub struct EpistemicNearestNeighbors {
    original_y: Option<InputHistory>,
    pub(crate) layout: EnnLayout,
    pub(crate) backend: EnnBackend,
    pub(crate) num_obs: usize,
    pub(crate) num_dim: usize,
    pub(crate) num_metrics: usize,
    pub(crate) scale_x: bool,
    pub(crate) x_scale: Array1<f64>,
    pub(crate) y_scale: Array1<f64>,
    pub(crate) y_bounds: Array2<f64>,
    y_sum: Array1<f64>,
    y_sumsq: Array1<f64>,
    pub(crate) auto_metric: Option<AutoMetric>,
    tied_groups: Vec<Vec<usize>>,
    pub(crate) dups: Option<DupTable>,
}

impl EpistemicNearestNeighbors {
    pub fn new(
        train_x: Array2<f64>,
        train_y: Array2<f64>,
        train_yvar: Option<Array2<f64>>,
        layout: EnnLayout,
    ) -> Result<Self, ENNError> {
        Self::new_with_storage(train_x, train_y, train_yvar, layout, None)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn new_with_storage(
        train_x: Array2<f64>,
        train_y: Array2<f64>,
        train_yvar: Option<Array2<f64>>,
        layout: EnnLayout,
        y_bounds: Option<Array2<f64>>,
    ) -> Result<Self, ENNError> {
        Self::new_configured(train_x, train_y, train_yvar, layout, y_bounds, Vec::new())
    }

    pub fn new_configured(
        train_x: Array2<f64>,
        train_y: Array2<f64>,
        train_yvar: Option<Array2<f64>>,
        layout: EnnLayout,
        y_bounds: Option<Array2<f64>>,
        tied_dims: Vec<Vec<usize>>,
    ) -> Result<Self, ENNError> {
        crate::metric_weights::require_weighting_for_ties(&tied_dims, layout.weighting())?;
        crate::metric_weights::validate_tied_dims(&tied_dims, train_x.ncols())?;
        let yvar_view = train_yvar.as_ref().map(|v| v.view());
        Self::validate_shapes(&train_x.view(), &train_y.view(), yvar_view.as_ref())?;
        claim_store_dir(layout.work_dir())?;
        Self::assemble(
            ModelRows {
                train_x,
                train_y,
                train_yvar,
            },
            layout,
            y_bounds,
            tied_dims,
            false,
        )
    }

    pub fn open(work_dir: impl Into<PathBuf>) -> Result<Self, ENNError> {
        let work_dir = work_dir.into();
        let identity = StoreIdentity::load(&work_dir)?;
        let layout = identity.layout(work_dir)?;
        Self::assemble(
            ModelRows {
                train_x: Array2::zeros((0, identity.num_dim)),
                train_y: Array2::zeros((0, identity.num_metrics)),
                train_yvar: None,
            },
            layout,
            Some(identity.y_bounds()),
            identity.tied_dims,
            true,
        )
    }

    fn assemble(
        rows: ModelRows,
        layout: EnnLayout,
        y_bounds: Option<Array2<f64>>,
        tied_dims: Vec<Vec<usize>>,
        reopen: bool,
    ) -> Result<Self, ENNError> {
        let ModelRows {
            train_x,
            train_y,
            train_yvar,
        } = rows;
        let auto_rows = (layout.metric_learning() == crate::metric_auto::MetricLearning::Auto)
            .then(|| (train_x.clone(), train_y.clone()));
        let num_dim = train_x.ncols();
        let num_metrics = train_y.ncols();
        let work_dir = layout.work_dir();
        let y_bounds = resolve_y_bounds(y_bounds.as_ref(), num_metrics)?;

        let original_y = (!crate::y_bounds::is_identity_bounds(&y_bounds)).then(|| train_y.clone());
        let (train_y, train_yvar) = if reopen {
            (train_y, train_yvar)
        } else {
            Self::ingress_warp_owned(train_y, train_yvar, &y_bounds)?
        };

        let num_obs = train_x.nrows();
        let (y_sum, y_sumsq) = column_sums_and_sumsq(train_y.view());
        let y_scale = scale_from_moments(num_obs, num_metrics, &y_sum, &y_sumsq, 0.0);
        let x_scale = Array1::ones(num_dim);
        let identity = StoreIdentity::of(&layout, (num_dim, num_metrics), &tied_dims, &y_bounds);
        if !reopen {
            identity.record(work_dir)?;
        }
        let backend = EnnBackend::new_disk(
            work_dir.to_path_buf(),
            train_x,
            train_y,
            train_yvar,
            x_scale.clone(),
            layout.aggregate_duplicates(),
        )?;
        let dups = DupTable::attach(&backend, num_metrics);

        let mut model = Self {
            original_y: None,
            layout,
            backend,
            num_obs,
            num_dim,
            num_metrics,
            scale_x: false,
            x_scale,
            y_scale,
            y_bounds,
            y_sum,
            y_sumsq,
            auto_metric: None,
            tied_groups: Vec::new(),
            dups,
        };
        model.observe_stored_rows_for_dups()?;
        if let Some(scale) = model.backend.metric_scale()? {
            model.scale_x = true;
            model.x_scale = scale;
        }
        if reopen || model.num_obs != model.backend.len() {
            sync_obs_stats_from_backend(&mut model)?;
        }
        model.store_original_outputs(original_y, reopen)?;
        model.set_tied_groups(tied_dims.clone())?;
        if let Some((x, y)) = auto_rows {
            model.enable_auto_metric(tied_dims, &x.view(), &y.view())?;
            if reopen {
                model.restore_auto_metric()?;
            }
        }
        if !reopen {
            identity.commit(model.layout.work_dir())?;
        }
        Ok(model)
    }

    pub fn new_empty(
        num_dim: usize,
        num_metrics: usize,
        layout: EnnLayout,
        pending_flush_threshold: Option<usize>,
    ) -> Result<Self, ENNError> {
        Self::new_empty_with_y_bounds(num_dim, num_metrics, layout, pending_flush_threshold, None)
    }

    pub fn add(
        &mut self,
        train_x: &ArrayView2<f64>,
        train_y: &ArrayView2<f64>,
        train_yvar: Option<&ArrayView2<f64>>,
    ) -> Result<(), ENNError> {
        let (y_z, yvar_z) = self.prepare_new_rows(train_x, train_y, train_yvar)?;
        if train_x.nrows() > 0 {
            let yvar_view = yvar_z.as_ref().map(|v| v.view());
            self.backend.wait_for_flush()?;
            let first_row = self.backend.len();
            if let Some(history) = &mut self.original_y {
                history.write_at(first_row, train_y)?;
            }
            self.backend
                .append_rows(train_x, &y_z.view(), yvar_view.as_ref())?;
            self.observe_dups(first_row, &y_z.view(), yvar_view.as_ref());
            accumulate_columns(&mut self.y_sum, &mut self.y_sumsq, y_z.view());
            let n = self.backend.len();
            self.y_scale = scale_from_moments(n, self.num_metrics, &self.y_sum, &self.y_sumsq, 0.0);

            self.num_obs = n;
            self.observe_auto_metric(train_x, train_y)?;
        }
        Ok(())
    }

    pub fn schedule_background_flush(&self) -> Result<(), ENNError> {
        self.backend.schedule_background_flush()
    }

    pub fn persist_index_to_disk(&self) -> Result<(), ENNError> {
        self.backend.persist_index_to_disk()
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
        crate::util::check_finite_query(x)?;
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

    pub(crate) fn backend_search(
        &self,
        x: &ArrayView2<f64>,
        search_k: i32,
        exclude_nearest: bool,
    ) -> Result<(Array2<f64>, Array2<i64>), ENNError> {
        self.backend.search(x, search_k, exclude_nearest)
    }
}

fn sync_obs_stats_from_backend(model: &mut EpistemicNearestNeighbors) -> Result<(), ENNError> {
    model.num_dim = model.backend.num_dim();
    model.num_metrics = model.backend.num_metrics();
    let n = model.backend.len();
    model.num_obs = n;
    if n == 0 {
        return Ok(());
    }
    let (mut y_sum, mut y_sumsq) = (
        Array1::zeros(model.num_metrics),
        Array1::zeros(model.num_metrics),
    );
    for start in (0..n).step_by(STATS_CHUNK_ROWS) {
        let indices: Vec<usize> = (start..(start + STATS_CHUNK_ROWS).min(n)).collect();
        let (_, y, _) = model.backend.train_rows_at(&indices)?;
        accumulate_columns(&mut y_sum, &mut y_sumsq, y.view());
    }
    model.y_sum = y_sum;
    model.y_sumsq = y_sumsq;
    model.y_scale = scale_from_moments(n, model.num_metrics, &model.y_sum, &model.y_sumsq, 0.0);
    model
        .backend
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

pub(crate) fn accumulate_columns(
    sum: &mut Array1<f64>,
    sumsq: &mut Array1<f64>,
    extra: ArrayView2<f64>,
) {
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
    use ndarray::array;

    #[test]
    fn test_enn_creation() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let train_y = array![[0.0], [1.0], [1.0], [2.0]];
        let model = crate::test_helpers::test_model(train_x, train_y, None, false);
        assert_eq!(model.len(), 4);
        assert_eq!(model.num_outputs(), 1);
    }

    #[test]
    fn store_is_written_to_work_dir_and_outlives_the_model() {
        let dir = tempfile::tempdir().unwrap();
        let store = dir.path().join("store");
        let model = EpistemicNearestNeighbors::new(
            array![[0.0, 0.0], [1.0, 0.0]],
            array![[0.0], [1.0]],
            None,
            EnnLayout::new(&store, false).unwrap(),
        )
        .unwrap();
        assert_eq!(model.layout.work_dir(), store);
        assert!(store.join("metadata.json").exists());
        drop(model);
        assert!(store.join("metadata.json").exists());
    }

    #[test]
    fn search_rejects_wrong_dimension_and_nonpositive_k() {
        let model = crate::test_helpers::test_model(
            array![[0.0, 0.0], [1.0, 0.0]],
            array![[0.0], [1.0]],
            None,
            false,
        );
        let access = model.index_access();
        let wrong_dim = array![[0.0, 0.0, 0.0]];
        let right_dim = array![[0.0, 0.0]];
        assert!(access
            .neighbor_distances_and_indices(&wrong_dim.view(), 1, false)
            .is_err());
        assert!(access
            .neighbor_distances_and_indices(&right_dim.view(), 0, false)
            .is_err());
    }

    #[test]
    fn test_enn_add() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0]];
        let train_y = array![[0.0], [1.0]];
        let mut model = crate::test_helpers::test_model(train_x, train_y, None, false);
        model
            .add(&array![[0.0, 1.0]].view(), &array![[1.0]].view(), None)
            .unwrap();
        assert_eq!(model.len(), 3);
    }

    #[test]
    fn kiss_scale_helpers() {
        let rows = array![[1.0, 2.0], [3.0, 4.0]];
        let (sum, sumsq) = column_sums_and_sumsq(rows.view());
        let mut sum2 = sum.clone();
        let mut sumsq2 = sumsq.clone();
        accumulate_columns(&mut sum2, &mut sumsq2, array![[0.0, 0.0]].view());
        let scale = scale_from_moments(2, 2, &sum, &sumsq, 1e-9);
        assert_eq!(scale.len(), 2);
    }

    #[test]
    fn test_new_empty_and_row_accessors() {
        let dir = tempfile::tempdir().unwrap();
        let mut model = EpistemicNearestNeighbors::new_empty(
            2,
            1,
            crate::layout::EnnLayout::new(dir.path(), false).unwrap(),
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
    fn kiss_model_accessor_helpers() {
        let mut model = crate::test_helpers::test_model(
            array![[0.0, 0.0], [1.0, 0.0]],
            array![[0.0], [1.0]],
            None,
            false,
        );
        assert!(!model.is_scale_x());
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
        let built = EpistemicNearestNeighbors::new_with_storage(
            x,
            y,
            None,
            crate::layout::EnnLayout::new(dir.path().to_path_buf(), true).unwrap(),
            None,
        )
        .unwrap();
        let stats = |m: &EpistemicNearestNeighbors| (m.num_obs, m.y_sum.clone(), m.y_sumsq.clone());
        let expected = stats(&built);
        drop(built);
        let reopened = EpistemicNearestNeighbors::open(dir.path()).unwrap();
        assert_eq!(stats(&reopened), expected);
    }
}
