use ndarray::{Array1, Array2, ArrayView2};

use super::EpistemicNearestNeighbors;
use crate::backend::{EnnBackend, TrainRowsAtResult};
use crate::error::ENNError;
use crate::layout::EnnLayout;
use crate::metric_auto::MetricLearning;
use crate::store_identity::{reject_existing_store, StoreIdentity};
use crate::y_bounds::{is_identity_bounds, resolve_y_bounds, warp_y, warp_yvar};

impl EpistemicNearestNeighbors {
    pub(crate) fn store_original_outputs(
        &mut self,
        rows: Option<Array2<f64>>,
        reopen: bool,
    ) -> Result<(), ENNError> {
        if let Some(rows) = rows {
            let rows = if reopen {
                Array2::zeros((0, self.num_metrics))
            } else {
                rows
            };
            self.original_y = Some(crate::input_history::InputHistory::for_named_model(
                rows,
                self,
                crate::input_history::ORIGINAL_OUTPUTS_FILE,
            )?);
        }
        Ok(())
    }

    pub fn new_empty_with_y_bounds(
        num_dim: usize,
        num_metrics: usize,
        layout: EnnLayout,
        pending_flush_threshold: Option<usize>,
        y_bounds: Option<Array2<f64>>,
    ) -> Result<Self, ENNError> {
        let work_dir = layout.work_dir();
        reject_existing_store(work_dir)?;
        let y_bounds = resolve_y_bounds(y_bounds.as_ref(), num_metrics)?;
        let backend = EnnBackend::new_empty_disk(
            work_dir.to_path_buf(),
            num_dim,
            num_metrics,
            pending_flush_threshold,
        )?;
        StoreIdentity::of(&layout, (num_dim, num_metrics), &[], &y_bounds).record(work_dir)?;
        let auto = layout.metric_learning() == MetricLearning::Auto;
        let mut model = Self {
            original_y: None,
            layout,
            backend,
            num_obs: 0,
            num_dim,
            num_metrics,
            scale_x: false,
            x_scale: Array1::ones(num_dim),
            y_scale: Array1::ones(num_metrics),
            y_bounds,
            y_sum: Array1::zeros(num_metrics),
            y_sumsq: Array1::zeros(num_metrics),
            auto_metric: None,
            tied_groups: Vec::new(),
            semantic_outputs: None,
        };
        if auto {
            model.enable_auto_metric(
                Vec::new(),
                &Array2::zeros((0, num_dim)).view(),
                &Array2::zeros((0, num_metrics)).view(),
            )?;
        }
        if !is_identity_bounds(&model.y_bounds) {
            model.original_y = Some(crate::input_history::InputHistory::for_named_model(
                Array2::zeros((0, num_metrics)),
                &model,
                crate::input_history::ORIGINAL_OUTPUTS_FILE,
            )?);
        }
        Ok(model)
    }

    pub(crate) fn ingress_warp_owned(
        train_y: Array2<f64>,
        train_yvar: Option<Array2<f64>>,
        y_bounds: &Array2<f64>,
    ) -> Result<(Array2<f64>, Option<Array2<f64>>), ENNError> {
        if is_identity_bounds(y_bounds) {
            return Ok((train_y, train_yvar));
        }
        let z = warp_y(train_y.view(), y_bounds)?;
        let zvar = match train_yvar {
            Some(yv) => Some(warp_yvar(train_y.view(), yv.view(), y_bounds)?),
            None => None,
        };
        Ok((z, zvar))
    }

    pub fn warp_observations(
        &self,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Result<(Array2<f64>, Option<Array2<f64>>), ENNError> {
        if is_identity_bounds(&self.y_bounds) {
            return Ok((y.to_owned(), yvar.map(|v| v.to_owned())));
        }
        let z = warp_y(*y, &self.y_bounds)?;
        let zvar = match yvar {
            Some(yv) => Some(warp_yvar(*y, *yv, &self.y_bounds)?),
            None => None,
        };
        Ok((z, zvar))
    }

    pub fn y_bounds(&self) -> &Array2<f64> {
        &self.y_bounds
    }

    pub fn train_rows_at(&self, indices: &[usize]) -> Result<TrainRowsAtResult, ENNError> {
        let (x, y_z, yvar_z) = self.rows().train_rows_at(indices)?;
        if is_identity_bounds(&self.y_bounds) {
            return Ok((x, y_z, yvar_z));
        }
        let y = self
            .original_y
            .as_ref()
            .ok_or_else(|| {
                crate::error::ENNError::InvalidParameter("missing original outputs".into())
            })?
            .gather(indices)?;
        let yvar = match yvar_z {
            Some(zv) => {
                let mut yv = Array2::zeros(zv.raw_dim());
                for i in 0..y.nrows() {
                    for j in 0..y.ncols() {
                        let d = crate::y_bounds::d_inv_dz(
                            y_z[[i, j]],
                            self.y_bounds[[j, 0]],
                            self.y_bounds[[j, 1]],
                        )
                        .abs();
                        yv[[i, j]] = zv[[i, j]] * d * d;
                    }
                }
                Some(yv)
            }
            None => None,
        };
        Ok((x, y, yvar))
    }

    pub fn row_y_natural(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        if let Some(history) = &self.original_y {
            return Ok(history.gather(&[i])?.row(0).to_owned());
        }
        self.rows().row_y(i)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn new_empty_with_y_bounds_stores_and_exposes_bounds() {
        let bounds = array![[0.0, 1.0]];
        let dir = tempfile::tempdir().unwrap();
        let model = EpistemicNearestNeighbors::new_empty_with_y_bounds(
            1,
            1,
            crate::layout::EnnLayout::new(dir.path(), false).unwrap(),
            None,
            Some(bounds.clone()),
        )
        .unwrap();
        assert!(crate::y_bounds::bounds_match(model.y_bounds(), &bounds));
        assert_eq!(model.len(), 0);
    }

    #[test]
    fn row_y_natural_inverse_warps_logit_storage() {
        let train_x = array![[0.0], [1.0]];
        let train_y = array![[0.2], [0.8]];
        let bounds = array![[0.0, 1.0]];
        let dir = tempfile::tempdir().unwrap();
        let model = EpistemicNearestNeighbors::new_with_storage(
            train_x,
            train_y.clone(),
            None,
            crate::layout::EnnLayout::new(dir.path(), false).unwrap(),
            Some(bounds),
        )
        .unwrap();
        let y0 = model.row_y_natural(0).unwrap();
        let y1 = model.row_y_natural(1).unwrap();
        assert!((y0[0] - 0.2).abs() < 1e-12);
        assert!((y1[0] - 0.8).abs() < 1e-12);

        let stored = model.rows().row_y(0).unwrap();
        assert!((stored[0] - y0[0]).abs() > 1e-6);
    }
}
