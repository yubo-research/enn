//! AUTO metric learning owned by the model.

use ndarray::{Array1, ArrayView2};

use super::EpistemicNearestNeighbors;
use crate::error::ENNError;
use crate::layout::EnnLayout;
use crate::metric_auto::{AutoMetric, MetricSnapshot};
use crate::metric_weights::validate_tied_dims;

impl EpistemicNearestNeighbors {
    pub(crate) fn observe_auto_metric(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
    ) -> Result<(), ENNError> {
        let update = if let Some(metric) = self.auto_metric.as_mut() {
            let x_flat: Vec<f64> = x.iter().copied().collect();
            let y_flat: Vec<f64> = y.iter().copied().collect();
            metric.observe(&x_flat, &y_flat, x.nrows())?
        } else {
            None
        };
        if let Some(update) = update {
            self.set_metric_scale(Array1::from(update.x_scale), update.rebuild)?;
        }
        Ok(())
    }

    pub fn enable_auto_metric(
        &mut self,
        tied: Vec<Vec<usize>>,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
    ) -> Result<(), ENNError> {
        if !matches!(self.layout, EnnLayout::DiskAuto { .. }) {
            return Err(ENNError::InvalidParameter(
                "enable_auto_metric requires EnnLayout::DiskAuto".into(),
            ));
        }
        validate_tied_dims(&tied, self.num_dim)?;
        let uniq: Vec<usize> = {
            let mut flat: Vec<usize> = tied.iter().flatten().copied().collect();
            flat.sort_unstable();
            flat.dedup();
            flat
        };
        self.set_unscaled_dims(uniq)?;
        self.auto_metric = Some(AutoMetric::new(self.num_dim, self.num_metrics, tied, 0)?);
        self.observe_auto_metric(x, y)
    }

    pub fn metric_snapshot(&self) -> Option<MetricSnapshot> {
        self.auto_metric.as_ref().map(|m| m.snapshot())
    }

    pub fn metric_tied(&self) -> Option<&[Vec<usize>]> {
        self.auto_metric.as_ref().map(|m| m.tied())
    }

    /// Weights currently applied to the metric, one per input dimension.
    pub fn metric_weights(&self) -> Option<&[f64]> {
        self.auto_metric.as_ref().map(|m| m.weights())
    }

    /// Update only the metric fields that are `Some`.
    pub fn metric_configure(
        &mut self,
        refit_growth: Option<f64>,
        rebuild_drift: Option<f64>,
        seed: Option<u64>,
        capacity: Option<usize>,
    ) -> Result<(), ENNError> {
        let metric = self.auto_metric.as_mut().ok_or_else(|| {
            ENNError::InvalidParameter("metric policy requires metric_learning=AUTO".into())
        })?;
        metric.configure(refit_growth, rebuild_drift, seed, capacity)
    }

    pub fn metric_set_weights(
        &mut self,
        weights: &[f64],
        rebuild_drift: Option<f64>,
    ) -> Result<bool, ENNError> {
        let update = {
            let metric = self.auto_metric.as_mut().ok_or_else(|| {
                ENNError::InvalidParameter("set_weights requires metric_learning=AUTO".into())
            })?;
            if let Some(drift) = rebuild_drift {
                metric.rebuild_drift = drift;
            }
            metric.set_weights(weights)?
        };
        let rebuild = update.rebuild;
        self.set_metric_scale(Array1::from(update.x_scale), rebuild)?;
        Ok(rebuild)
    }
}
