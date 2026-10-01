//! AUTO metric learning owned by the model.

use ndarray::{Array1, ArrayView2};

use super::EpistemicNearestNeighbors;
use crate::error::ENNError;
use crate::metric_auto::AutoMetric;

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
        let mut flat = Vec::new();
        for group in &tied {
            if group.is_empty() {
                return Err(ENNError::InvalidParameter("tied_dims groups must be non-empty".into()));
            }
            for &j in group {
                if j >= self.num_dim {
                    return Err(ENNError::InvalidParameter(format!(
                        "tied_dims entries must be in [0, {}), got {j}",
                        self.num_dim
                    )));
                }
                flat.push(j);
            }
        }
        let mut uniq = flat.clone();
        uniq.sort_unstable();
        uniq.dedup();
        if uniq.len() != flat.len() {
            return Err(ENNError::InvalidParameter("tied_dims groups must be disjoint".into()));
        }
        self.set_unscaled_dims(uniq)?;
        self.auto_metric = Some(AutoMetric::new(self.num_dim, self.num_metrics, tied, 0)?);
        self.observe_auto_metric(x, y)
    }

    pub fn metric_weights(&self) -> Option<Array1<f64>> {
        self.auto_metric.as_ref().map(|m| Array1::from(m.weights().to_vec()))
    }

    pub fn metric_built(&self) -> Option<Array1<f64>> {
        self.auto_metric.as_ref().map(|m| Array1::from(m.built().to_vec()))
    }

    pub fn metric_heldout_gain(&self) -> Option<f64> {
        self.auto_metric.as_ref().and_then(|m| m.heldout_gain)
    }

    pub fn metric_num_seen(&self) -> usize {
        self.auto_metric.as_ref().map(|m| m.num_seen()).unwrap_or(0)
    }

    pub fn metric_num_refits(&self) -> usize {
        self.auto_metric.as_ref().map(|m| m.num_refits).unwrap_or(0)
    }

    pub fn metric_num_rescales(&self) -> usize {
        self.auto_metric.as_ref().map(|m| m.num_rescales).unwrap_or(0)
    }

    pub fn metric_num_rebuilds(&self) -> usize {
        self.auto_metric.as_ref().map(|m| m.num_rebuilds).unwrap_or(0)
    }

    pub fn metric_uses_learned(&self) -> bool {
        self.auto_metric.as_ref().is_some_and(|m| m.uses_learned_metric())
    }

    pub fn metric_tied(&self) -> Option<&[Vec<usize>]> {
        self.auto_metric.as_ref().map(|m| m.tied())
    }

    pub fn metric_configure(
        &mut self,
        refit_growth: f64,
        rebuild_drift: f64,
        seed: u64,
        capacity: usize,
    ) -> Result<(), ENNError> {
        let metric = self.auto_metric.as_mut().ok_or_else(|| {
            ENNError::InvalidParameter("metric policy requires metric_learning=AUTO".into())
        })?;
        metric.configure(refit_growth, rebuild_drift, seed, capacity)
    }

    pub fn metric_set_weights(&mut self, weights: &[f64], rebuild_drift: Option<f64>) -> Result<bool, ENNError> {
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
