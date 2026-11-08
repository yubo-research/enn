use ndarray::{Array1, ArrayView2};

use super::EpistemicNearestNeighbors;
use crate::error::ENNError;
use crate::metric_auto::{AutoMetric, AutoMetricState, MetricSnapshot};
use crate::metric_weights::validate_tied_dims;

impl EpistemicNearestNeighbors {
    pub(crate) fn observe_auto_metric(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
    ) -> Result<(), ENNError> {
        let Some(metric) = self.auto_metric.as_mut() else {
            return Ok(());
        };
        let refits_before = metric.num_refits();
        let x_flat: Vec<f64> = x.iter().copied().collect();
        let y_flat: Vec<f64> = y.iter().copied().collect();
        let update = metric.observe(&x_flat, &y_flat, x.nrows())?;
        let refitted = metric.num_refits() != refits_before;
        if let Some(update) = update {
            self.set_metric_scale(Array1::from(update.x_scale), update.rebuild)?;
        }
        if refitted {
            self.save_auto_metric_state()?;
        }
        Ok(())
    }

    fn save_auto_metric_state(&self) -> Result<(), ENNError> {
        match &self.auto_metric {
            Some(metric) => metric.save_state(self.layout.work_dir()),
            None => Ok(()),
        }
    }

    fn replay_stored_rows(&self, metric: &mut AutoMetric) -> Result<(), ENNError> {
        for start in (0..self.num_obs).step_by(super::STATS_CHUNK_ROWS) {
            let end = (start + super::STATS_CHUNK_ROWS).min(self.num_obs);
            let indices: Vec<usize> = (start..end).collect();
            let (x, y, _) = self.backend.train_rows_at(&indices)?;
            let y = match &self.original_y {
                Some(history) => history.gather(&indices)?,
                None => y,
            };
            let x_flat: Vec<f64> = x.iter().copied().collect();
            let y_flat: Vec<f64> = y.iter().copied().collect();
            metric.replay_rows(&x_flat, &y_flat, indices.len());
        }
        Ok(())
    }

    pub(crate) fn restore_auto_metric(&mut self) -> Result<(), ENNError> {
        let Some(mut metric) = self.auto_metric.take() else {
            return Ok(());
        };
        let state = AutoMetricState::load(self.layout.work_dir())?;
        if let Some(state) = &state {
            metric.apply_policy(state)?;
        }
        self.replay_stored_rows(&mut metric)?;
        if self.scale_x {
            metric.restore_scale(&self.x_scale.to_vec());
        }
        if let Some(state) = state {
            metric.restore_progress(state);
        }
        self.auto_metric = Some(metric);
        Ok(())
    }

    pub fn enable_auto_metric(
        &mut self,
        tied: Vec<Vec<usize>>,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
    ) -> Result<(), ENNError> {
        if !self.layout.weighting() {
            return Err(ENNError::InvalidParameter(
                "enable_auto_metric requires weighting".into(),
            ));
        }
        validate_tied_dims(&tied, self.num_dim)?;
        self.auto_metric = Some(AutoMetric::new(self.num_dim, self.num_metrics, tied, 0)?);
        self.observe_auto_metric(x, y)
    }

    pub fn metric_snapshot(&self) -> Option<MetricSnapshot> {
        self.auto_metric.as_ref().map(|m| m.snapshot())
    }

    pub fn metric_tied(&self) -> Option<&[Vec<usize>]> {
        self.auto_metric.as_ref().map(|m| m.tied())
    }

    pub fn metric_weights(&self) -> Option<&[f64]> {
        self.auto_metric.as_ref().map(|m| m.weights())
    }

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
        metric.configure(refit_growth, rebuild_drift, seed, capacity)?;
        if refit_growth.is_some() || rebuild_drift.is_some() || seed.is_some() || capacity.is_some()
        {
            self.save_auto_metric_state()?;
        }
        Ok(())
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
        self.save_auto_metric_state()?;
        Ok(rebuild)
    }
}
