use std::path::Path;

use ennbo_bpann::{bpann_read_json, bpann_write_json};
use serde::{Deserialize, Serialize};

use super::{AutoMetric, MetricCounters};
use crate::error::ENNError;

pub(crate) const METRIC_STATE_FILE: &str = "metric_state.json";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct AutoMetricState {
    seed: u64,
    capacity: usize,
    refit_growth: Option<f64>,
    rebuild_drift: Option<f64>,
    next_refit: usize,
    heldout_gain: Option<f64>,
    counters: MetricCounters,
}

fn file_error(e: ennbo_bpann::BpannError) -> ENNError {
    ENNError::InvalidParameter(e.to_string())
}

impl AutoMetricState {
    pub(crate) fn load(work_dir: &Path) -> Result<Option<Self>, ENNError> {
        bpann_read_json(&work_dir.join(METRIC_STATE_FILE)).map_err(file_error)
    }
}

impl AutoMetric {
    pub(crate) fn save_state(&self, work_dir: &Path) -> Result<(), ENNError> {
        let state = AutoMetricState {
            seed: self.seed(),
            capacity: self.capacity(),
            refit_growth: Some(self.refit_growth),
            rebuild_drift: Some(self.rebuild_drift),
            next_refit: self.next_refit,
            heldout_gain: self.heldout_gain,
            counters: self.counters,
        };
        bpann_write_json(&work_dir.join(METRIC_STATE_FILE), &state).map_err(file_error)
    }

    pub(crate) fn apply_policy(&mut self, state: &AutoMetricState) -> Result<(), ENNError> {
        self.configure(
            Some(state.refit_growth.unwrap_or(f64::INFINITY)),
            Some(state.rebuild_drift.unwrap_or(f64::INFINITY)),
            Some(state.seed),
            Some(state.capacity),
        )
    }

    pub(crate) fn replay_rows(&mut self, x: &[f64], y: &[f64], n: usize) {
        let (d, m) = (self.rows.num_dim(), self.rows.num_outputs());
        for i in 0..n {
            self.rows
                .push_row(&x[i * d..(i + 1) * d], &y[i * m..(i + 1) * m]);
        }
    }

    pub(crate) fn restore_scale(&mut self, x_scale: &[f64]) {
        self.weights = x_scale.iter().map(|s| s.powi(-2)).collect();
        self.built.clone_from(&self.weights);
    }

    pub(crate) fn restore_progress(&mut self, state: AutoMetricState) {
        self.next_refit = state.next_refit;
        self.heldout_gain = state.heldout_gain;
        self.counters = state.counters;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn state_round_trips_including_infinite_policy_and_gain() {
        let dir = TempDir::new().unwrap();
        let mut saved = AutoMetric::new(2, 1, vec![], 3).unwrap();
        saved
            .configure(Some(f64::INFINITY), Some(f64::INFINITY), Some(9), Some(5))
            .unwrap();
        saved.heldout_gain = Some(f64::NEG_INFINITY);
        saved.next_refit = 77;
        saved.counters.num_refits = 4;
        saved.save_state(dir.path()).unwrap();

        let state = AutoMetricState::load(dir.path()).unwrap().unwrap();
        let mut restored = AutoMetric::new(2, 1, vec![], 0).unwrap();
        restored.apply_policy(&state).unwrap();
        restored.restore_progress(state);
        assert_eq!(restored.seed(), 9);
        assert_eq!(restored.capacity(), 5);
        assert_eq!(restored.refit_growth, f64::INFINITY);
        assert_eq!(restored.rebuild_drift, f64::INFINITY);
        assert_eq!(restored.next_refit, 77);
        assert_eq!(restored.heldout_gain, None);
        assert_eq!(restored.counters.num_refits, 4);
    }

    #[test]
    fn a_store_without_state_loads_none() {
        let dir = TempDir::new().unwrap();
        assert_eq!(AutoMetricState::load(dir.path()).unwrap(), None);
    }

    #[test]
    fn replayed_rows_match_rows_seen_by_observe() {
        let x: Vec<f64> = (0..40).map(f64::from).collect();
        let y: Vec<f64> = (0..20).map(f64::from).collect();
        let mut live = AutoMetric::new(2, 1, vec![], 1).unwrap();
        let mut replayed = AutoMetric::new(2, 1, vec![], 1).unwrap();
        live.configure(None, None, None, Some(4)).ok();
        replayed.configure(None, None, None, Some(4)).ok();
        live.observe(&x, &y, 20).unwrap();
        replayed.replay_rows(&x, &y, 20);
        assert_eq!(live.rows.x(), replayed.rows.x());
        assert_eq!(live.rows.y(), replayed.rows.y());
        assert_eq!(live.num_seen(), replayed.num_seen());
    }

    #[test]
    fn restore_scale_sets_weights_and_marks_them_built() {
        let mut metric = AutoMetric::new(2, 1, vec![], 0).unwrap();
        metric.restore_scale(&[0.5, 2.0]);
        assert_eq!(metric.weights(), &[4.0, 0.25]);
        assert_eq!(metric.built(), &[4.0, 0.25]);
    }
}
