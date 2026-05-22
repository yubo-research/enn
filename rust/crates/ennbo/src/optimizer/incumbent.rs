use ndarray::{Array1, Array2};
use rand::RngCore;

use super::Optimizer;
use crate::error::ENNError;
use crate::util::argmax_random_tie;

impl Optimizer {
    fn pick_noise_aware_incumbent(&mut self, candidate_indices: &[usize]) -> Result<(), ENNError> {
        let surrogate = self.surrogate.as_ref().unwrap();
        let n_cand = candidate_indices.len();
        let mut x_cand = Array2::zeros((n_cand, self.num_dim));
        for (r, &idx) in candidate_indices.iter().enumerate() {
            for d in 0..self.num_dim {
                x_cand[[r, d]] = self.obs_store.x_at(idx)[d];
            }
        }
        let pred = surrogate.predict(&x_cand.view())?;
        let mut best = candidate_indices[0];
        let mut best_mu = pred.mu[[0, 0]];
        for (r, &idx) in candidate_indices.iter().enumerate().skip(1) {
            if pred.mu[[r, 0]] > best_mu {
                best_mu = pred.mu[[r, 0]];
                best = idx;
            }
        }
        self.incumbent_idx = Some(best);
        self.incumbent_x_unit = Some(self.obs_store.x_at(best).clone());
        self.incumbent_y_scalar = Some(Array1::from_elem(1, best_mu));
        Ok(())
    }

    pub(crate) fn reset_incumbent_tracker(&mut self) {
        self.incumbent_tracker.reset();
    }

    pub fn update_incumbent(&mut self, rng: &mut dyn RngCore) -> Result<(), ENNError> {
        if self.obs_store.is_empty() {
            self.incumbent_idx = None;
            self.incumbent_x_unit = None;
            self.incumbent_y_scalar = None;
            return Ok(());
        }

        if self.incumbent_tracker.observation_count() != self.obs_store.len() {
            if let Some(y_obs) = self.obs_store.y_obs_array() {
                self.incumbent_tracker.rebuild(&y_obs.view());
            }
        }
        let candidate_indices = self.incumbent_tracker.ask();

        if candidate_indices.is_empty() {
            self.incumbent_idx = None;
            self.incumbent_x_unit = None;
            self.incumbent_y_scalar = None;
            return Ok(());
        }

        if self.tr_state.is_morbo() {
            let n_cand = candidate_indices.len();
            let mut y_rows = Array2::zeros((n_cand, self.tr_state.num_metrics()));
            for (r, &idx) in candidate_indices.iter().enumerate() {
                let y_row = self.obs_store.y_at(idx);
                for m in 0..y_rows.ncols() {
                    y_rows[[r, m]] = y_row[m];
                }
            }
            if self.tr_state.morbo().map(|m| m.noise_aware()).unwrap_or(false) {
                if let Some(surrogate) = self.surrogate.as_ref() {
                    let mut x_cand = Array2::zeros((n_cand, self.num_dim));
                    for (r, &idx) in candidate_indices.iter().enumerate() {
                        for d in 0..self.num_dim {
                            x_cand[[r, d]] = self.obs_store.x_at(idx)[d];
                        }
                    }
                    y_rows = surrogate.predict(&x_cand.view())?.mu;
                }
            }
            let scores = self
                .tr_state
                .morbo_scalarize(&y_rows.view(), true)
                .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
            let best_pos = argmax_random_tie(scores.as_slice().unwrap_or(&[]), rng);
            let best_idx = candidate_indices[best_pos];
            self.incumbent_idx = Some(best_idx);
            self.incumbent_x_unit = Some(self.obs_store.x_at(best_idx).clone());
            self.incumbent_y_scalar = Some(y_rows.row(best_pos).to_owned());
            return Ok(());
        }

        if self.config.noise_aware && self.surrogate.is_some() {
            return self.pick_noise_aware_incumbent(&candidate_indices);
        }

        let best_idx = candidate_indices
            .into_iter()
            .max_by(|&a, &b| {
                self.obs_store.y_at(a)[0].total_cmp(&self.obs_store.y_at(b)[0])
            })
            .ok_or_else(|| ENNError::InvalidParameter("No incumbent candidates".to_string()))?;

        self.incumbent_idx = Some(best_idx);
        self.incumbent_x_unit = Some(self.obs_store.x_at(best_idx).clone());
        self.incumbent_y_scalar = Some(self.obs_store.y_at(best_idx).clone());

        Ok(())
    }
}
