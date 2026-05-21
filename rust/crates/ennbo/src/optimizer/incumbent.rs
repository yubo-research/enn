use ndarray::{Array1, Array2};
use rand::RngCore;

use super::Optimizer;
use crate::error::ENNError;

impl Optimizer {
    pub(crate) fn reset_incumbent_tracker(&mut self) {
        self.incumbent_tracker.reset();
    }

    pub fn update_incumbent(&mut self, _rng: &mut dyn RngCore) -> Result<(), ENNError> {
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

        let best_idx = if self.config.noise_aware {
            if let Some(surrogate) = self.surrogate.as_ref() {
                let n_cand = candidate_indices.len();
                let num_dim = self.num_dim;
                let mut x_cand = Array2::zeros((n_cand, num_dim));
                for (r, &idx) in candidate_indices.iter().enumerate() {
                    for d in 0..num_dim {
                        x_cand[[r, d]] = self.obs_store.x_at(idx)[d];
                    }
                }
                let pred = surrogate.predict(&x_cand.view())?;
                let mut best = candidate_indices[0];
                let mut best_mu = pred.mu[[0, 0]];
                for (r, &idx) in candidate_indices.iter().enumerate().skip(1) {
                    let mu = pred.mu[[r, 0]];
                    if mu > best_mu {
                        best_mu = mu;
                        best = idx;
                    }
                }
                self.incumbent_idx = Some(best);
                self.incumbent_x_unit = Some(self.obs_store.x_at(best).clone());
                self.incumbent_y_scalar = Some(Array1::from_elem(1, best_mu));
                return Ok(());
            }
            candidate_indices
                .into_iter()
                .max_by(|&a, &b| {
                    let a_y = self.obs_store.y_at(a)[0];
                    let b_y = self.obs_store.y_at(b)[0];
                    a_y.total_cmp(&b_y)
                })
                .ok_or_else(|| ENNError::InvalidParameter("No incumbent candidates".to_string()))?
        } else {
            candidate_indices
                .into_iter()
                .max_by(|&a, &b| {
                    let a_y = self.obs_store.y_at(a)[0];
                    let b_y = self.obs_store.y_at(b)[0];
                    a_y.total_cmp(&b_y)
                })
                .ok_or_else(|| ENNError::InvalidParameter("No incumbent candidates".to_string()))?
        };

        self.incumbent_idx = Some(best_idx);
        self.incumbent_x_unit = Some(self.obs_store.x_at(best_idx).clone());
        self.incumbent_y_scalar = Some(self.obs_store.y_at(best_idx).clone());

        Ok(())
    }
}
