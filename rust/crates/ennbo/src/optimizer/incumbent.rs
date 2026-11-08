use ndarray::{Array1, Array2, ArrayView2, Axis};
use rand::RngCore;
use serde::{Deserialize, Serialize};

use super::Optimizer;
use crate::error::ENNError;
use crate::util::argmax_random_tie;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(super) struct BestObservation {
    #[serde(with = "crate::exact")]
    x_unit: Array1<f64>,
    #[serde(with = "crate::exact")]
    y: Array1<f64>,
}

impl Optimizer {
    pub(crate) fn record_best_observation(
        &mut self,
        x_unit: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
    ) {
        if self.tr_state.is_morbo() {
            return;
        }
        for (x_row, y_row) in x_unit.rows().into_iter().zip(y.rows()) {
            let better = match &self.best_observation {
                Some(best) => y_row[0] > best.y[0],
                None => !y_row[0].is_nan(),
            };
            if better {
                self.best_observation = Some(BestObservation {
                    x_unit: x_row.to_owned(),
                    y: y_row.to_owned(),
                });
            }
        }
    }

    pub fn best_x(&self) -> Option<Array1<f64>> {
        let best = self.best_observation.as_ref()?;
        let row = best.x_unit.view().insert_axis(Axis(0));
        let natural = crate::candidates::from_unit(&row, &self.bounds.view());
        Some(natural.row(0).to_owned())
    }

    pub fn best_y(&self) -> Option<&Array1<f64>> {
        self.best_observation.as_ref().map(|best| &best.y)
    }

    #[allow(dead_code)]
    pub(crate) fn reset_incumbent_tracker(&mut self) {
        self.incumbent_tracker.reset();
    }

    pub fn update_incumbent(&mut self, rng: &mut dyn RngCore) -> Result<(), ENNError> {
        if self.obs_access().observations_empty() {
            self.incumbent_idx = None;
            self.incumbent_x_unit = None;
            self.incumbent_y_scalar = None;
            return Ok(());
        }

        if self.incumbent_tracker.observation_count() != self.obs_count() {
            if let Some(y_nat) = self.y_obs() {
                self.incumbent_tracker.rebuild(&y_nat.view());
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
                let y_row = self.obs_access().obs_row_y(idx)?;
                for m in 0..y_rows.ncols() {
                    y_rows[[r, m]] = y_row[m];
                }
            }
            if self.config.noise_aware {
                if let Some(surrogate) = self.surrogate.as_ref() {
                    let mut x_cand = Array2::zeros((n_cand, self.num_dim));
                    for (r, &idx) in candidate_indices.iter().enumerate() {
                        let x_row = self.obs_access().obs_row_x(idx)?;
                        for d in 0..self.num_dim {
                            x_cand[[r, d]] = x_row[d];
                        }
                    }

                    y_rows = surrogate.predict(&x_cand.view())?.mu;
                }
            }
            let scores = self
                .tr_state
                .morbo()
                .ok_or_else(|| {
                    ENNError::InvalidParameter("Morbo incumbent without Morbo".to_string())
                })?
                .scalarize_local(&y_rows.view())
                .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
            let best_pos = argmax_random_tie(scores.as_slice().unwrap_or(&[]), rng);
            let best_idx = candidate_indices[best_pos];
            self.incumbent_idx = Some(best_idx);
            self.incumbent_x_unit = Some(self.obs_access().obs_row_x(best_idx)?);
            self.incumbent_y_scalar = Some(y_rows.row(best_pos).to_owned());
            return Ok(());
        }

        let best_idx = candidate_indices
            .into_iter()
            .max_by(|&a, &b| {
                self.obs_access()
                    .obs_row_y(a)
                    .and_then(|ya| {
                        self.obs_access()
                            .obs_row_y(b)
                            .map(|yb| ya[0].total_cmp(&yb[0]))
                    })
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .ok_or_else(|| ENNError::InvalidParameter("No incumbent candidates".to_string()))?;

        self.incumbent_idx = Some(best_idx);
        self.incumbent_x_unit = Some(self.obs_access().obs_row_x(best_idx)?);
        self.incumbent_y_scalar = Some(self.obs_access().obs_row_y(best_idx)?);

        Ok(())
    }
}
