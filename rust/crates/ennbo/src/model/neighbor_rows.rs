use ndarray::{Array1, ArrayView2};

use super::{EpistemicNearestNeighbors, STATS_CHUNK_ROWS};
use crate::error::ENNError;

impl EpistemicNearestNeighbors {
    pub fn aggregates_duplicates(&self) -> bool {
        self.dups.is_some()
    }

    pub fn num_rows(&self) -> usize {
        self.dups.as_ref().map_or(self.num_obs, |d| d.num_rows())
    }

    pub(crate) fn neighbor_y(&self, obs: usize) -> Result<Array1<f64>, ENNError> {
        match &self.dups {
            Some(dups) => Ok(Array1::from(dups.mean_at(obs).to_vec())),
            None => self.rows().row_y(obs),
        }
    }

    pub(crate) fn neighbor_count(&self, obs: usize) -> f64 {
        self.dups.as_ref().map_or(1.0, |d| d.count_at(obs))
    }

    pub(crate) fn neighbor_yvar(&self, obs: usize) -> Result<Option<Array1<f64>>, ENNError> {
        match &self.dups {
            Some(dups) => Ok(dups.mean_yvar_at(obs).map(Array1::from)),
            None => self.rows().row_yvar(obs),
        }
    }

    pub(super) fn observe_dups(
        &mut self,
        first_obs: usize,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) {
        let Some(dups) = self.dups.as_mut() else {
            return;
        };
        for i in 0..y.nrows() {
            let yvar_row = yvar.map(|v| v.row(i).to_vec());
            dups.observe(first_obs + i, &y.row(i).to_vec(), yvar_row.as_deref());
        }
    }

    pub(super) fn observe_stored_rows_for_dups(&mut self) -> Result<(), ENNError> {
        if self.dups.is_none() {
            return Ok(());
        }
        let n = self.backend.len();
        for start in (0..n).step_by(STATS_CHUNK_ROWS) {
            let indices: Vec<usize> = (start..(start + STATS_CHUNK_ROWS).min(n)).collect();
            let (y, yvar) = self.backend.train_y_rows_at(&indices)?;
            self.observe_dups(start, &y.view(), yvar.as_ref().map(|v| v.view()).as_ref());
        }
        Ok(())
    }
}
