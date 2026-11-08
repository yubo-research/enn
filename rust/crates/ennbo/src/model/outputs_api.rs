use ndarray::Array1;

use super::EpistemicNearestNeighbors;
use crate::error::ENNError;

impl EpistemicNearestNeighbors {
    pub(crate) fn moments(&self) -> (Array1<f64>, Array1<f64>) {
        (self.y_sum.clone(), self.y_sumsq.clone())
    }

    pub(crate) fn restore_moments(
        &mut self,
        y_sum: Array1<f64>,
        y_sumsq: Array1<f64>,
        from_row: usize,
    ) -> Result<(), ENNError> {
        let later: Vec<usize> = (from_row..self.len()).collect();
        let (_, y, _) = self.rows().train_rows_at(&later)?;
        self.y_sum = y_sum;
        self.y_sumsq = y_sumsq;
        crate::model::accumulate_columns(&mut self.y_sum, &mut self.y_sumsq, y.view());
        self.y_scale = crate::model::scale_from_moments(
            self.len(),
            self.num_metrics,
            &self.y_sum,
            &self.y_sumsq,
            0.0,
        );
        Ok(())
    }

    pub fn layout(&self) -> &crate::layout::EnnLayout {
        &self.layout
    }
}
