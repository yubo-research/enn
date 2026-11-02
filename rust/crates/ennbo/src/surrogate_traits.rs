use ndarray::{Array1, Array2, Array3, ArrayView2};
use rand::RngCore;

use crate::error::ENNError;
use crate::surrogate_state::SurrogateState;

#[derive(Debug, Clone)]
pub struct SurrogatePrediction {
    pub mu: Array2<f64>,
    pub se: Array2<f64>,
}

pub trait Surrogate: Send + Sync {
    fn fit(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<(), ENNError>;

    fn fit_append(
        &mut self,
        x_new: &ArrayView2<f64>,
        y_new: &ArrayView2<f64>,
        yvar_new: Option<&ArrayView2<f64>>,
        rng: &mut dyn RngCore,
    ) -> Result<(), ENNError>;

    fn validate_append(
        &self,
        _x_new: &ArrayView2<f64>,
        _y_new: &ArrayView2<f64>,
        _yvar_new: Option<&ArrayView2<f64>>,
    ) -> Result<(), ENNError> {
        Ok(())
    }

    fn predict(&self, x: &ArrayView2<f64>) -> Result<SurrogatePrediction, ENNError>;

    fn sample(
        &self,
        x: &ArrayView2<f64>,
        num_samples: usize,
        rng: &mut dyn RngCore,
    ) -> Result<Array3<f64>, ENNError>;

    fn lengthscales(&self) -> Option<Array1<f64>>;

    fn durable_state(&self) -> Result<Option<SurrogateState>, ENNError> {
        Ok(None)
    }

    fn restore_state(&mut self, _saved: Option<SurrogateState>) -> Result<(), ENNError> {
        Ok(())
    }

    fn flush_rows(&self) -> Result<(), ENNError> {
        Ok(())
    }

    fn set_replaying(&mut self, _replaying: bool) {}

    fn fitted_num_metrics(&self) -> Option<usize>;
    fn observation_count(&self) -> Option<usize>;
    fn observation_row_x(&self, idx: usize) -> Result<Array1<f64>, ENNError>;
    fn observation_row_y(&self, idx: usize) -> Result<Array1<f64>, ENNError>;
    fn observations_y(&self) -> Result<Option<Array2<f64>>, ENNError>;
    fn naturalize_observations_y(&self, y_warped: Array2<f64>) -> Array2<f64>;
    fn warp_observations_y(&self, y: &ArrayView2<f64>) -> Result<Array2<f64>, ENNError>;
    fn observations_x(&self) -> Result<Option<Array2<f64>>, ENNError>;
    fn schedule_background_flush(&self) -> Result<(), ENNError>;
    fn wait_for_background_flush(&self) -> Result<(), ENNError>;
    fn release_observation_pages(&self) -> Result<(), ENNError>;
    fn clear_observations(&mut self) -> Result<(), ENNError>;
}

pub type BoxedSurrogate = Box<dyn Surrogate + Send + Sync>;
