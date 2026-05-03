//! Epistemic Nearest Neighbors model implementation.

use ndarray::{Array1, Array2, ArrayView2, Axis};

use crate::error::ENNError;
use crate::index::{ENNIndex, IndexDriver};

/// Epistemic Nearest Neighbors model.
///
/// This is the main ENN surrogate model that provides uncertainty-aware
/// predictions using k-nearest neighbors with epistemic variance modeling.
pub struct EpistemicNearestNeighbors {
    /// Training inputs.
    pub(crate) train_x: Array2<f64>,
    /// Training targets.
    pub(crate) train_y: Array2<f64>,
    /// Observation noise variance (optional).
    pub(crate) train_yvar: Option<Array2<f64>>,
    /// Number of observations.
    pub(crate) num_obs: usize,
    /// Number of input dimensions.
    pub(crate) num_dim: usize,
    /// Number of output metrics.
    pub(crate) num_metrics: usize,
    pub(crate) scale_x: bool,
    pub(crate) x_scale: Array1<f64>,
    pub(crate) train_x_scaled: Array2<f64>,
    /// Scale factors for outputs.
    pub(crate) y_scale: Array1<f64>,
    /// KNN index.
    pub(crate) index: ENNIndex,
}

impl EpistemicNearestNeighbors {
    /// Create a new ENN model.
    pub fn new(
        train_x: Array2<f64>,
        train_y: Array2<f64>,
        train_yvar: Option<Array2<f64>>,
        scale_x: bool,
        driver: IndexDriver,
    ) -> Result<Self, ENNError> {
        if train_x.nrows() != train_y.nrows() {
            return Err(ENNError::InvalidShape {
                expected: vec![train_y.nrows(), train_x.ncols()],
                got: vec![train_x.nrows(), train_x.ncols()],
            });
        }

        if let Some(ref yvar) = train_yvar {
            if yvar.shape() != train_y.shape() {
                return Err(ENNError::InvalidShape {
                    expected: train_y.shape().to_vec(),
                    got: yvar.shape().to_vec(),
                });
            }
        }

        let num_obs = train_x.nrows();
        let num_dim = train_x.ncols();
        let num_metrics = train_y.ncols();

        let x_scale = if scale_x {
            Self::compute_scale(train_x.view(), 1e-12)
        } else {
            Array1::ones(num_dim)
        };

        let y_scale = Self::compute_scale(train_y.view(), 0.0);

        let train_x_scaled = if scale_x {
            &train_x / &x_scale.view().insert_axis(Axis(0))
        } else {
            train_x.clone()
        };

        let index = ENNIndex::new(
            train_x_scaled.clone(),
            num_dim,
            x_scale.clone(),
            scale_x,
            driver,
        )?;

        Ok(Self {
            train_x,
            train_y,
            train_yvar,
            num_obs,
            num_dim,
            num_metrics,
            scale_x,
            x_scale,
            train_x_scaled,
            y_scale,
            index,
        })
    }

    fn compute_scale(data: ArrayView2<f64>, min_val: f64) -> Array1<f64> {
        if data.nrows() < 2 {
            return Array1::ones(data.ncols());
        }
        Array1::from_iter((0..data.ncols()).map(|j| {
            let std = data.column(j).var(0.0).sqrt();
            if std.is_finite() && std > min_val { std } else { 1.0 }
        }))
    }

    /// Add new observations to the model.
    pub fn add(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Result<(), ENNError> {
        if x.nrows() != y.nrows() {
            return Err(ENNError::InvalidShape {
                expected: vec![y.nrows(), x.ncols()],
                got: vec![x.nrows(), x.ncols()],
            });
        }

        if let Some(yv) = yvar {
            if yv.shape() != y.shape() {
                return Err(ENNError::InvalidShape {
                    expected: y.shape().to_vec(),
                    got: yv.shape().to_vec(),
                });
            }
            if self.train_yvar.is_none() {
                return Err(ENNError::InvalidParameter(
                    "yvar provided but model has no existing yvar".to_string(),
                ));
            }
        } else if self.train_yvar.is_some() {
            return Err(ENNError::InvalidParameter(
                "yvar must be provided if model has existing yvar".to_string(),
            ));
        }

        self.train_x = ndarray::concatenate![Axis(0), self.train_x.view(), x.view()];
        self.train_y = ndarray::concatenate![Axis(0), self.train_y.view(), y.view()];

        if let Some(ref mut yvar_model) = self.train_yvar {
            if let Some(yv) = yvar {
                *yvar_model = ndarray::concatenate![Axis(0), yvar_model.view(), yv.view()];
            }
        }

        self.num_obs = self.train_x.nrows();
        self.y_scale = Self::compute_scale(self.train_y.view(), 0.0);

        if self.scale_x {
            self.x_scale = Self::compute_scale(self.train_x.view(), 1e-12);
            self.train_x_scaled = &self.train_x / &self.x_scale.view().insert_axis(Axis(0));
            let driver = self.index.driver();
            self.index = ENNIndex::new(
                self.train_x_scaled.clone(),
                self.num_dim,
                self.x_scale.clone(),
                true,
                driver,
            )?;
        } else {
            self.index.add(x)?;
        }

        Ok(())
    }

    /// Get the number of observations.
    pub fn len(&self) -> usize {
        self.num_obs
    }

    /// Check if model is empty.
    pub fn is_empty(&self) -> bool {
        self.num_obs == 0
    }

    /// Get number of outputs.
    pub fn num_outputs(&self) -> usize {
        self.num_metrics
    }

    /// Get k nearest neighbors for query points.
    pub fn neighbors(
        &self,
        x: &ArrayView2<f64>,
        k: i32,
        exclude_nearest: bool,
    ) -> Result<Array2<usize>, ENNError> {
        if x.ncols() != self.num_dim {
            return Err(ENNError::InvalidShape {
                expected: vec![x.nrows(), self.num_dim],
                got: vec![x.nrows(), x.ncols()],
            });
        }

        if k < 0 {
            return Err(ENNError::InvalidParameter(
                format!("k must be non-negative, got {}", k)
            ));
        }

        if self.num_obs == 0 {
            return Ok(Array2::zeros((x.nrows(), 0)));
        }

        if exclude_nearest && self.num_obs <= 1 {
            return Err(ENNError::InvalidParameter(
                format!("exclude_nearest=true requires at least 2 observations, got {}", self.num_obs)
            ));
        }

        let search_k = if exclude_nearest {
            ((k + 1) as usize).min(self.num_obs)
        } else {
            (k as usize).min(self.num_obs)
        };

        if search_k == 0 {
            return Ok(Array2::zeros((x.nrows(), 0)));
        }

        let (_, idx_full) = self.index.search(x, search_k as i32, exclude_nearest)?;

        let k_out = (k as usize).min(idx_full.ncols());
        let mut result = Array2::zeros((x.nrows(), k_out));
        for i in 0..x.nrows() {
            for j in 0..k_out {
                result[[i, j]] = idx_full[[i, j]] as usize;
            }
        }

        Ok(result)
    }

    /// Training inputs (read-only).
    pub fn train_x(&self) -> &Array2<f64> {
        &self.train_x
    }

    /// Training targets (read-only).
    pub fn train_y(&self) -> &Array2<f64> {
        &self.train_y
    }

    pub fn train_yvar(&self) -> Option<&Array2<f64>> {
        self.train_yvar.as_ref()
    }

    pub(crate) fn y_scale(&self) -> &Array1<f64> {
        &self.y_scale
    }

    /// Input scale as a single row `(1, num_dim)` for NumPy-style broadcasting.
    pub fn x_scale_row(&self) -> Array2<f64> {
        self.x_scale.clone().insert_axis(Axis(0))
    }

    /// Output scale as a single row `(1, num_metrics)` for NumPy-style broadcasting.
    pub fn y_scale_row(&self) -> Array2<f64> {
        self.y_scale.clone().insert_axis(Axis(0))
    }

    /// Whether training inputs are divided by per-dimension std scales.
    pub fn scale_x_enabled(&self) -> bool {
        self.scale_x
    }

    pub(crate) fn index(&self) -> &ENNIndex {
        &self.index
    }

    pub fn neighbor_distances_and_indices(
        &self,
        x: &ArrayView2<f64>,
        search_k: i32,
        exclude_nearest: bool,
    ) -> Result<(Array2<f64>, Array2<i64>), ENNError> {
        Ok(self.index.search(x, search_k, exclude_nearest)?)
    }

    pub(crate) fn num_obs(&self) -> usize {
        self.num_obs
    }

    pub fn num_dim(&self) -> usize {
        self.num_dim
    }

    pub fn num_metrics(&self) -> usize {
        self.num_metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_enn_creation() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let train_y = array![[0.0], [1.0], [1.0], [2.0]];

        let model = EpistemicNearestNeighbors::new(
            train_x,
            train_y,
            None,
            false,
            IndexDriver::Exact,
        )
        .unwrap();

        assert_eq!(model.len(), 4);
        assert_eq!(model.num_outputs(), 1);
    }

    #[test]
    fn test_enn_add() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0]];
        let train_y = array![[0.0], [1.0]];

        let mut model = EpistemicNearestNeighbors::new(
            train_x,
            train_y,
            None,
            false,
            IndexDriver::Exact,
        )
        .unwrap();

        let new_x = array![[0.0, 1.0]];
        let new_y = array![[1.0]];

        model.add(&new_x.view(), &new_y.view(), None).unwrap();

        assert_eq!(model.len(), 3);
    }
}
