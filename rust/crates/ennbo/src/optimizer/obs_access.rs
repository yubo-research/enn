use ndarray::{Array1, Array2};

use super::Optimizer;
use crate::error::ENNError;

/// Public accessor for optimizer observation row reads (kiss external-test surface).
pub struct ObsAccess<'a> {
    opt: &'a Optimizer,
}

impl<'a> ObsAccess<'a> {
    pub(crate) fn new(opt: &'a Optimizer) -> Self {
        Self { opt }
    }

    pub fn observations_empty(&self) -> bool {
        self.opt.obs_count() == 0
    }

    pub fn obs_row_x(&self, idx: usize) -> Result<Array1<f64>, ENNError> {
        if let Some(surrogate) = self.opt.surrogate() {
            return surrogate.observation_row_x(idx);
        }
        self.opt
            .fallback_x
            .get(idx)
            .cloned()
            .ok_or_else(|| ENNError::InvalidParameter(format!("observation index {idx} out of range")))
    }

    pub fn obs_row_y(&self, idx: usize) -> Result<Array1<f64>, ENNError> {
        if let Some(surrogate) = self.opt.surrogate() {
            return surrogate.observation_row_y(idx);
        }
        self.opt
            .fallback_y
            .get(idx)
            .cloned()
            .ok_or_else(|| ENNError::InvalidParameter(format!("observation index {idx} out of range")))
    }
}

pub fn build_obs_array2(vecs: &[Array1<f64>]) -> Array2<f64> {
    let n = vecs.len();
    let d = vecs[0].len();
    let mut result = Array2::zeros((n, d));
    for (i, v) in vecs.iter().enumerate() {
        for j in 0..d {
            result[[i, j]] = v[j];
        }
    }
    result
}
