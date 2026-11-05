use ndarray::Array2;

use crate::error::ENNError;
use crate::fit_samples::FitSamples;
use crate::layout::EnnLayout;
use crate::neighbor_count::NeighborCount;
use crate::params::ENNParams;

#[derive(Debug, Clone)]
pub struct ENNSurrogateConfig {
    pub k: NeighborCount,
    pub fit_samples: FitSamples,
    pub layout: EnnLayout,
    pub y_bounds: Option<Array2<f64>>,
    pub tied_dims: Vec<Vec<usize>>,
}

impl ENNSurrogateConfig {
    pub fn new(layout: EnnLayout) -> Self {
        Self {
            k: NeighborCount::default(),
            fit_samples: FitSamples::default(),
            layout,
            y_bounds: None,
            tied_dims: Vec::new(),
        }
    }

    pub fn validate(&self, num_dim: usize) -> Result<(), ENNError> {
        if self.tied_dims.is_empty() {
            return Ok(());
        }
        crate::metric_weights::require_weighting_for_ties(
            &self.tied_dims,
            self.layout.weighting(),
        )?;
        crate::metric_weights::validate_tied_dims(&self.tied_dims, num_dim)
    }

    pub(crate) fn default_params(&self) -> Result<ENNParams, ENNError> {
        ENNParams::new(self.k.get(), 1.0, 0.0).map_err(|e| {
            ENNError::InvalidParameter(format!("Failed to create default params: {e}"))
        })
    }
}
