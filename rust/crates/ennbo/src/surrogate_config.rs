use ndarray::Array2;

use crate::error::ENNError;
use crate::fit_samples::FitSamples;
use crate::layout::EnnLayout;
use crate::neighbor_count::NeighborCount;
use crate::output_blend::SemanticOutputs;
use crate::params::ENNParams;

#[derive(Debug, Clone)]
pub struct ENNSurrogateConfig {
    pub k: NeighborCount,
    pub fit_samples: FitSamples,
    pub layout: EnnLayout,
    pub y_bounds: Option<Array2<f64>>,
    pub tied_dims: Vec<Vec<usize>>,
    pub semantic_only: bool,
    pub semantic_projection: Option<Box<Array2<f64>>>,
    pub semantic_outputs: Option<Box<SemanticOutputs>>,
}

impl ENNSurrogateConfig {
    pub fn new(layout: EnnLayout) -> Self {
        Self {
            k: NeighborCount::default(),
            fit_samples: FitSamples::default(),
            layout,
            y_bounds: None,
            tied_dims: Vec::new(),
            semantic_only: false,
            semantic_projection: None,
            semantic_outputs: None,
        }
    }

    pub fn validate_outputs(&self, num_metrics: usize) -> Result<(), ENNError> {
        match &self.semantic_outputs {
            Some(outputs) if outputs.loadings.nrows() != num_metrics => {
                Err(ENNError::InvalidParameter(format!(
                    "output_projection has {} outputs but the optimizer has {num_metrics} metrics",
                    outputs.loadings.nrows()
                )))
            }
            _ => Ok(()),
        }
    }

    pub(crate) fn store_layout(&self) -> EnnLayout {
        let outputs = self.semantic_outputs.as_deref();
        self.layout
            .clone()
            .with_semantic_only(self.semantic_only)
            .with_output_identity(
                outputs.map(|o| crate::semantic::output_fingerprint(&o.loadings.view())),
                outputs.and_then(|o| o.alpha),
            )
    }

    pub fn validate(&self, num_dim: usize) -> Result<(), ENNError> {
        crate::semantic::validate(
            self.semantic_projection.as_deref(),
            num_dim,
            self.semantic_only,
        )?;
        if self.tied_dims.is_empty() {
            return Ok(());
        }
        crate::metric_weights::require_weighting_for_ties(
            &self.tied_dims,
            self.layout.weighting(),
        )?;
        let dimensions = crate::semantic::feature_count(
            self.semantic_projection.as_deref(),
            num_dim,
            self.semantic_only,
        );
        crate::metric_weights::validate_tied_dims(&self.tied_dims, dimensions)
    }

    pub(crate) fn default_params(&self) -> Result<ENNParams, ENNError> {
        ENNParams::new(self.k.get(), 1.0, 0.0).map_err(|e| {
            ENNError::InvalidParameter(format!("Failed to create default params: {e}"))
        })
    }
}
