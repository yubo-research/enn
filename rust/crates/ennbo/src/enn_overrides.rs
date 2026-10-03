//! ENN-surrogate fields that an optimizer override may replace.

use std::path::PathBuf;

use crate::backend::EnnStorage;
use crate::error::ENNError;
use crate::fit_samples::FitSamples;
use crate::index::IndexDriver;
use crate::layout::EnnLayout;
use crate::metric_auto::MetricLearning;
use crate::surrogate::ENNSurrogateConfig;

/// Overrides for an [`ENNSurrogateConfig`]. Each `Some` field replaces the config value.
#[derive(Debug, Clone, Default)]
pub struct EnnOverrides {
    pub index_driver: Option<IndexDriver>,
    pub fit_samples: Option<FitSamples>,
    pub num_fit_candidates: Option<usize>,
    pub infer_aleatoric_variance: Option<bool>,
    pub scale_x: Option<bool>,
    pub y_bounds: Option<ndarray::Array2<f64>>,
    pub enn_storage: Option<EnnStorage>,
    pub work_dir: Option<PathBuf>,
    pub metric_learning: Option<MetricLearning>,
    pub tied_dims: Option<Vec<Vec<usize>>>,
    pub affine_calibrate: Option<bool>,
}

impl EnnOverrides {
    fn layout(&self, layout: &EnnLayout) -> Result<EnnLayout, ENNError> {
        EnnLayout::try_from_parts(
            self.index_driver.unwrap_or(layout.index_driver()),
            Some(self.enn_storage.unwrap_or(layout.storage())),
            self.work_dir
                .clone()
                .or_else(|| layout.work_dir().map(|p| p.to_path_buf())),
            self.scale_x.unwrap_or(layout.scale_x()),
            self.metric_learning.unwrap_or(layout.metric_learning()),
        )
    }

    /// Return `base` with every set field replaced.
    pub fn apply(&self, base: &ENNSurrogateConfig) -> Result<ENNSurrogateConfig, ENNError> {
        let mut enn = base.clone();
        if let Some(fs) = self.fit_samples {
            enn.fit_samples = fs;
        }
        if let Some(nfc) = self.num_fit_candidates {
            enn.num_fit_candidates = nfc;
        }
        if let Some(ale) = self.infer_aleatoric_variance {
            enn.infer_aleatoric_variance = ale;
        }
        enn.layout = self.layout(&enn.layout)?;
        if let Some(yb) = self.y_bounds.clone() {
            enn.y_bounds = Some(yb);
        }
        if let Some(tied) = self.tied_dims.clone() {
            enn.tied_dims = tied;
        }
        if let Some(cal) = self.affine_calibrate {
            enn.affine_calibrate = cal;
        }
        Ok(enn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apply_replaces_only_set_fields() {
        let base = ENNSurrogateConfig::default();
        let overrides = EnnOverrides {
            fit_samples: Some(FitSamples::Frozen),
            scale_x: Some(true),
            tied_dims: Some(vec![vec![0, 1]]),
            ..Default::default()
        };
        let out = overrides.apply(&base).unwrap();
        assert_eq!(out.fit_samples, FitSamples::Frozen);
        assert!(out.layout.scale_x());
        assert_eq!(out.tied_dims, vec![vec![0, 1]]);
        assert_eq!(out.num_fit_candidates, base.num_fit_candidates);
        assert_eq!(out.k, base.k);
    }

    #[test]
    fn layout_rejects_auto_metric_in_memory() {
        let overrides = EnnOverrides {
            metric_learning: Some(MetricLearning::Auto),
            ..Default::default()
        };
        assert!(overrides.layout(&ENNSurrogateConfig::default().layout).is_err());
    }
}
