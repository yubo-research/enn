//! ENN-surrogate fields that an optimizer override may replace.

use std::num::NonZeroUsize;
use std::path::PathBuf;

use crate::backend::EnnStorage;
use crate::error::ENNError;
use crate::fit_samples::{FitSamples, ScaleSearch};
use crate::index::IndexDriver;
use crate::layout::EnnLayout;
use crate::metric_auto::MetricLearning;
use crate::surrogate::ENNSurrogateConfig;

/// Scale-search overrides. Each `Some` field replaces the search setting.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SearchOverrides {
    pub num_fit_samples: Option<NonZeroUsize>,
    pub num_fit_candidates: Option<NonZeroUsize>,
    pub infer_aleatoric_variance: Option<bool>,
    pub affine_calibrate: Option<bool>,
}

impl SearchOverrides {
    /// Whether no field is set.
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// Search settings of `base` with every set field replaced. A frozen `base` needs
    /// `num_fit_samples`, because frozen scales have no search settings to adjust.
    pub fn apply(&self, base: FitSamples) -> Result<ScaleSearch, ENNError> {
        let mut search = match (base.search(), self.num_fit_samples) {
            (Some(s), Some(n)) => ScaleSearch { num_fit_samples: n, ..s },
            (Some(s), None) => s,
            (None, Some(n)) => ScaleSearch::with_samples(n),
            (None, None) => {
                return Err(ENNError::InvalidParameter(
                    "scale-search overrides on frozen scales require num_fit_samples".into(),
                ))
            }
        };
        if let Some(nfc) = self.num_fit_candidates {
            search.num_fit_candidates = nfc;
        }
        if let Some(ale) = self.infer_aleatoric_variance {
            search.infer_aleatoric_variance = ale;
        }
        if let Some(cal) = self.affine_calibrate {
            search.affine_calibrate = cal;
        }
        Ok(search)
    }
}

/// Override of the fit-sample policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitOverride {
    /// Freeze the scales. Search settings do not exist in this variant.
    Frozen,
    /// Run the scale search, adjusting the base settings.
    Draw(SearchOverrides),
}

/// Overrides for an [`ENNSurrogateConfig`]. Each `Some` field replaces the config value.
#[derive(Debug, Clone, Default)]
pub struct EnnOverrides {
    pub index_driver: Option<IndexDriver>,
    pub fit: Option<FitOverride>,
    pub scale_x: Option<bool>,
    pub y_bounds: Option<ndarray::Array2<f64>>,
    pub enn_storage: Option<EnnStorage>,
    pub work_dir: Option<PathBuf>,
    pub metric_learning: Option<MetricLearning>,
    pub tied_dims: Option<Vec<Vec<usize>>>,
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
        match self.fit {
            None => {}
            Some(FitOverride::Frozen) => enn.fit_samples = FitSamples::Frozen,
            Some(FitOverride::Draw(search)) => {
                enn.fit_samples = FitSamples::Draw(search.apply(enn.fit_samples)?);
            }
        }
        enn.layout = self.layout(&enn.layout)?;
        if let Some(yb) = self.y_bounds.clone() {
            enn.y_bounds = Some(yb);
        }
        if let Some(tied) = self.tied_dims.clone() {
            enn.tied_dims = tied;
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
            fit: Some(FitOverride::Frozen),
            scale_x: Some(true),
            tied_dims: Some(vec![vec![0, 1]]),
            ..Default::default()
        };
        let out = overrides.apply(&base).unwrap();
        assert_eq!(out.fit_samples, FitSamples::Frozen);
        assert!(out.layout.scale_x());
        assert_eq!(out.tied_dims, vec![vec![0, 1]]);
        assert_eq!(out.k, base.k);
    }

    #[test]
    fn search_overrides_keep_base_settings_and_need_a_count_when_frozen() {
        let only_cal = SearchOverrides {
            affine_calibrate: Some(true),
            ..Default::default()
        };
        assert!(!only_cal.is_empty());
        assert!(SearchOverrides::default().is_empty());
        let base = FitSamples::draw(5, 7).unwrap();
        let s = only_cal.apply(base).unwrap();
        assert_eq!((s.num_fit_samples.get(), s.num_fit_candidates.get()), (5, 7));
        assert!(s.affine_calibrate);
        assert!(only_cal.apply(FitSamples::Frozen).is_err());
        let counted = SearchOverrides {
            num_fit_samples: NonZeroUsize::new(3),
            infer_aleatoric_variance: Some(false),
            ..only_cal
        };
        let s = counted.apply(FitSamples::Frozen).unwrap();
        assert_eq!(s.num_fit_samples.get(), 3);
        assert!(!s.infer_aleatoric_variance && s.affine_calibrate);
        assert_eq!(counted.apply(base).unwrap().num_fit_candidates.get(), 7);
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
