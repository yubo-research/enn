use std::num::NonZeroUsize;

use crate::error::ENNError;
use crate::fit_samples::{FitSamples, ScaleSearch};
use crate::surrogate::ENNSurrogateConfig;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SearchOverrides {
    pub num_fit_samples: Option<NonZeroUsize>,
    pub num_fit_candidates: Option<NonZeroUsize>,
    pub infer_aleatoric_variance: Option<bool>,
    pub affine_calibrate: Option<bool>,
}

impl SearchOverrides {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    pub fn apply(&self, base: FitSamples) -> Result<ScaleSearch, ENNError> {
        let mut search = match (base.search(), self.num_fit_samples) {
            (Some(s), Some(n)) => ScaleSearch {
                num_fit_samples: n,
                ..s
            },
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum FitOverride {
    Frozen,
    Draw(SearchOverrides),
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct EnnOverrides {
    pub fit: Option<FitOverride>,
    pub weighting: Option<bool>,
    pub aggregate_duplicates: Option<bool>,
    #[serde(with = "crate::exact")]
    pub y_bounds: Option<ndarray::Array2<f64>>,
    pub tied_dims: Option<Vec<Vec<usize>>>,
}

impl EnnOverrides {
    pub fn apply(&self, base: &ENNSurrogateConfig) -> Result<ENNSurrogateConfig, ENNError> {
        let mut enn = base.clone();
        match self.fit {
            None => {}
            Some(FitOverride::Frozen) => enn.fit_samples = FitSamples::Frozen,
            Some(FitOverride::Draw(search)) => {
                enn.fit_samples = FitSamples::Draw(search.apply(enn.fit_samples)?);
            }
        }
        if let Some(weighting) = self.weighting {
            enn.layout = enn.layout.with_weighting(weighting);
        }
        if let Some(aggregate) = self.aggregate_duplicates {
            enn.layout = enn.layout.with_aggregate_duplicates(aggregate);
        }
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
        let dir = tempfile::tempdir().unwrap();
        let base =
            ENNSurrogateConfig::new(crate::layout::EnnLayout::new(dir.path(), false).unwrap());
        let overrides = EnnOverrides {
            fit: Some(FitOverride::Frozen),
            weighting: Some(true),
            tied_dims: Some(vec![vec![0, 1]]),
            ..Default::default()
        };
        let out = overrides.apply(&base).unwrap();
        assert_eq!(out.fit_samples, FitSamples::Frozen);
        assert!(out.layout.weighting());
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
        assert_eq!(
            (s.num_fit_samples.get(), s.num_fit_candidates.get()),
            (5, 7)
        );
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
}
