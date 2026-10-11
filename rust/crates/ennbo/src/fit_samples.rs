use std::num::NonZeroUsize;

use crate::error::ENNError;

pub const DEFAULT_FIT_SAMPLES: NonZeroUsize = match NonZeroUsize::new(10) {
    Some(n) => n,
    None => panic!("10 is nonzero"),
};

pub const DEFAULT_FIT_CANDIDATES: NonZeroUsize = match NonZeroUsize::new(30) {
    Some(n) => n,
    None => panic!("30 is nonzero"),
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScaleSearch {
    pub num_fit_samples: NonZeroUsize,
    pub num_fit_candidates: NonZeroUsize,
    pub infer_aleatoric_variance: bool,
    pub affine_calibrate: bool,
}

impl ScaleSearch {
    pub fn with_samples(num_fit_samples: NonZeroUsize) -> Self {
        Self {
            num_fit_samples,
            num_fit_candidates: DEFAULT_FIT_CANDIDATES,
            infer_aleatoric_variance: true,
            affine_calibrate: false,
        }
    }

    pub fn from_counts(
        num_fit_samples: Option<usize>,
        num_fit_candidates: Option<usize>,
    ) -> Result<Self, ENNError> {
        let samples = num_fit_samples.map(nonzero_samples).transpose()?;
        let candidates = num_fit_candidates.map(nonzero_candidates).transpose()?;
        Ok(Self {
            num_fit_candidates: candidates.unwrap_or(DEFAULT_FIT_CANDIDATES),
            ..Self::with_samples(samples.unwrap_or(DEFAULT_FIT_SAMPLES))
        })
    }
}

impl Default for ScaleSearch {
    fn default() -> Self {
        Self::with_samples(DEFAULT_FIT_SAMPLES)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitSamples {
    Frozen,
    Draw(ScaleSearch),
}

impl Default for FitSamples {
    fn default() -> Self {
        Self::Draw(ScaleSearch::default())
    }
}

impl FitSamples {
    pub fn from_count(count: Option<usize>) -> Result<Self, ENNError> {
        match count {
            None => Ok(Self::Frozen),
            Some(n) => Ok(Self::Draw(ScaleSearch::with_samples(nonzero_samples(n)?))),
        }
    }

    pub fn draw(num_fit_samples: usize, num_fit_candidates: usize) -> Result<Self, ENNError> {
        ScaleSearch::from_counts(Some(num_fit_samples), Some(num_fit_candidates)).map(Self::Draw)
    }

    pub fn count(self) -> Option<usize> {
        self.search().map(|s| s.num_fit_samples.get())
    }

    pub fn search(self) -> Option<ScaleSearch> {
        match self {
            Self::Frozen => None,
            Self::Draw(s) => Some(s),
        }
    }

    pub fn is_frozen(self) -> bool {
        matches!(self, Self::Frozen)
    }
}

pub fn nonzero_samples(n: usize) -> Result<NonZeroUsize, ENNError> {
    NonZeroUsize::new(n)
        .ok_or_else(|| ENNError::InvalidParameter("num_fit_samples must be > 0, got 0".into()))
}

pub fn nonzero_candidates(n: usize) -> Result<NonZeroUsize, ENNError> {
    NonZeroUsize::new(n)
        .ok_or_else(|| ENNError::InvalidParameter("num_fit_candidates must be > 0, got 0".into()))
}

#[cfg(test)]
pub(crate) fn test_search(
    num_fit_samples: usize,
    num_fit_candidates: usize,
    infer_aleatoric_variance: bool,
) -> ScaleSearch {
    ScaleSearch {
        infer_aleatoric_variance,
        ..ScaleSearch::from_counts(Some(num_fit_samples), Some(num_fit_candidates)).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_count_maps_none_to_frozen_and_rejects_zero() {
        assert_eq!(FitSamples::from_count(None).unwrap(), FitSamples::Frozen);
        assert!(FitSamples::from_count(Some(0)).is_err());
        let draw = FitSamples::from_count(Some(7)).unwrap();
        assert_eq!(draw.count(), Some(7));
        assert!(!draw.is_frozen());
        assert_eq!(FitSamples::Frozen.count(), None);
        assert!(FitSamples::Frozen.is_frozen());
        assert!(FitSamples::Frozen.search().is_none());
    }

    #[test]
    fn draw_sets_counts_and_keeps_default_flags() {
        let s = FitSamples::draw(4, 9).unwrap().search().unwrap();
        assert_eq!(s.num_fit_samples.get(), 4);
        assert_eq!(s.num_fit_candidates.get(), 9);
        assert!(s.infer_aleatoric_variance);
        assert!(!s.affine_calibrate);
        assert!(FitSamples::draw(0, 9).is_err());
        let err = FitSamples::draw(4, 0).unwrap_err();
        assert!(err.to_string().contains("num_fit_candidates"), "{err}");
        assert_eq!(
            FitSamples::default().count(),
            Some(DEFAULT_FIT_SAMPLES.get())
        );
    }

    #[test]
    fn from_counts_fills_defaults_and_rejects_zero() {
        assert_eq!(
            ScaleSearch::from_counts(None, None).unwrap(),
            ScaleSearch::default()
        );
        let s = ScaleSearch::from_counts(None, Some(3)).unwrap();
        assert_eq!(s.num_fit_samples, DEFAULT_FIT_SAMPLES);
        assert_eq!(s.num_fit_candidates.get(), 3);
        assert!(ScaleSearch::from_counts(Some(0), None).is_err());
        assert!(ScaleSearch::from_counts(None, Some(0)).is_err());
    }
}
