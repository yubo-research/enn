//! How the ENN scale search runs, or that it does not run.

use std::num::NonZeroUsize;

use crate::error::ENNError;

/// Draw count in the default ENN surrogate config.
pub const DEFAULT_FIT_SAMPLES: NonZeroUsize = match NonZeroUsize::new(10) {
    Some(n) => n,
    None => panic!("10 is nonzero"),
};

/// Candidate count in the default ENN surrogate config.
pub const DEFAULT_FIT_CANDIDATES: usize = 30;

/// Settings read only by the scale search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ScaleSearch {
    /// Observations sampled to score each candidate.
    pub num_fit_samples: NonZeroUsize,
    /// Random scale candidates scored per search.
    pub num_fit_candidates: usize,
    /// Search the aleatoric scale too; otherwise it stays 0.
    pub infer_aleatoric_variance: bool,
    /// Fit an affine calibrator after the search.
    pub affine_calibrate: bool,
}

impl ScaleSearch {
    /// Default search with `num_fit_samples` draws.
    pub fn with_samples(num_fit_samples: NonZeroUsize) -> Self {
        Self {
            num_fit_samples,
            num_fit_candidates: DEFAULT_FIT_CANDIDATES,
            infer_aleatoric_variance: true,
            affine_calibrate: false,
        }
    }
}

impl Default for ScaleSearch {
    fn default() -> Self {
        Self::with_samples(DEFAULT_FIT_SAMPLES)
    }
}

/// Fit-sample policy for an ENN surrogate.
///
/// `Frozen` is the Python `num_fit_samples is None` contract: skip the scale search and
/// keep epistemic scale 1 and aleatoric scale 0. Search settings exist only under `Draw`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitSamples {
    /// Skip the scale search.
    Frozen,
    /// Run the scale search with these settings.
    Draw(ScaleSearch),
}

impl Default for FitSamples {
    fn default() -> Self {
        Self::Draw(ScaleSearch::default())
    }
}

impl FitSamples {
    /// `None` means frozen. `Some(n)` is a default search with `n` draws. Zero is rejected.
    pub fn from_count(count: Option<usize>) -> Result<Self, ENNError> {
        match count {
            None => Ok(Self::Frozen),
            Some(n) => Ok(Self::Draw(ScaleSearch::with_samples(nonzero_samples(n)?))),
        }
    }

    /// Default search with `num_fit_samples` draws and `num_fit_candidates` candidates.
    pub fn draw(num_fit_samples: usize, num_fit_candidates: usize) -> Result<Self, ENNError> {
        Ok(Self::Draw(ScaleSearch {
            num_fit_candidates,
            ..ScaleSearch::with_samples(nonzero_samples(num_fit_samples)?)
        }))
    }

    /// Draw count, or `None` when frozen.
    pub fn count(self) -> Option<usize> {
        self.search().map(|s| s.num_fit_samples.get())
    }

    /// Search settings, or `None` when frozen.
    pub fn search(self) -> Option<ScaleSearch> {
        match self {
            Self::Frozen => None,
            Self::Draw(s) => Some(s),
        }
    }

    /// Whether the scale search is skipped.
    pub fn is_frozen(self) -> bool {
        matches!(self, Self::Frozen)
    }
}

/// Reject a zero draw count.
pub fn nonzero_samples(n: usize) -> Result<NonZeroUsize, ENNError> {
    NonZeroUsize::new(n)
        .ok_or_else(|| ENNError::InvalidParameter("num_fit_samples must be > 0, got 0".into()))
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
        assert_eq!(s.num_fit_candidates, 9);
        assert!(s.infer_aleatoric_variance);
        assert!(!s.affine_calibrate);
        assert!(FitSamples::draw(0, 9).is_err());
        assert_eq!(FitSamples::default().count(), Some(DEFAULT_FIT_SAMPLES.get()));
    }
}
