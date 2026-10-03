//! How many observations the ENN scale search draws, or none.

use std::num::NonZeroUsize;

use crate::error::ENNError;

/// Draw count in the default ENN surrogate config.
pub const DEFAULT_FIT_SAMPLES: NonZeroUsize = match NonZeroUsize::new(10) {
    Some(n) => n,
    None => panic!("10 is nonzero"),
};

/// Fit-sample policy for an ENN surrogate.
///
/// `Frozen` is the Python `num_fit_samples is None` contract: skip the scale search and
/// keep epistemic scale 1 and aleatoric scale 0. A zero draw count is not a variant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitSamples {
    /// Skip the scale search.
    Frozen,
    /// Run the scale search on this many sampled observations.
    Draw(NonZeroUsize),
}

impl FitSamples {
    /// `None` means frozen. Zero is rejected.
    pub fn from_count(count: Option<usize>) -> Result<Self, ENNError> {
        match count {
            None => Ok(Self::Frozen),
            Some(n) => NonZeroUsize::new(n).map(Self::Draw).ok_or_else(|| {
                ENNError::InvalidParameter("num_fit_samples must be > 0, got 0".into())
            }),
        }
    }

    /// Draw count, or `None` when frozen.
    pub fn count(self) -> Option<usize> {
        match self {
            Self::Frozen => None,
            Self::Draw(n) => Some(n.get()),
        }
    }

    /// Whether the scale search is skipped.
    pub fn is_frozen(self) -> bool {
        matches!(self, Self::Frozen)
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
    }
}
