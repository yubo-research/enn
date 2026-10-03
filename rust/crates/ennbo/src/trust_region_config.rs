//! Trust-region configuration variants for the optimizer.

use crate::error::ENNError;
use crate::morbo_trust_region::MorboTRSettings;
use crate::trust_region::TRLengthConfig;

/// Which trust-region family a config override selects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrustRegionKind {
    /// Single-objective TuRBO.
    Turbo,
    /// Multi-objective MORBO.
    Morbo,
}

impl TrustRegionKind {
    /// Accept only the two wire spellings Python sends.
    pub fn parse(name: &str) -> Result<Self, ENNError> {
        match name {
            "turbo" => Ok(Self::Turbo),
            "morbo" => Ok(Self::Morbo),
            _ => Err(ENNError::InvalidParameter(format!(
                "Unknown trust_region: {name}"
            ))),
        }
    }
}

/// Trust-region configuration (TuRBO or Morbo).
#[derive(Debug, Clone)]
pub enum TrustRegionConfig {
    Turbo(TRLengthConfig),
    Morbo(MorboTRSettings),
}

impl Default for TrustRegionConfig {
    fn default() -> Self {
        TrustRegionConfig::Turbo(TRLengthConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trust_region_kind_rejects_unknown_spelling() {
        assert!(TrustRegionKind::parse("not_morbo").is_err());
        assert_eq!(TrustRegionKind::parse("morbo").unwrap(), TrustRegionKind::Morbo);
        assert_eq!(TrustRegionKind::parse("turbo").unwrap(), TrustRegionKind::Turbo);
        assert!(TrustRegionKind::parse("MORBO").is_err());
    }
}
