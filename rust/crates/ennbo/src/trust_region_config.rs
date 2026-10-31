use crate::error::ENNError;
use crate::morbo_trust_region::MorboTRSettings;
use crate::trust_region::TRLengthConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrustRegionKind {
    Turbo,
    Morbo,
}

impl TrustRegionKind {
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
        assert_eq!(
            TrustRegionKind::parse("morbo").unwrap(),
            TrustRegionKind::Morbo
        );
        assert_eq!(
            TrustRegionKind::parse("turbo").unwrap(),
            TrustRegionKind::Turbo
        );
        assert!(TrustRegionKind::parse("MORBO").is_err());
    }
}
