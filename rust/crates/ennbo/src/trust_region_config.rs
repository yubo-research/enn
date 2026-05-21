//! Trust-region configuration variants for the optimizer.

use crate::morbo_trust_region::MorboTRSettings;
use crate::trust_region::TRLengthConfig;

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
