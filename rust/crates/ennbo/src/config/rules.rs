//! Init-kind and Python-facing optimizer checks, split out of `config.rs`.

use super::{AcquisitionConfig, OptimizerConfig, SurrogateConfig};
use crate::error::ENNError;
use crate::surrogate::ENNSurrogateConfig;

/// How the optimizer is initialized. Used by [`OptimizerConfig::validate_kind`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerInitKind {
    /// Hybrid or TuRBO initialization.
    Hybrid,
    /// Latin-hypercube initialization with no surrogate.
    LhdOnly,
}

/// Fluent optimizer configuration. Mirrors the Python factory keywords.
pub struct TurboEnnBuilder {
    pub(super) config: OptimizerConfig,
    pub(super) kind: OptimizerInitKind,
    pub(super) num_init: Option<usize>,
}

impl TurboEnnBuilder {
    /// Set the acquisition function.
    pub fn acquisition(mut self, acquisition: super::AcquisitionConfig) -> Self {
        self.config.acquisition = acquisition;
        self
    }

    /// Replace candidate generation settings.
    pub fn candidates(mut self, candidates: super::CandidateConfig) -> Self {
        self.config.candidates = candidates;
        self
    }

    /// Set the trust region. Pass [`crate::TrustRegionConfig::Morbo`] for MORBO.
    pub fn trust_region(mut self, trust_region: crate::TrustRegionConfig) -> Self {
        self.config.trust_region = trust_region;
        self
    }

    /// Install an ENN surrogate.
    pub fn surrogate(mut self, surrogate: ENNSurrogateConfig) -> Self {
        self.config.surrogate = SurrogateConfig::ENN(surrogate);
        self
    }

    /// Mark this config as LHD-only (no surrogate allowed).
    pub fn lhd_only(mut self) -> Self {
        self.kind = OptimizerInitKind::LhdOnly;
        self.config.surrogate = SurrogateConfig::None;
        self
    }

    /// Initialization budget. `None` means the factory default.
    pub fn num_init(mut self, num_init: Option<usize>) -> Self {
        self.num_init = num_init;
        self
    }

    /// Validate and return the config plus the optional init budget.
    pub fn build(self) -> Result<(OptimizerConfig, Option<usize>), ENNError> {
        self.config.validate_kind(self.kind)?;
        Ok((self.config, self.num_init))
    }
}

/// Flags for the Python dataclass checks. Field names match the historical errors.
pub struct OptimizerRuleSet {
    /// `init_strategy` is Latin-hypercube only.
    pub lhd_only: bool,
    /// A surrogate is installed.
    pub has_surrogate: bool,
}

/// Python dataclass checks. Names match the historical Python errors.
pub fn validate_optimizer_rules(
    rules: &OptimizerRuleSet,
    acquisition: &AcquisitionConfig,
) -> Result<(), ENNError> {
    if rules.lhd_only && rules.has_surrogate {
        return Err(ENNError::InvalidParameter(
            "init_strategy='lhd_only' requires NoSurrogateConfig surrogate".into(),
        ));
    }
    if !rules.has_surrogate && matches!(acquisition, AcquisitionConfig::Thompson) {
        return Err(ENNError::InvalidParameter(
            "DrawAcquisitionConfig (Thompson sampling) requires a surrogate. NoSurrogateConfig is not compatible with DrawAcquisitionConfig.".into(),
        ));
    }
    if !rules.has_surrogate && matches!(acquisition, AcquisitionConfig::UCB { .. }) {
        return Err(ENNError::InvalidParameter(
            "UCBAcquisitionConfig requires a surrogate. NoSurrogateConfig is not compatible with UCBAcquisitionConfig.".into(),
        ));
    }
    Ok(())
}
