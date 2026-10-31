use super::{AcquisitionConfig, OptimizerConfig, SurrogateConfig};
use crate::error::ENNError;
use crate::surrogate::ENNSurrogateConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizerInitKind {
    Hybrid,
    LhdOnly,
}

pub struct TurboEnnBuilder {
    pub(super) config: OptimizerConfig,
    pub(super) kind: OptimizerInitKind,
    pub(super) num_init: Option<usize>,
}

impl TurboEnnBuilder {
    pub fn acquisition(mut self, acquisition: super::AcquisitionConfig) -> Self {
        self.config.acquisition = acquisition;
        self
    }

    pub fn candidates(mut self, candidates: super::CandidateConfig) -> Self {
        self.config.candidates = candidates;
        self
    }

    pub fn trust_region(mut self, trust_region: crate::TrustRegionConfig) -> Self {
        self.config.trust_region = trust_region;
        self
    }

    pub fn surrogate(mut self, surrogate: ENNSurrogateConfig) -> Self {
        self.config.surrogate = SurrogateConfig::ENN(surrogate);
        self
    }

    pub fn lhd_only(mut self) -> Self {
        self.kind = OptimizerInitKind::LhdOnly;
        self.config.surrogate = SurrogateConfig::None;
        self
    }

    pub fn num_init(mut self, num_init: Option<usize>) -> Self {
        self.num_init = num_init;
        self
    }

    pub fn build(self) -> Result<(OptimizerConfig, Option<usize>), ENNError> {
        self.config.validate_kind(self.kind)?;
        Ok((self.config, self.num_init))
    }
}

pub struct OptimizerRuleSet {
    pub lhd_only: bool,
    pub has_surrogate: bool,
}

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
