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
    pub has_trust_region: bool,
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
    if rules.lhd_only && rules.has_trust_region {
        return Err(ENNError::InvalidParameter(
            "init_strategy='lhd_only' requires NoTRConfig trust region".into(),
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

fn reject(message: String) -> ENNError {
    ENNError::InvalidParameter(message)
}

pub fn validate_candidate_counts(counts: &[(&str, f64)]) -> Result<(), ENNError> {
    match counts.iter().find(|(_, value)| *value < 0.0) {
        Some((name, value)) => Err(reject(format!("{name} must be >= 0, got {value}"))),
        None => Ok(()),
    }
}

pub fn validate_positive_option(name: &str, value: Option<f64>) -> Result<(), ENNError> {
    match value {
        Some(value) if value <= 0.0 => Err(reject(format!("{name} must be > 0, got {value}"))),
        _ => Ok(()),
    }
}

pub struct FitSettings {
    pub num_fit_samples: Option<f64>,
    pub num_fit_candidates: Option<f64>,
    pub infer_aleatoric_variance_scale: Option<bool>,
    pub affine_calibrate: bool,
}

pub fn validate_fit_settings(settings: &FitSettings) -> Result<(), ENNError> {
    validate_positive_option("num_fit_samples", settings.num_fit_samples)?;
    validate_positive_option("num_fit_candidates", settings.num_fit_candidates)?;
    if settings.num_fit_samples.is_some() {
        return Ok(());
    }
    let search_only = [
        ("num_fit_candidates", settings.num_fit_candidates.is_some()),
        (
            "infer_aleatoric_variance_scale",
            settings.infer_aleatoric_variance_scale.is_some(),
        ),
        ("affine_calibrate", settings.affine_calibrate),
    ];
    let set_names: Vec<&str> = search_only
        .iter()
        .filter(|(_, is_set)| *is_set)
        .map(|(name, _)| *name)
        .collect();
    if set_names.is_empty() {
        return Ok(());
    }
    Err(reject(format!(
        "{} require num_fit_samples; num_fit_samples=None freezes the scales",
        set_names.join(", ")
    )))
}

pub fn validate_multi_objective(num_metrics: f64, alpha: f64) -> Result<(), ENNError> {
    if num_metrics < 2.0 {
        return Err(reject(format!(
            "num_metrics must be >= 2 for MORBO, got {num_metrics}"
        )));
    }
    if alpha <= 0.0 {
        return Err(reject(format!("alpha must be > 0, got {alpha}")));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn candidate_counts_reject_the_first_negative_by_name() {
        assert!(validate_candidate_counts(&[("a", 0.0), ("b", 1.0)]).is_ok());
        let message = validate_candidate_counts(&[("a", 0.0), ("b", -1.0)])
            .unwrap_err()
            .to_string();
        assert!(message.contains("b must be >= 0, got -1"), "{message}");
    }

    #[test]
    fn positive_option_accepts_none_and_rejects_zero() {
        assert!(validate_positive_option("n", None).is_ok());
        assert!(validate_positive_option("n", Some(1.0)).is_ok());
        assert!(validate_positive_option("n", Some(0.0)).is_err());
    }

    #[test]
    fn fit_settings_name_every_option_that_needs_samples() {
        let settings = FitSettings {
            num_fit_samples: None,
            num_fit_candidates: Some(4.0),
            infer_aleatoric_variance_scale: None,
            affine_calibrate: true,
        };
        let message = validate_fit_settings(&settings).unwrap_err().to_string();
        assert!(
            message.contains("num_fit_candidates, affine_calibrate require num_fit_samples"),
            "{message}"
        );
        let sampled = FitSettings {
            num_fit_samples: Some(2.0),
            ..settings
        };
        assert!(validate_fit_settings(&sampled).is_ok());
    }

    #[test]
    fn multi_objective_requires_two_metrics_and_positive_alpha() {
        assert!(validate_multi_objective(2.0, 0.05).is_ok());
        assert!(validate_multi_objective(1.0, 0.05).is_err());
        assert!(validate_multi_objective(2.0, 0.0).is_err());
    }
}
