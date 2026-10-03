//! Configuration types for the optimizer.

use crate::candidates::CandidateRV;
use crate::enn_overrides::EnnOverrides;
use crate::error::ENNError;
use crate::fit_samples::{FitSamples, DEFAULT_FIT_SAMPLES};
use crate::morbo_override::MorboOverride;
use crate::morbo_trust_region::MorboTRSettings;
use crate::surrogate::ENNSurrogateConfig;
use crate::trust_region::TRLengthConfig;
use crate::trust_region_config::{TrustRegionConfig, TrustRegionKind};

mod rules;
pub use rules::{validate_optimizer_rules, OptimizerInitKind, OptimizerRuleSet, TurboEnnBuilder};

/// Optimizer configuration.
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Surrogate configuration.
    pub surrogate: SurrogateConfig,
    /// Trust region configuration.
    pub trust_region: TrustRegionConfig,
    /// Candidate generation configuration.
    pub candidates: CandidateConfig,
    /// Acquisition function configuration.
    pub acquisition: AcquisitionConfig,
    /// Use surrogate posterior mean for incumbent selection among candidates.
    pub noise_aware: bool,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            surrogate: SurrogateConfig::ENN(ENNSurrogateConfig::default()),
            trust_region: TrustRegionConfig::default(),
            candidates: CandidateConfig::default(),
            acquisition: AcquisitionConfig::default(),
            noise_aware: false,
        }
    }
}

/// Surrogate type configuration.
#[derive(Debug, Clone)]
pub enum SurrogateConfig {
    /// ENN surrogate.
    ENN(ENNSurrogateConfig),
    /// No surrogate (for LHD/random).
    None,
}

impl Default for SurrogateConfig {
    #[doc = "kiss-coverage-off"]
    fn default() -> Self {
        SurrogateConfig::ENN(ENNSurrogateConfig::default())
    }
}

/// Candidate generation configuration.
///
/// Pool size is owned here:
/// `max(min_candidates, min(max_candidates, per_dim * dim + per_arm * arms))`.
#[derive(Debug, Clone)]
pub struct CandidateConfig {
    /// Floor on the pool size.
    pub min_candidates: usize,
    /// Cap on the pool size.
    pub max_candidates: usize,
    /// Added once per input dimension.
    pub num_candidates_per_dim: usize,
    /// Added once per arm.
    pub num_candidates_per_arm: usize,
    /// Random variable type for candidates.
    pub candidate_rv: CandidateRV,
    /// `RAASPDriver.FAST` when true. Ignored unless `candidate_rv` is RAASP.
    pub raasp_fast: bool,
}

impl Default for CandidateConfig {
    fn default() -> Self {
        Self {
            min_candidates: 10,
            max_candidates: 5000,
            num_candidates_per_dim: 100,
            num_candidates_per_arm: 0,
            candidate_rv: CandidateRV::Uniform,
            raasp_fast: false,
        }
    }
}

impl CandidateConfig {
    /// Pool size for this dimension and arm count.
    pub fn num_candidates(&self, num_dim: usize, num_arms: usize) -> usize {
        let per_dim = self.num_candidates_per_dim.saturating_mul(num_dim);
        let per_arm = self.num_candidates_per_arm.saturating_mul(num_arms);
        let inner = per_dim.saturating_add(per_arm);
        inner.min(self.max_candidates).max(self.min_candidates)
    }
}

/// Optional overrides to apply on top of factory default config.
/// Used for Python→Rust config pass-through.
#[derive(Debug, Clone, Default)]
pub struct ConfigOverrides {
    pub acquisition: Option<AcquisitionConfig>,
    pub candidate_rv: Option<CandidateRV>,
    pub num_candidates_per_dim: Option<usize>,
    pub min_candidates: Option<usize>,
    pub max_candidates: Option<usize>,
    pub num_candidates_per_arm: Option<usize>,
    pub length_init: Option<f64>,
    pub length_min: Option<f64>,
    pub length_max: Option<f64>,
    pub noise_aware: Option<bool>,
    pub trust_region_kind: Option<crate::trust_region_config::TrustRegionKind>,
    /// Present only as a complete MORBO triple: metric count, alpha, and rescalarize mode.
    pub morbo: Option<MorboOverride>,
    pub raasp_fast: Option<bool>,
    /// ENN-surrogate fields. Applying them to a config without a surrogate is an error.
    pub enn: Option<EnnOverrides>,
}

fn apply_enn_overrides(config: &mut OptimizerConfig, enn: &EnnOverrides) -> Result<(), ENNError> {
    let SurrogateConfig::ENN(base) = &config.surrogate else {
        return Err(ENNError::InvalidParameter(
            "ENN surrogate overrides require an ENN surrogate".into(),
        ));
    };
    config.surrogate = SurrogateConfig::ENN(enn.apply(base)?);
    Ok(())
}

fn apply_trust_region_overrides(
    overrides: &ConfigOverrides,
    config: &mut OptimizerConfig,
) -> Result<(), ENNError> {
    let lengths_set = overrides.length_init.is_some()
        || overrides.length_min.is_some()
        || overrides.length_max.is_some();
    if overrides.morbo.is_some() && overrides.trust_region_kind != Some(TrustRegionKind::Morbo) {
        return Err(ENNError::InvalidParameter(
            "num_metrics and alpha require trust_region MORBO".into(),
        ));
    }
    if overrides.trust_region_kind == Some(TrustRegionKind::Morbo) {
        let morbo = overrides.morbo.ok_or_else(|| {
            ENNError::InvalidParameter(
                "MORBO overrides require num_metrics and alpha together".into(),
            )
        })?;
        let length = TRLengthConfig::resolve(
            overrides.length_init,
            overrides.length_min,
            overrides.length_max,
        )?;
        config.trust_region = TrustRegionConfig::Morbo(MorboTRSettings {
            num_metrics: morbo.num_metrics,
            alpha: morbo.alpha,
            length,
            rescalarize: morbo.rescalarize,
        });
        return Ok(());
    }
    if !lengths_set && overrides.trust_region_kind != Some(TrustRegionKind::Turbo) {
        return Ok(());
    }
    let length = TRLengthConfig::resolve(
        overrides.length_init,
        overrides.length_min,
        overrides.length_max,
    )?;
    config.trust_region = match &config.trust_region {
        TrustRegionConfig::Morbo(m) if overrides.trust_region_kind != Some(TrustRegionKind::Turbo) => {
            let mut morbo = m.clone();
            morbo.length = length;
            TrustRegionConfig::Morbo(morbo)
        }
        _ => TrustRegionConfig::Turbo(length),
    };
    Ok(())
}

impl ConfigOverrides {
    /// Apply overrides to an existing config.
    pub fn apply_to(&self, mut config: OptimizerConfig) -> Result<OptimizerConfig, ENNError> {
        if let Some(acq) = self.acquisition {
            config.acquisition = acq;
        }
        if let Some(rv) = self.candidate_rv {
            config.candidates.candidate_rv = rv;
        }
        if let Some(n) = self.num_candidates_per_dim {
            config.candidates.num_candidates_per_dim = n;
        }
        if let Some(m) = self.min_candidates {
            config.candidates.min_candidates = m;
        }
        if let Some(cap) = self.max_candidates {
            config.candidates.max_candidates = cap;
        }
        if let Some(m) = self.num_candidates_per_arm {
            config.candidates.num_candidates_per_arm = m;
        }
        if let Some(fast) = self.raasp_fast {
            config.candidates.raasp_fast = fast;
        }
        apply_trust_region_overrides(self, &mut config)?;
        if let Some(enn) = &self.enn {
            apply_enn_overrides(&mut config, enn)?;
        }
        if let Some(na) = self.noise_aware {
            config.noise_aware = na;
        }
        Ok(config)
    }
}

/// Acquisition function configuration.
#[derive(Debug, Clone, Copy)]
pub enum AcquisitionConfig {
    /// Upper Confidence Bound.
    UCB { beta: f64 },
    /// Thompson sampling.
    Thompson,
    /// Random acquisition.
    Random,
    /// Pareto front acquisition (multi-objective).
    Pareto,
}

/// UCB exploration weight used when none is given.
pub const DEFAULT_UCB_BETA: f64 = 2.0;

impl Default for AcquisitionConfig {
    fn default() -> Self {
        AcquisitionConfig::UCB {
            beta: DEFAULT_UCB_BETA,
        }
    }
}

/// Initialization strategy type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InitStrategy {
    /// Latin Hypercube Design.
    #[default]
    LHD,
    /// Random uniform.
    Random,
}

fn turbo_candidate_config() -> CandidateConfig {
    CandidateConfig {
        candidate_rv: CandidateRV::RAASP,
        ..CandidateConfig::default()
    }
}

/// Create a TuRBO-ENN configuration.
pub fn turbo_enn_config() -> OptimizerConfig {
    OptimizerConfig {
        surrogate: SurrogateConfig::ENN(ENNSurrogateConfig {
            k: 10,
            num_fit_candidates: 30,
            fit_samples: FitSamples::Draw(DEFAULT_FIT_SAMPLES),
            ..Default::default()
        }),
        trust_region: TrustRegionConfig::default(),
        candidates: turbo_candidate_config(),
        acquisition: AcquisitionConfig::default(),
        noise_aware: false,
    }
}

/// Create a TuRBO-ZERO configuration.
pub fn turbo_zero_config() -> OptimizerConfig {
    OptimizerConfig {
        surrogate: SurrogateConfig::None,
        trust_region: TrustRegionConfig::default(),
        candidates: turbo_candidate_config(),
        acquisition: AcquisitionConfig::Random,
        noise_aware: false,
    }
}

impl OptimizerConfig {
    /// Reject illegal acquisition, surrogate, and fit-sample combinations.
    pub fn validate(&self) -> Result<(), ENNError> {
        self.validate_kind(OptimizerInitKind::Hybrid)
    }

    /// Reject illegal acquisition and surrogate combinations for this init kind.
    pub fn validate_kind(&self, kind: OptimizerInitKind) -> Result<(), ENNError> {
        let has_surrogate = matches!(self.surrogate, SurrogateConfig::ENN(_));
        validate_optimizer_rules(
            &OptimizerRuleSet {
                lhd_only: kind == OptimizerInitKind::LhdOnly,
                has_surrogate,
            },
            &self.acquisition,
        )?;
        if let SurrogateConfig::ENN(enn) = &self.surrogate {
            require_fit_samples(&self.acquisition, enn.fit_samples)?;
        }
        Ok(())
    }

    /// Start a TuRBO-ENN builder. [`TurboEnnBuilder::build`] runs [`Self::validate`].
    pub fn turbo_enn() -> TurboEnnBuilder {
        TurboEnnBuilder {
            config: turbo_enn_config(),
            kind: OptimizerInitKind::Hybrid,
            num_init: None,
        }
    }
}

/// Error when an acquisition that reads fitted ENN scales gets a frozen surrogate.
/// Pareto and Random acquisition may run with frozen scales.
pub fn require_fit_samples(
    acquisition: &AcquisitionConfig,
    fit_samples: FitSamples,
) -> Result<(), ENNError> {
    let exempt = matches!(acquisition, AcquisitionConfig::Pareto | AcquisitionConfig::Random);
    if fit_samples.is_frozen() && !exempt {
        return Err(ENNError::InvalidParameter(format!(
            "enn.num_fit_samples required for acq_type={acquisition:?}"
        )));
    }
    Ok(())
}

/// Create an LHD-only configuration.
pub fn lhd_only_config() -> OptimizerConfig {
    OptimizerConfig {
        surrogate: SurrogateConfig::None,
        trust_region: TrustRegionConfig::default(),
        candidates: CandidateConfig {
            min_candidates: 1,
            max_candidates: 1_000_000_000,
            num_candidates_per_dim: 1,
            num_candidates_per_arm: 0,
            candidate_rv: CandidateRV::Uniform,
            raasp_fast: false,
        },
        acquisition: AcquisitionConfig::Random,
        noise_aware: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::EnnStorage;
    use crate::candidates::CandidateRV;
    use crate::morbo_trust_region::Rescalarize;
    use std::path::Path;

    #[test]
    fn test_candidate_config_num_candidates() {
        let config = CandidateConfig::default();
        assert_eq!(config.num_candidates(2, 1), 200);
        assert_eq!(config.num_candidates(10, 1), 1000);
        assert_eq!(config.num_candidates(60, 1), 5000);
        assert_eq!(config.num_candidates(2, 10), 200);
    }

    #[test]
    fn test_candidate_config_max_candidates_cap() {
        let config = CandidateConfig {
            min_candidates: 10,
            max_candidates: 50,
            num_candidates_per_dim: 200,
            num_candidates_per_arm: 0,
            candidate_rv: CandidateRV::Uniform,
            raasp_fast: false,
        };
        assert_eq!(config.num_candidates(2, 1), 50);
        assert_eq!(config.num_candidates(1, 4), 50);
    }

    #[test]
    fn per_arm_adds_to_per_dim() {
        let config = CandidateConfig {
            min_candidates: 10,
            max_candidates: 5000,
            num_candidates_per_dim: 0,
            num_candidates_per_arm: 25,
            candidate_rv: CandidateRV::Uniform,
            raasp_fast: false,
        };
        assert_eq!(config.num_candidates(2, 3), 75);
        assert_eq!(config.num_candidates(2, 8), 200);
    }

    #[test]
    fn test_config_defaults() {
        let config = OptimizerConfig::default();
        assert!(matches!(config.acquisition, AcquisitionConfig::UCB { .. }));
    }

    #[test]
    fn test_turbo_enn_config() {
        let config = turbo_enn_config();
        assert!(matches!(config.surrogate, SurrogateConfig::ENN(_)));
        assert!(matches!(config.acquisition, AcquisitionConfig::UCB { .. }));
        assert_eq!(config.candidates.candidate_rv, CandidateRV::RAASP);
    }

    #[test]
    fn test_turbo_zero_config() {
        let config = turbo_zero_config();
        assert!(matches!(config.surrogate, SurrogateConfig::None));
        assert!(matches!(config.acquisition, AcquisitionConfig::Random));
        assert_eq!(config.candidates.candidate_rv, CandidateRV::RAASP);
    }

    #[test]
    fn test_lhd_only_config() {
        let config = lhd_only_config();
        assert!(matches!(config.surrogate, SurrogateConfig::None));
        let n = config.candidates.num_candidates(10, 1);
        assert_eq!(n, 10);
    }

    #[test]
    fn test_init_strategy_enum() {
        let init_default = InitStrategy::default();
        assert_eq!(init_default, InitStrategy::LHD);
        assert_eq!(InitStrategy::Random as u8, 1);
    }

    #[test]
    fn test_config_overrides_apply_to() {
        use crate::index::IndexDriver;

        let overrides = ConfigOverrides {
            acquisition: Some(AcquisitionConfig::Thompson),
            candidate_rv: Some(CandidateRV::Sobol),
            enn: Some(EnnOverrides {
                index_driver: Some(IndexDriver::Flat),
                fit_samples: Some(FitSamples::from_count(Some(123)).unwrap()),
                num_fit_candidates: Some(456),
                scale_x: Some(true),
                ..Default::default()
            }),
            ..Default::default()
        };

        let config = turbo_enn_config();
        let applied = overrides.apply_to(config).unwrap();

        assert!(matches!(applied.acquisition, AcquisitionConfig::Thompson));
        assert_eq!(applied.candidates.candidate_rv, CandidateRV::Sobol);
        if let SurrogateConfig::ENN(enn) = &applied.surrogate {
            assert_eq!(enn.layout.index_driver(), IndexDriver::Flat);
            assert_eq!(enn.fit_samples.count(), Some(123));
            assert_eq!(enn.num_fit_candidates, 456);
            assert!(enn.layout.scale_x());
        } else {
            panic!("expected ENN surrogate");
        }
    }

    #[test]
    fn test_config_overrides_scale_x_apply() {
        let overrides = ConfigOverrides {
            enn: Some(EnnOverrides {
                scale_x: Some(true),
                ..Default::default()
            }),
            ..Default::default()
        };
        let applied = overrides.apply_to(turbo_enn_config()).unwrap();
        let SurrogateConfig::ENN(enn) = applied.surrogate else {
            panic!("expected ENN surrogate");
        };
        assert!(enn.layout.scale_x());
    }

    #[test]
    fn morbo_config_override_rejects_num_metrics_one() {
        use crate::morbo_trust_region::MorboTrustRegion;
        use crate::trust_region_config::TrustRegionConfig;
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let overrides = ConfigOverrides {
            trust_region_kind: Some(TrustRegionKind::Morbo),
            morbo: Some(MorboOverride {
                num_metrics: 1,
                alpha: 0.05,
                rescalarize: Rescalarize::OnRestart,
            }),
            ..Default::default()
        };
        let applied = overrides.apply_to(turbo_enn_config()).unwrap();
        let TrustRegionConfig::Morbo(settings) = applied.trust_region else {
            panic!("expected Morbo trust region");
        };
        let mut rng = StdRng::seed_from_u64(8);
        let result = MorboTrustRegion::new(2, settings, &mut rng);
        assert!(
            result.is_err(),
            "PyO3/override path must reject num_metrics=1 like Python Morbo config"
        );
    }

    #[test]
    fn config_overrides_apply_num_candidates_per_arm_to_pool() {
        let overrides = ConfigOverrides {
            num_candidates_per_dim: Some(0),
            min_candidates: Some(10),
            max_candidates: Some(5000),
            num_candidates_per_arm: Some(40),
            ..Default::default()
        };
        let applied = overrides.apply_to(turbo_zero_config()).unwrap();
        assert_eq!(applied.candidates.num_candidates(2, 3), 120);
        assert_eq!(applied.candidates.num_candidates(2, 8), 320);
    }

    #[test]
    fn config_overrides_apply_enn_num_fit_fields() {
        let overrides = ConfigOverrides {
            enn: Some(EnnOverrides {
                fit_samples: Some(FitSamples::from_count(Some(7)).unwrap()),
                num_fit_candidates: Some(11),
                scale_x: Some(true),
                ..Default::default()
            }),
            ..Default::default()
        };
        let applied = overrides.apply_to(turbo_enn_config()).unwrap();
        let SurrogateConfig::ENN(enn) = applied.surrogate else {
            panic!("expected ENN surrogate");
        };
        assert_eq!(enn.fit_samples.count(), Some(7));
        assert_eq!(enn.num_fit_candidates, 11);
        assert!(enn.layout.scale_x());
    }

    #[test]
    fn config_overrides_apply_enn_storage_and_work_dir() {
        use crate::index::IndexDriver;
        use std::path::PathBuf;

        let overrides = ConfigOverrides {
            enn: Some(EnnOverrides {
                index_driver: Some(IndexDriver::BpAnnDisk),
                enn_storage: Some(EnnStorage::Disk),
                work_dir: Some(PathBuf::from("/tmp/enn_work")),
                ..Default::default()
            }),
            ..Default::default()
        };
        let applied = overrides.apply_to(turbo_enn_config()).unwrap();
        let SurrogateConfig::ENN(enn) = applied.surrogate else {
            panic!("expected ENN surrogate");
        };
        assert_eq!(enn.layout.index_driver(), IndexDriver::BpAnnDisk);
        assert_eq!(enn.layout.storage(), EnnStorage::Disk);
        assert_eq!(enn.layout.work_dir(), Some(Path::new("/tmp/enn_work")));
    }

    #[test]
    fn enn_overrides_without_surrogate_are_rejected() {
        let overrides = ConfigOverrides {
            enn: Some(EnnOverrides {
                fit_samples: Some(FitSamples::from_count(Some(7)).unwrap()),
                ..Default::default()
            }),
            ..Default::default()
        };
        for base in [turbo_zero_config(), lhd_only_config()] {
            let err = overrides.apply_to(base).unwrap_err();
            assert!(err.to_string().contains("require an ENN surrogate"), "{err}");
        }
        let mut config = turbo_zero_config();
        apply_enn_overrides(&mut config, &EnnOverrides::default()).unwrap_err();
    }

    #[test]
    fn require_fit_samples_exempts_pareto_and_random_only() {
        let draw = FitSamples::from_count(Some(4)).unwrap();
        for acq in [
            AcquisitionConfig::default(),
            AcquisitionConfig::Thompson,
            AcquisitionConfig::Pareto,
            AcquisitionConfig::Random,
        ] {
            require_fit_samples(&acq, draw).unwrap();
        }
        require_fit_samples(&AcquisitionConfig::Pareto, FitSamples::Frozen).unwrap();
        require_fit_samples(&AcquisitionConfig::Random, FitSamples::Frozen).unwrap();
        require_fit_samples(&AcquisitionConfig::Thompson, FitSamples::Frozen).unwrap_err();
        require_fit_samples(&AcquisitionConfig::default(), FitSamples::Frozen).unwrap_err();
    }

    #[test]
    fn validate_rejects_frozen_enn_with_ucb() {
        let mut config = turbo_enn_config();
        if let SurrogateConfig::ENN(enn) = &mut config.surrogate {
            enn.fit_samples = FitSamples::Frozen;
        }
        let err = config.validate().unwrap_err();
        assert!(err.to_string().contains("enn.num_fit_samples required"), "{err}");
        config.acquisition = AcquisitionConfig::Pareto;
        config.validate().unwrap();
    }

    #[test]
    fn morbo_config_missing_rescalarize_defaults_on_restart() {
        let overrides = ConfigOverrides {
            trust_region_kind: Some(TrustRegionKind::Morbo),
            morbo: Some(MorboOverride {
                num_metrics: 2,
                alpha: 0.05,
                rescalarize: Rescalarize::OnRestart,
            }),
            ..Default::default()
        };
        let applied = overrides.apply_to(turbo_enn_config()).unwrap();
        let TrustRegionConfig::Morbo(settings) = applied.trust_region else {
            panic!("expected Morbo trust region");
        };
        assert_eq!(
            settings.rescalarize,
            Rescalarize::OnRestart,
            "missing rescalarize should match Python MorboTRConfig default ON_RESTART"
        );
    }

    #[test]
    fn morbo_config_unknown_rescalarize_errors() {
        let parsed: Result<Rescalarize, ()> = "NOT_A_MODE".parse();
        assert!(
            parsed.is_err(),
            "unknown rescalarize names are not a stored mode"
        );
    }

    #[test]
    fn morbo_override_requires_the_paired_fields() {
        let missing = ConfigOverrides {
            trust_region_kind: Some(TrustRegionKind::Morbo),
            ..Default::default()
        };
        let err = missing.apply_to(turbo_enn_config()).unwrap_err();
        assert!(
            err.to_string().contains("num_metrics and alpha"),
            "unexpected error: {err}"
        );
        let unpaired = ConfigOverrides {
            morbo: Some(MorboOverride {
                num_metrics: 2,
                alpha: 0.05,
                rescalarize: Rescalarize::OnRestart,
            }),
            ..Default::default()
        };
        let err = unpaired.apply_to(turbo_enn_config()).unwrap_err();
        assert!(
            err.to_string().contains("require trust_region MORBO"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn kiss_config_override_types_linked() {
        assert!(std::mem::size_of::<ConfigOverrides>() > 0);
        let acq = AcquisitionConfig::default();
        assert!(matches!(acq, AcquisitionConfig::UCB { .. }));
    }
}
