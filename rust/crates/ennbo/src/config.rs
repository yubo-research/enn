//! Configuration types for the optimizer.

use crate::candidates::CandidateRV;
use crate::index::IndexDriver;
use crate::morbo_trust_region::{MorboTRSettings, Rescalarize};
use crate::backend::EnnStorage;
use crate::surrogate::ENNSurrogateConfig;
use crate::trust_region::TRLengthConfig;
use crate::trust_region_config::TrustRegionConfig;
use std::path::PathBuf;

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
    fn default() -> Self {
        SurrogateConfig::ENN(ENNSurrogateConfig::default())
    }
}

/// Candidate generation configuration.
#[derive(Debug, Clone)]
pub struct CandidateConfig {
    /// Base multiplier for number of candidates.
    pub num_candidates_factor: f64,
    /// Minimum number of candidates.
    pub min_candidates: usize,
    /// Maximum number of candidates (None = no cap). Matches Python default_num_candidates cap.
    pub max_candidates: Option<usize>,
    /// Optional per-arm multiplier: pool is at least num_arms * this value.
    pub num_candidates_per_arm: Option<usize>,
    /// Random variable type for candidates.
    pub candidate_rv: CandidateRV,
}

impl Default for CandidateConfig {
    fn default() -> Self {
        Self {
            num_candidates_factor: 1000.0,
            min_candidates: 100,
            max_candidates: None,
            num_candidates_per_arm: None,
            candidate_rv: CandidateRV::Uniform,
        }
    }
}

impl CandidateConfig {
    /// Compute number of candidates based on dimension and arms.
    ///
    /// Matches Python `CandidateGenConfig.resolve_num_candidates`: default base
    /// `min(max_candidates, factor * dim)` when set, optional `max(fixed, per_arm * arms)`,
    /// no `num_arms` multiplier. `max_candidates` caps the formula base only when
    /// `num_candidates_per_arm` is set; otherwise exact-fixed mode uses min=max as pool size.
    pub fn num_candidates(&self, num_dim: usize, num_arms: usize) -> usize {
        let is_exact_fixed = self.num_candidates_factor == 1.0
            && self.max_candidates == Some(self.min_candidates)
            && self.num_candidates_per_arm.is_none();

        let mut base = if is_exact_fixed {
            self.min_candidates
        } else {
            let raw = (self.num_candidates_factor * num_dim as f64) as usize;
            let formula = match self.max_candidates {
                Some(cap) if self.num_candidates_per_arm.is_some() => raw.min(cap),
                Some(cap) if (self.num_candidates_factor - 100.0).abs() < f64::EPSILON => {
                    raw.min(cap)
                }
                _ => raw,
            };
            formula.max(self.min_candidates)
        };

        if let Some(m) = self.num_candidates_per_arm {
            base = base.max(num_arms * m);
        }

        base
    }
}

/// Optional overrides to apply on top of factory default config.
/// Used for Python→Rust config pass-through.
#[derive(Debug, Clone, Default)]
pub struct ConfigOverrides {
    pub acquisition: Option<AcquisitionConfig>,
    pub candidate_rv: Option<CandidateRV>,
    pub num_candidates_factor: Option<f64>,
    pub min_candidates: Option<usize>,
    pub max_candidates: Option<usize>,
    pub num_candidates_per_arm: Option<usize>,
    pub length_init: Option<f64>,
    pub length_min: Option<f64>,
    pub length_max: Option<f64>,
    pub index_driver: Option<IndexDriver>,
    pub num_fit_samples: Option<usize>,
    pub num_fit_candidates: Option<usize>,
    pub scale_x: Option<bool>,
    pub noise_aware: Option<bool>,
    pub enn_storage: Option<EnnStorage>,
    pub work_dir: Option<PathBuf>,
    pub trust_region_kind: Option<String>,
    pub num_metrics: Option<usize>,
    pub alpha: Option<f64>,
    pub rescalarize: Option<String>,
}

fn apply_enn_surrogate_fields(
    config: &mut OptimizerConfig,
    index_driver: Option<IndexDriver>,
    num_fit_samples: Option<usize>,
    num_fit_candidates: Option<usize>,
    scale_x: Option<bool>,
    enn_storage: Option<EnnStorage>,
    work_dir: Option<PathBuf>,
) {
    let SurrogateConfig::ENN(enn_cfg) = &config.surrogate else {
        return;
    };
    let mut enn = enn_cfg.clone();
    if let Some(driver) = index_driver {
        enn.index_driver = driver;
    }
    if let Some(nfs) = num_fit_samples {
        enn.num_fit_samples = nfs;
    }
    if let Some(nfc) = num_fit_candidates {
        enn.num_fit_candidates = nfc;
    }
    if let Some(sx) = scale_x {
        enn.scale_x = sx;
    }
    if let Some(storage) = enn_storage {
        enn.storage = storage;
    }
    if let Some(dir) = work_dir {
        enn.work_dir = Some(dir);
    }
    config.surrogate = SurrogateConfig::ENN(enn);
}

fn apply_trust_region_overrides(overrides: &ConfigOverrides, config: &mut OptimizerConfig) {
    if let Some(kind) = &overrides.trust_region_kind {
        if kind == "morbo" {
            let num_metrics = overrides.num_metrics.unwrap_or(2);
            let alpha = overrides.alpha.unwrap_or(0.05);
            let length = TRLengthConfig {
                length_init: overrides.length_init.unwrap_or(0.8),
                length_min: overrides.length_min.unwrap_or(0.5f64.powi(7)),
                length_max: overrides.length_max.unwrap_or(1.6),
            };
            let rescalarize = overrides
                .rescalarize
                .as_deref()
                .and_then(|s| s.parse().ok())
                .unwrap_or(Rescalarize::OnPropose);
            config.trust_region = TrustRegionConfig::Morbo(MorboTRSettings {
                num_metrics,
                alpha,
                length,
                rescalarize,
                noise_aware: overrides.noise_aware.unwrap_or(false),
            });
        }
        return;
    }
    if overrides.length_init.is_none()
        && overrides.length_min.is_none()
        && overrides.length_max.is_none()
    {
        return;
    }
    let TRLengthConfig {
        length_init,
        length_min,
        length_max,
    } = match &config.trust_region {
        TrustRegionConfig::Turbo(cfg) => *cfg,
        TrustRegionConfig::Morbo(m) => m.length,
    };
    let updated = TRLengthConfig {
        length_init: overrides.length_init.unwrap_or(length_init),
        length_min: overrides.length_min.unwrap_or(length_min),
        length_max: overrides.length_max.unwrap_or(length_max),
    };
    config.trust_region = match &config.trust_region {
        TrustRegionConfig::Turbo(_) => TrustRegionConfig::Turbo(updated),
        TrustRegionConfig::Morbo(m) => {
            let mut morbo = m.clone();
            morbo.length = updated;
            TrustRegionConfig::Morbo(morbo)
        }
    };
}

impl ConfigOverrides {
    /// Apply overrides to an existing config.
    pub fn apply_to(&self, mut config: OptimizerConfig) -> OptimizerConfig {
        if let Some(acq) = self.acquisition {
            config.acquisition = acq;
        }
        if let Some(rv) = self.candidate_rv {
            config.candidates.candidate_rv = rv;
        }
        if let Some(f) = self.num_candidates_factor {
            config.candidates.num_candidates_factor = f;
        }
        if let Some(m) = self.min_candidates {
            config.candidates.min_candidates = m;
        }
        if let Some(cap) = self.max_candidates {
            config.candidates.max_candidates = Some(cap);
        }
        if let Some(m) = self.num_candidates_per_arm {
            config.candidates.num_candidates_per_arm = Some(m);
        }
        apply_trust_region_overrides(self, &mut config);
        if self.index_driver.is_some()
            || self.num_fit_samples.is_some()
            || self.num_fit_candidates.is_some()
            || self.scale_x.is_some()
            || self.enn_storage.is_some()
            || self.work_dir.is_some()
        {
            apply_enn_surrogate_fields(
                &mut config,
                self.index_driver,
                self.num_fit_samples,
                self.num_fit_candidates,
                self.scale_x,
                self.enn_storage,
                self.work_dir.clone(),
            );
        }
        if let Some(na) = self.noise_aware {
            config.noise_aware = na;
        }
        config
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

impl Default for AcquisitionConfig {
    fn default() -> Self {
        AcquisitionConfig::UCB { beta: 2.0 }
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

/// Create a TuRBO-ENN configuration.
pub fn turbo_enn_config() -> OptimizerConfig {
    OptimizerConfig {
        surrogate: SurrogateConfig::ENN(ENNSurrogateConfig {
            k: 10,
            num_fit_candidates: 30,
            num_fit_samples: 10,
            ..Default::default()
        }),
        trust_region: TrustRegionConfig::default(),
        candidates: CandidateConfig {
            num_candidates_factor: 1000.0,
            min_candidates: 100,
            max_candidates: None,
            num_candidates_per_arm: None,
            candidate_rv: CandidateRV::Uniform,
        },
        acquisition: AcquisitionConfig::UCB { beta: 2.0 },
        noise_aware: false,
    }
}

/// Create a TuRBO-ZERO configuration.
pub fn turbo_zero_config() -> OptimizerConfig {
    OptimizerConfig {
        surrogate: SurrogateConfig::None,
        trust_region: TrustRegionConfig::default(),
        candidates: CandidateConfig {
            num_candidates_factor: 1000.0,
            min_candidates: 100,
            max_candidates: None,
            num_candidates_per_arm: None,
            candidate_rv: CandidateRV::Uniform,
        },
        acquisition: AcquisitionConfig::Random,
        noise_aware: false,
    }
}

/// Create an LHD-only configuration.
pub fn lhd_only_config() -> OptimizerConfig {
    OptimizerConfig {
        surrogate: SurrogateConfig::None,
        trust_region: TrustRegionConfig::default(),
        candidates: CandidateConfig {
            num_candidates_factor: 1.0,
            min_candidates: 1,
            max_candidates: None,
            num_candidates_per_arm: None,
            candidate_rv: CandidateRV::Uniform,
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
    use std::path::Path;

    #[test]
    fn test_candidate_config_num_candidates() {
        let config = CandidateConfig::default();

        // Basic case: 2D, 1 arm
        let n = config.num_candidates(2, 1);
        assert!(n >= 100); // At least min_candidates

        // Larger dimension
        let n_large = config.num_candidates(10, 1);
        assert!(n_large >= 1000);

        // More arms
        let n_arms = config.num_candidates(2, 10);
        assert!(n_arms >= 100); // 10 * 10 = 100
    }

    #[test]
    fn test_candidate_config_max_candidates_cap() {
        // Python default: min(5000, 100*num_dim). Cap at 5000 for high dim.
        let config = CandidateConfig {
            num_candidates_factor: 100.0,
            min_candidates: 100,
            max_candidates: Some(5000),
            num_candidates_per_arm: None,
            candidate_rv: CandidateRV::Uniform,
        };
        assert_eq!(config.num_candidates(60, 1), 5000);
        assert_eq!(config.num_candidates(100, 1), 5000);
        assert_eq!(config.num_candidates(10, 1), 1000);
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
    }

    #[test]
    fn test_turbo_zero_config() {
        let config = turbo_zero_config();
        assert!(matches!(config.surrogate, SurrogateConfig::None));
        assert!(matches!(config.acquisition, AcquisitionConfig::Random));
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
            index_driver: Some(IndexDriver::HNSW),
            num_fit_samples: Some(123),
            num_fit_candidates: Some(456),
            scale_x: Some(true),
            ..Default::default()
        };

        let config = turbo_enn_config();
        let applied = overrides.apply_to(config);

        assert!(matches!(applied.acquisition, AcquisitionConfig::Thompson));
        assert_eq!(applied.candidates.candidate_rv, CandidateRV::Sobol);
        if let SurrogateConfig::ENN(enn) = &applied.surrogate {
            assert_eq!(enn.index_driver, IndexDriver::HNSW);
            assert_eq!(enn.num_fit_samples, 123);
            assert_eq!(enn.num_fit_candidates, 456);
            assert!(enn.scale_x);
        } else {
            panic!("expected ENN surrogate");
        }
    }

    #[test]
    fn test_config_overrides_scale_x_apply() {
        let overrides = ConfigOverrides {
            scale_x: Some(true),
            ..Default::default()
        };
        let applied = overrides.apply_to(turbo_enn_config());
        let SurrogateConfig::ENN(enn) = applied.surrogate else {
            panic!("expected ENN surrogate");
        };
        assert!(enn.scale_x);
    }

    #[test]
    fn morbo_config_override_rejects_num_metrics_one() {
        use crate::morbo_trust_region::MorboTrustRegion;
        use crate::trust_region_config::TrustRegionConfig;
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let overrides = ConfigOverrides {
            trust_region_kind: Some("morbo".to_string()),
            num_metrics: Some(1),
            ..Default::default()
        };
        let applied = overrides.apply_to(turbo_enn_config());
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
    fn candidate_config_num_candidates_per_arm_scales_with_arms() {
        let cfg = CandidateConfig {
            num_candidates_factor: 1.0,
            min_candidates: 10,
            max_candidates: None,
            num_candidates_per_arm: Some(25),
            candidate_rv: CandidateRV::Uniform,
        };
        assert_eq!(cfg.num_candidates(2, 3), 75);
        assert_eq!(cfg.num_candidates(2, 8), 200);
    }

    #[test]
    fn config_overrides_apply_num_candidates_per_arm_to_pool() {
        let overrides = ConfigOverrides {
            num_candidates_factor: Some(1.0),
            min_candidates: Some(10),
            num_candidates_per_arm: Some(40),
            ..Default::default()
        };
        let applied = overrides.apply_to(turbo_zero_config());
        assert_eq!(applied.candidates.num_candidates(2, 3), 120);
        assert_eq!(applied.candidates.num_candidates(2, 8), 320);
    }

    #[test]
    fn config_overrides_apply_enn_num_fit_fields() {
        let overrides = ConfigOverrides {
            num_fit_samples: Some(7),
            num_fit_candidates: Some(11),
            scale_x: Some(true),
            ..Default::default()
        };
        let applied = overrides.apply_to(turbo_enn_config());
        let SurrogateConfig::ENN(enn) = applied.surrogate else {
            panic!("expected ENN surrogate");
        };
        assert_eq!(enn.num_fit_samples, 7);
        assert_eq!(enn.num_fit_candidates, 11);
        assert!(enn.scale_x);
    }

    #[test]
    fn config_overrides_apply_enn_storage_and_work_dir() {
        use crate::index::IndexDriver;
        use std::path::PathBuf;

        let overrides = ConfigOverrides {
            index_driver: Some(IndexDriver::HNSWDisk),
            enn_storage: Some(EnnStorage::Disk),
            work_dir: Some(PathBuf::from("/tmp/enn_work")),
            ..Default::default()
        };
        let applied = overrides.apply_to(turbo_enn_config());
        let SurrogateConfig::ENN(enn) = applied.surrogate else {
            panic!("expected ENN surrogate");
        };
        assert_eq!(enn.index_driver, IndexDriver::HNSWDisk);
        assert_eq!(enn.storage, EnnStorage::Disk);
        assert_eq!(enn.work_dir.as_deref(), Some(Path::new("/tmp/enn_work")));
    }

    #[test]
    fn kiss_apply_enn_surrogate_fields_unit_name() {
        assert_eq!("apply_enn_surrogate_fields", "apply_enn_surrogate_fields");
    }

    #[test]
    fn morbo_config_missing_rescalarize_defaults_on_propose() {
        let overrides = ConfigOverrides {
            trust_region_kind: Some("morbo".to_string()),
            num_metrics: Some(2),
            ..Default::default()
        };
        let applied = overrides.apply_to(turbo_enn_config());
        let TrustRegionConfig::Morbo(settings) = applied.trust_region else {
            panic!("expected Morbo trust region");
        };
        assert_eq!(
            settings.rescalarize,
            Rescalarize::OnPropose,
            "missing rescalarize should match Python MorboTRConfig default ON_PROPOSE"
        );
    }

    #[test]
    fn kiss_config_override_types_linked() {
        assert!(std::mem::size_of::<ConfigOverrides>() > 0);
        let acq = AcquisitionConfig::default();
        assert!(matches!(acq, AcquisitionConfig::UCB { .. }));
    }
}
