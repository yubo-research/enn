//! Configuration types for the optimizer.

use crate::candidates::CandidateRV;
use crate::index::IndexDriver;
use crate::surrogate::ENNSurrogateConfig;
use crate::trust_region::TRLengthConfig;

/// Optimizer configuration.
#[derive(Debug, Clone)]
pub struct OptimizerConfig {
    /// Surrogate configuration.
    pub surrogate: SurrogateConfig,
    /// Trust region configuration.
    pub trust_region: TRLengthConfig,
    /// Candidate generation configuration.
    pub candidates: CandidateConfig,
    /// Acquisition function configuration.
    pub acquisition: AcquisitionConfig,
    /// Trailing observations limit (None = keep all).
    pub trailing_obs: Option<usize>,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            surrogate: SurrogateConfig::ENN(ENNSurrogateConfig::default()),
            trust_region: TRLengthConfig::default(),
            candidates: CandidateConfig::default(),
            acquisition: AcquisitionConfig::default(),
            trailing_obs: None,
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
    /// Random variable type for candidates.
    pub candidate_rv: CandidateRV,
}

impl Default for CandidateConfig {
    fn default() -> Self {
        Self {
            num_candidates_factor: 1000.0,
            min_candidates: 100,
            max_candidates: None,
            candidate_rv: CandidateRV::Uniform,
        }
    }
}

impl CandidateConfig {
    /// Compute number of candidates based on dimension and arms.
    pub fn num_candidates(&self, num_dim: usize, num_arms: usize) -> usize {
        let base = (self.num_candidates_factor * num_dim as f64) as usize;
        let adjusted = base.max(self.min_candidates);
        let with_arms = adjusted.max(num_arms * 10); // At least 10x the number of arms
        match self.max_candidates {
            Some(cap) => with_arms.min(cap),
            None => with_arms,
        }
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
    pub length_init: Option<f64>,
    pub length_min: Option<f64>,
    pub length_max: Option<f64>,
    pub index_driver: Option<IndexDriver>,
    pub trailing_obs: Option<usize>,
    pub num_fit_samples: Option<usize>,
    pub num_fit_candidates: Option<usize>,
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
        if self.length_init.is_some() || self.length_min.is_some() || self.length_max.is_some() {
            config.trust_region = TRLengthConfig {
                length_init: self.length_init.unwrap_or(config.trust_region.length_init),
                length_min: self.length_min.unwrap_or(config.trust_region.length_min),
                length_max: self.length_max.unwrap_or(config.trust_region.length_max),
            };
        }
        if let Some(driver) = self.index_driver {
            if let SurrogateConfig::ENN(enn_cfg) = &config.surrogate {
                let mut enn = enn_cfg.clone();
                enn.index_driver = driver;
                config.surrogate = SurrogateConfig::ENN(enn);
            }
        }
        if let Some(t) = self.trailing_obs {
            config.trailing_obs = Some(t);
        }
        if let Some(nfs) = self.num_fit_samples {
            if let SurrogateConfig::ENN(enn_cfg) = &config.surrogate {
                let mut enn = enn_cfg.clone();
                enn.num_fit_samples = nfs;
                config.surrogate = SurrogateConfig::ENN(enn);
            }
        }
        if let Some(nfc) = self.num_fit_candidates {
            if let SurrogateConfig::ENN(enn_cfg) = &config.surrogate {
                let mut enn = enn_cfg.clone();
                enn.num_fit_candidates = nfc;
                config.surrogate = SurrogateConfig::ENN(enn);
            }
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
        trust_region: TRLengthConfig::default(),
        candidates: CandidateConfig {
            num_candidates_factor: 1000.0,
            min_candidates: 100,
            max_candidates: None,
            candidate_rv: CandidateRV::Uniform,
        },
        acquisition: AcquisitionConfig::UCB { beta: 2.0 },
        trailing_obs: None,
    }
}

/// Create a TuRBO-ZERO configuration.
pub fn turbo_zero_config() -> OptimizerConfig {
    OptimizerConfig {
        surrogate: SurrogateConfig::None,
        trust_region: TRLengthConfig::default(),
        candidates: CandidateConfig {
            num_candidates_factor: 1000.0,
            min_candidates: 100,
            max_candidates: None,
            candidate_rv: CandidateRV::Uniform,
        },
        acquisition: AcquisitionConfig::Random,
        trailing_obs: None,
    }
}

/// Create an LHD-only configuration.
pub fn lhd_only_config() -> OptimizerConfig {
    OptimizerConfig {
        surrogate: SurrogateConfig::None,
        trust_region: TRLengthConfig::default(), // Minimal TR
        candidates: CandidateConfig {
            num_candidates_factor: 1.0,
            min_candidates: 1,
            max_candidates: None,
            candidate_rv: CandidateRV::Uniform,
        },
        acquisition: AcquisitionConfig::Random,
        trailing_obs: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::candidates::CandidateRV;

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
        // With factor=1.0, min=1, but num_arms*10=10 minimum applies
        assert_eq!(n, 10); // 10 * 1 arms = 10
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
            trailing_obs: Some(20),
            index_driver: Some(IndexDriver::HNSW),
            num_fit_samples: Some(123),
            num_fit_candidates: Some(456),
            ..Default::default()
        };

        let config = turbo_enn_config();
        let applied = overrides.apply_to(config);

        assert!(matches!(applied.acquisition, AcquisitionConfig::Thompson));
        assert_eq!(applied.candidates.candidate_rv, CandidateRV::Sobol);
        assert_eq!(applied.trailing_obs, Some(20));
        if let SurrogateConfig::ENN(enn) = &applied.surrogate {
            assert_eq!(enn.index_driver, IndexDriver::HNSW);
            assert_eq!(enn.num_fit_samples, 123);
            assert_eq!(enn.num_fit_candidates, 456);
        } else {
            panic!("expected ENN surrogate");
        }
    }
}
