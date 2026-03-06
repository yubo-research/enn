//! Core ENN algorithm implementations in Rust.
//!
//! This crate provides the algorithmic core of the Epistemic Nearest Neighbors
//! library, with implementations designed for parity with the Python reference.

pub mod acquisition;
pub mod candidates;
pub mod config;
pub mod draw;
pub mod error;
pub mod fit;
pub mod hash;
pub mod hypervolume;
pub mod index;
pub mod model;
pub mod optimizer;
pub mod optimizer_factory;
pub mod params;
pub mod posterior;
pub mod surrogate;
pub mod strategy;
pub mod traits;
pub mod stats;
pub mod trust_region;
pub mod util;

pub use acquisition::{
    AcquisitionError, ParetoAcquisition, RandomAcquisition, ThompsonAcquisition,
    UCBAcquisition,
};
pub use candidates::{CandidateRV, generate_candidates, generate_lhd, to_unit, from_unit};
pub use config::{OptimizerConfig, CandidateConfig, AcquisitionConfig, SurrogateConfig,
    ConfigOverrides, InitStrategy, turbo_enn_config, turbo_zero_config, lhd_only_config};
pub use draw::{Candidates, ConditionalPosteriorDrawInternals, DrawInternals, NeighborData};
pub use error::{ENNError, EPS_VAR};
pub use fit::{enn_fit, subsample_loglik};
pub use hash::{normal_hash_batch_multi_seed, normal_hash_batch_multi_seed_fast};
pub use hypervolume::hypervolume_2d_max;
pub use index::{ENNIndex, IndexDriver, IndexError};
pub use model::EpistemicNearestNeighbors;
pub use optimizer::{Optimizer, Telemetry};
pub use optimizer_factory::{create_optimizer_enn, create_optimizer_lhd, create_optimizer_zero};
pub use params::{ENNNormal, ENNParams, ParamsError, PosteriorFlags};
pub use posterior::{compute_posterior_internals, WeightedPosteriorData};
pub use surrogate::{Surrogate, ENNSurrogate, ENNSurrogateConfig, SurrogatePrediction};
pub use strategy::Strategy;
pub use traits::PosteriorComputation;
pub use stats::WeightedStats;
pub use trust_region::{NoTrustRegion, TRLengthConfig, TrustRegionError, TurboTrustRegion};
pub use util::{calculate_sobol_indices, pareto_front_2d_maximize, standardize_y};
