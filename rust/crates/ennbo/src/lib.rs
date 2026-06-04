//! Core ENN algorithm implementations in Rust.
//!
//! This crate provides the algorithmic core of the Epistemic Nearest Neighbors
//! library, with implementations designed for parity with the Python reference.

#![allow(clippy::pedantic, clippy::nursery, clippy::cargo)]

pub mod ennbo_build {
    include!("ennbo_build_api.inc.rs");
    use super::link_search;
    define_ennbo_build_api!(link_search);
}
pub mod link_search;
pub mod acquisition;
pub mod candidates;
pub mod config;
pub mod draw;
pub mod error;
pub mod fit;
pub mod fitter;
pub mod hash;
pub mod hypervolume;
pub mod incumbent_tracker;
pub mod index;
pub mod knn;
pub mod backend;
pub mod disk_hnsw;
pub mod model;
pub mod morbo_trust_region;
pub mod optimizer;
pub mod optimizer_factory;
pub mod params;
pub mod posterior;
pub mod stats;
pub mod strategy;
pub mod surrogate;
pub mod traits;
pub mod trust_region;
pub mod trust_region_config;
pub mod util;

#[cfg(test)]
pub(crate) mod test_helpers;

pub use acquisition::{
    AcquisitionError, ParetoAcquisition, RandomAcquisition, ThompsonAcquisition, UCBAcquisition,
};
pub use candidates::{from_unit, generate_candidates, generate_lhd, to_unit, CandidateRV};
pub use config::{
    lhd_only_config, turbo_enn_config, turbo_zero_config, AcquisitionConfig, CandidateConfig,
    ConfigOverrides, InitStrategy, OptimizerConfig, SurrogateConfig,
};
pub use draw::{Candidates, ConditionalPosteriorDrawInternals, DrawInternals, NeighborData};
pub use error::{ENNError, EPS_VAR};
pub use fitter::ENNFitter;
pub use hash::{normal_hash_batch_multi_seed, normal_hash_batch_multi_seed_fast};
pub use hypervolume::hypervolume_2d_max;
pub use incumbent_tracker::IncrementalIncumbentTracker;
pub use index::{ENNIndex, IndexDriver, IndexError};
pub use fit::{subsample_loglik, subsample_loglik_model};
pub use model::EpistemicNearestNeighbors;
pub use model::{EnnIndexAccess, EnnRowAccess};
pub use backend::{EnnBackend, EnnStorage, InMemoryEnnBackend};
pub use backend::DiskHnswEnnBackend;
pub use optimizer::obs_access::ObsAccess;
pub use optimizer::{ObservationDelta, Optimizer, Telemetry};
pub use optimizer_factory::{create_optimizer_enn, create_optimizer_lhd, create_optimizer_zero};
pub use params::{ENNNormal, ENNParams, ParamsError, PosteriorFlags};
pub use posterior::{
    compute_conditional_posterior_internals, compute_posterior_internals, WeightedPosteriorData,
};
pub use stats::WeightedStats;
pub use strategy::Strategy;
pub use surrogate::{ENNSurrogate, ENNSurrogateConfig, Surrogate, SurrogatePrediction};
pub use traits::PosteriorComputation;
pub use morbo_trust_region::{MorboTRSettings, MorboTrustRegion, Rescalarize};
pub use trust_region::{NoTrustRegion, TRLengthConfig, TrustRegionError, TurboTrustRegion};
pub use trust_region_config::TrustRegionConfig;
pub use util::{argmax_random_tie, calculate_sobol_indices, pareto_front_2d_maximize, standardize_y};
