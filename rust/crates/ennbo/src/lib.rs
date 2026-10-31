
#![allow(clippy::pedantic, clippy::nursery, clippy::cargo)]
#![allow(missing_docs)]

#[allow(missing_docs)]
pub mod ennbo_build {
    include!("ennbo_build_api.inc.rs");
    use super::link_search;
    define_ennbo_build_api!(link_search);
}
#[allow(missing_docs)]
pub mod link_search;
#[allow(missing_docs)]
pub mod acquisition;
#[allow(missing_docs)]
pub mod candidates;
#[allow(missing_docs)]
pub mod config;
#[allow(missing_docs)]
pub mod draw;
#[allow(missing_docs)]
pub mod error;
#[allow(missing_docs)]
pub mod file_config;
#[allow(missing_docs)]
pub mod fit;
#[allow(missing_docs)]
pub mod fitter;
#[allow(missing_docs)]
pub mod fit_samples;
#[allow(missing_docs)]
pub mod neighbor_count;
#[allow(missing_docs)]
pub mod enn_overrides;
#[allow(missing_docs)]
pub mod benchmarks;
#[allow(missing_docs)]
pub mod calibration;
#[allow(missing_docs)]
pub mod candidates_fast;
#[allow(missing_docs)]
pub mod hash;
#[allow(missing_docs)]
pub mod metric_auto;
#[allow(missing_docs)]
pub mod metric_loo;
#[allow(missing_docs)]
pub mod metric_sobol;
#[allow(missing_docs)]
pub mod metric_weights;
#[allow(missing_docs)]
pub mod ndtri;
#[allow(missing_docs)]
pub mod numpy_pcg;
#[allow(missing_docs)]
pub mod numpy_ziggurat_tables;
#[allow(missing_docs)]
pub mod numpy_seed;
#[allow(missing_docs)]
pub mod normal_sample;
#[allow(missing_docs)]
pub mod philox;
#[allow(missing_docs)]
pub mod philox_hash;
#[cfg(test)]
mod golden_parity;
#[allow(missing_docs)]
pub mod hypervolume;
#[allow(missing_docs)]
pub mod incremental_fit;
#[allow(missing_docs)]
pub mod incumbent_tracker;
#[allow(missing_docs)]
pub mod index;
#[allow(missing_docs)]
pub mod knn;
#[allow(missing_docs)]
pub mod backend;
#[allow(missing_docs)]
pub mod layout;
#[allow(missing_docs)]
pub mod disk_bpann;
#[allow(missing_docs)]
pub mod model;
#[allow(missing_docs)]
pub mod morbo_override;
#[allow(missing_docs)]
pub mod morbo_trust_region;
#[allow(missing_docs)]
pub mod optimizer;
#[allow(missing_docs)]
pub mod optimizer_factory;
#[allow(missing_docs)]
pub mod params;
#[allow(missing_docs)]
pub mod posterior;
#[allow(missing_docs)]
pub mod reservoir;
#[allow(missing_docs)]
pub mod stats;
#[allow(missing_docs)]
pub mod strategy;
#[allow(missing_docs)]
pub mod surrogate;
#[allow(missing_docs)]
pub mod semantic;
#[allow(missing_docs)]
pub mod surrogate_affine;
pub(crate) mod surrogate_state;
#[allow(missing_docs)]
pub mod traits;
#[allow(missing_docs)]
pub mod trust_region;
#[allow(missing_docs)]
pub mod trust_region_config;
#[allow(missing_docs)]
pub mod util;
#[allow(missing_docs)]
pub mod y_bounds;

#[cfg(test)]
pub(crate) mod test_helpers;

pub use acquisition::AcquisitionError;
pub use acquisition::ParetoAcquisition;
pub use acquisition::RandomAcquisition;
pub use acquisition::ThompsonAcquisition;
pub use acquisition::UCBAcquisition;
pub use candidates::from_unit;
pub use candidates::generate_candidates;
pub use candidates::generate_lhd;
pub use candidates::sobol_sequence;
pub use candidates::to_unit;
pub use candidates::CandidateRV;
pub use config::lhd_only_config;
pub use config::require_fit_samples;
pub use enn_overrides::EnnOverrides;
pub use fit_samples::FitSamples;
pub use fit_samples::ScaleSearch;
pub use neighbor_count::NeighborCount;
pub use config::turbo_enn_config;
pub use config::turbo_zero_config;
pub use config::validate_optimizer_rules;
pub use config::AcquisitionConfig;
pub use config::DEFAULT_UCB_BETA;
pub use config::OptimizerRuleSet;
pub use config::CandidateConfig;
pub use config::ConfigOverrides;
pub use config::InitStrategy;
pub use config::OptimizerConfig;
pub use config::OptimizerInitKind;
pub use config::SurrogateConfig;
pub use config::TurboEnnBuilder;
pub use file_config::default_config_path;
pub use file_config::ensure_config_file;
pub use file_config::install_bpann_tuning_from_config;
pub use file_config::set_config_path;
pub use file_config::BpannConfig;
pub use file_config::Config;
pub use file_config::ConfigFile;
pub use draw::Candidates;
pub use draw::ConditionalPosteriorDrawInternals;
pub use draw::DrawInternals;
pub use draw::NeighborData;
pub use error::ENNError;
pub use error::EPS_VAR;
pub use benchmarks::Ackley;
pub use benchmarks::DoubleAckley;
pub use calibration::AffineCalibrator;
pub use fitter::ENNFitter;
pub use incremental_fit::{AddToken, IncrementalAsk, IncrementalFit};
pub use hash::normal_hash_batch_multi_seed;
pub use hash::normal_hash_batch_multi_seed_fast;
pub use hypervolume::hypervolume_2d_max;
pub use incumbent_tracker::IncrementalIncumbentTracker;
pub use index::ENNIndex;
pub use index::IndexDriver;
pub use index::IndexError;
pub use fit::subsample_loglik;
pub use fit::subsample_loglik_model;
pub use model::EpistemicNearestNeighbors;
pub use model::EnnIndexAccess;
pub use model::EnnRowAccess;
pub use backend::EnnBackend;
pub use backend::EnnStorage;
pub use layout::EnnLayout;
pub use backend::InMemoryEnnBackend;
pub use backend::DiskBpannEnnBackend;
pub use optimizer::obs_access::ObsAccess;
pub use optimizer::ObservationDelta;
pub use optimizer::Optimizer;
pub use optimizer::Telemetry;
pub use optimizer_factory::create_optimizer_enn;
pub use optimizer_factory::create_optimizer_enn_with_overrides;
pub use optimizer_factory::create_optimizer_lhd;
pub use optimizer_factory::create_optimizer_zero;
pub use params::ENNNormal;
pub use params::ENNParams;
pub use params::ParamsError;
pub use params::PosteriorFlags;
pub use posterior::compute_conditional_posterior_internals;
pub use posterior::compute_posterior_internals;
pub use posterior::WeightedPosteriorData;
pub use stats::WeightedStats;
pub use strategy::Strategy;
pub use surrogate::ENNSurrogate;
pub use surrogate::ENNSurrogateConfig;
pub use surrogate::Surrogate;
pub use surrogate::SurrogatePrediction;
pub use traits::PosteriorComputation;
pub use morbo_trust_region::MorboTRSettings;
pub use morbo_trust_region::MorboTrustRegion;
pub use morbo_trust_region::Rescalarize;
pub use trust_region::NoTrustRegion;
pub use trust_region::TRLengthConfig;
pub use trust_region::TrustRegionError;
pub use trust_region::TurboTrustRegion;
pub use trust_region_config::TrustRegionConfig;
pub use trust_region_config::TrustRegionKind;
pub use util::argmax_random_tie;
pub use util::calculate_sobol_indices;
pub use util::pareto_front_2d_maximize;
pub use util::standardize_y;
pub use y_bounds::bounds_to_json;
pub use y_bounds::is_identity_bounds;
pub use y_bounds::unbounded_bounds;
pub use y_bounds::validate_bounds;

pub mod semantic_projection;
pub mod semantic_relationships;

pub mod categorical;

pub mod categorical_plan;
pub mod categorical_encoder;
pub mod input_history;

mod input_history_disk;

mod categorical_values;

pub mod semantic_text;
pub mod semantic_encoder;

pub mod adaptive;
pub mod adaptive_basis;
mod adaptive_geometry;
mod adaptive_scales;
mod adaptive_predictor;
mod adaptive_selection;
#[cfg(test)]
mod adaptive_tests;
