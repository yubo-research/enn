#![doc = include_str!("../README.md")]
//! Core ENN algorithm implementations in Rust.
//!
//! This crate provides the algorithmic core of the Epistemic Nearest Neighbors
//! library, with implementations designed for parity with the Python reference.

#![allow(clippy::pedantic, clippy::nursery, clippy::cargo)]
#![warn(missing_docs)]

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
pub mod surrogate_affine;
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

/// Error from an acquisition function.
pub use acquisition::AcquisitionError;
/// Pareto (non-dominated) acquisition.
pub use acquisition::ParetoAcquisition;
/// Uniform random acquisition.
pub use acquisition::RandomAcquisition;
/// Thompson sampling acquisition.
pub use acquisition::ThompsonAcquisition;
/// Upper confidence bound acquisition.
pub use acquisition::UCBAcquisition;
/// Map a point from the unit cube into natural bounds.
pub use candidates::from_unit;
/// Draw candidate points inside a trust region.
pub use candidates::generate_candidates;
/// Draw a Latin-hypercube design.
pub use candidates::generate_lhd;
/// Scrambled Sobol points in the unit cube.
pub use candidates::sobol_sequence;
/// Map a natural-unit point into the unit cube.
pub use candidates::to_unit;
/// Distribution used to draw candidates.
pub use candidates::CandidateRV;
/// Default Latin-hypercube-only optimizer config.
pub use config::lhd_only_config;
/// Reject a missing `num_fit_samples` on non-Pareto acquisition.
pub use config::require_num_fit_samples;
/// Default TuRBO-ENN optimizer config.
pub use config::turbo_enn_config;
/// Default TuRBO-ZERO optimizer config.
pub use config::turbo_zero_config;
/// Dataclass-shaped optimizer checks used by the Python facade.
pub use config::validate_optimizer_rules;
/// Which acquisition the optimizer runs.
pub use config::AcquisitionConfig;
/// Flags for [`validate_optimizer_rules`].
pub use config::OptimizerRuleSet;
/// How many candidates to draw.
pub use config::CandidateConfig;
/// Binding-level overrides applied on top of a factory config.
pub use config::ConfigOverrides;
/// How the trust region is initialized.
pub use config::InitStrategy;
/// Full optimizer configuration.
pub use config::OptimizerConfig;
/// Hybrid TuRBO init versus Latin-hypercube only.
pub use config::OptimizerInitKind;
/// Surrogate attached to an optimizer, if any.
pub use config::SurrogateConfig;
/// Fluent builder for a TuRBO-ENN config.
pub use config::TurboEnnBuilder;
/// Default path of the BPANN config file.
pub use file_config::default_config_path;
/// Create the BPANN config file when it is missing.
pub use file_config::ensure_config_file;
/// Load BPANN tuning from the config file into the process.
pub use file_config::install_bpann_tuning_from_config;
/// Point subsequent config loads at `path`.
pub use file_config::set_config_path;
/// BPANN block of the config file.
pub use file_config::BpannConfig;
/// Parsed config file.
pub use file_config::Config;
/// On-disk config file handle.
pub use file_config::ConfigFile;
/// Candidate batch held by a draw.
pub use draw::Candidates;
/// Internals of a conditional posterior draw.
pub use draw::ConditionalPosteriorDrawInternals;
/// Internals of one posterior draw.
pub use draw::DrawInternals;
/// Neighbor indices and distances for one query.
pub use draw::NeighborData;
/// Errors returned by `ennbo`.
pub use error::ENNError;
/// Floor used when a variance would otherwise be zero.
pub use error::EPS_VAR;
/// Noisy Ackley objective. Noise uses one `StdRng` owned by the value.
pub use benchmarks::Ackley;
/// Two Ackley objectives on an even-dimensional input.
pub use benchmarks::DoubleAckley;
/// Post-hoc affine map of a posterior.
pub use calibration::AffineCalibrator;
/// Stateful fitter for ENN variance scales.
pub use fitter::ENNFitter;
/// Hash a batch of normal draws for several seeds.
pub use hash::normal_hash_batch_multi_seed;
/// Faster path of [`normal_hash_batch_multi_seed`].
pub use hash::normal_hash_batch_multi_seed_fast;
/// 2-D hypervolume for maximization.
pub use hypervolume::hypervolume_2d_max;
/// Incumbent tracker that updates as observations arrive.
pub use incumbent_tracker::IncrementalIncumbentTracker;
/// Neighbor index owned by a model.
pub use index::ENNIndex;
/// Flat memory scan or on-disk B+ANN.
pub use index::IndexDriver;
/// Error from a neighbor index.
pub use index::IndexError;
/// Subsample log-likelihood used to fit ENN scales.
pub use fit::subsample_loglik;
/// Same fit scored against a model that is already built.
pub use fit::subsample_loglik_model;
/// Epistemic nearest-neighbor model.
pub use model::EpistemicNearestNeighbors;
/// Index sync and search helpers for a model.
pub use model::EnnIndexAccess;
/// Row reads for a model.
pub use model::EnnRowAccess;
/// Storage backend for training rows.
pub use backend::EnnBackend;
/// In-memory or on-disk row storage.
pub use backend::EnnStorage;
/// Legal index, storage, and metric combination.
pub use layout::EnnLayout;
/// Training rows kept in process memory.
pub use backend::InMemoryEnnBackend;
/// Training rows and a B+ANN index stored on disk.
pub use backend::DiskBpannEnnBackend;
/// Read observations already stored on an optimizer.
pub use optimizer::obs_access::ObsAccess;
/// Rows appended by one `tell`.
pub use optimizer::ObservationDelta;
/// TuRBO optimizer. Ask and tell use natural units.
pub use optimizer::Optimizer;
/// Timings from the last ask or tell.
pub use optimizer::Telemetry;
/// Build a TuRBO-ENN optimizer from bounds and one seed.
pub use optimizer_factory::create_optimizer_enn;
/// [`create_optimizer_enn`] plus [`ConfigOverrides`].
pub use optimizer_factory::create_optimizer_enn_with_overrides;
/// Build a Latin-hypercube-only optimizer.
pub use optimizer_factory::create_optimizer_lhd;
/// Build a TuRBO-ZERO optimizer.
pub use optimizer_factory::create_optimizer_zero;
/// Posterior mean and standard errors, with sampling.
pub use params::ENNNormal;
/// Neighbor count and variance scales for a posterior.
pub use params::ENNParams;
/// Error from an invalid [`ENNParams`] value.
pub use params::ParamsError;
/// Flags for posterior queries.
pub use params::PosteriorFlags;
/// Conditional posterior internals before they are packed into [`ENNNormal`].
pub use posterior::compute_conditional_posterior_internals;
/// Posterior internals before they are packed into [`ENNNormal`].
pub use posterior::compute_posterior_internals;
/// Weighted neighbor data inside a posterior.
pub use posterior::WeightedPosteriorData;
/// Weighted mean and variance of neighbor targets.
pub use stats::WeightedStats;
/// Ask and tell policy owned by an [`Optimizer`].
pub use strategy::Strategy;
/// ENN surrogate used inside TuRBO-ENN.
pub use surrogate::ENNSurrogate;
/// Settings for [`ENNSurrogate`].
pub use surrogate::ENNSurrogateConfig;
/// Surrogate trait used by the optimizer.
pub use surrogate::Surrogate;
/// Mean and standard error returned by a surrogate.
pub use surrogate::SurrogatePrediction;
/// Trait for posterior computation on a model.
pub use traits::PosteriorComputation;
/// MORBO trust-region settings.
pub use morbo_trust_region::MorboTRSettings;
/// MORBO trust region.
pub use morbo_trust_region::MorboTrustRegion;
/// When MORBO rescales its objectives.
pub use morbo_trust_region::Rescalarize;
/// Trust region that does not move.
pub use trust_region::NoTrustRegion;
/// Length schedule for a TuRBO trust region.
pub use trust_region::TRLengthConfig;
/// Error from a trust-region update.
pub use trust_region::TrustRegionError;
/// TuRBO trust region.
pub use trust_region::TurboTrustRegion;
/// TuRBO or MORBO trust-region choice.
pub use trust_region_config::TrustRegionConfig;
/// Wire enum for a trust-region override.
pub use trust_region_config::TrustRegionKind;
/// Argmax that breaks ties at random.
pub use util::argmax_random_tie;
/// Sobol indices of the columns of `x` against `y`.
pub use util::calculate_sobol_indices;
/// 2-D Pareto front for maximization.
pub use util::pareto_front_2d_maximize;
/// Column-wise standardization of a target matrix.
pub use util::standardize_y;
/// JSON form of a y-bounds matrix.
pub use y_bounds::bounds_to_json;
/// Whether every y interval is the whole real line.
pub use y_bounds::is_identity_bounds;
/// One open interval `(−∞, +∞)` per metric.
pub use y_bounds::unbounded_bounds;
/// Check that y bounds have shape `(num_metrics, 2)`.
pub use y_bounds::validate_bounds;
