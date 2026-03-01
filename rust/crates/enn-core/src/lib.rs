//! Core ENN algorithm implementations in Rust.
//!
//! This crate provides the algorithmic core of the Epistemic Nearest Neighbors
//! library, with implementations designed for parity with the Python reference.

pub mod acquisition;
pub mod draw;
pub mod error;
pub mod hash;
pub mod hypervolume;
pub mod index;
pub mod model;
pub mod params;
pub mod posterior;
pub mod traits;
pub mod stats;
pub mod trust_region;
pub mod util;

pub use acquisition::{
    AcquisitionError, ParetoAcquisition, RandomAcquisition, ThompsonAcquisition,
    UCBAcquisition,
};
pub use draw::{Candidates, ConditionalPosteriorDrawInternals, DrawInternals, NeighborData};
pub use error::{ENNError, EPS_VAR};
pub use hash::{normal_hash_batch_multi_seed, normal_hash_batch_multi_seed_fast};
pub use hypervolume::hypervolume_2d_max;
pub use index::{ENNIndex, IndexDriver, IndexError};
pub use model::EpistemicNearestNeighbors;
pub use params::{ENNNormal, ENNParams, ParamsError, PosteriorFlags};
pub use posterior::{compute_posterior_internals, WeightedPosteriorData};
pub use traits::PosteriorComputation;
pub use stats::WeightedStats;
pub use trust_region::{NoTrustRegion, TRLengthConfig, TrustRegionError, TurboTrustRegion};
pub use util::{calculate_sobol_indices, pareto_front_2d_maximize, standardize_y};
