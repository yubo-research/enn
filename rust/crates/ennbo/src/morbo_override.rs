//! MORBO fields that an optimizer override must supply together.

use crate::morbo_trust_region::Rescalarize;

/// Metric count, alpha, and rescalarize mode for one MORBO override.
///
/// These fields are one value. A caller cannot store a metric count without alpha,
/// or store an arbitrary rescalarize string.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MorboOverride {
    /// Number of objectives. MORBO rejects values below 2 when the trust region is built.
    pub num_metrics: usize,
    /// Chebyshev scalarization weight.
    pub alpha: f64,
    /// When scalarization weights are redrawn.
    pub rescalarize: Rescalarize,
}
