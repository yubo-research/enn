use crate::morbo_trust_region::Rescalarize;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MorboOverride {
    pub num_metrics: usize,
    pub alpha: f64,
    pub rescalarize: Rescalarize,
}
