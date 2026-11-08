use crate::morbo_trust_region::Rescalarize;

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MorboOverride {
    pub num_metrics: usize,
    #[serde(with = "crate::exact")]
    pub alpha: f64,
    pub rescalarize: Rescalarize,
}
