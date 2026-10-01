//! AUTO metric policy: reservoir, refit schedule, rescale versus re-partition.

use crate::error::ENNError;
use crate::metric_loo::loo_loglik;
use crate::metric_sobol::MIN_DEPENDENCE_ROWS;
use crate::metric_weights::{dependence_weights, too_few_rows, validate_tied_dims};
use crate::reservoir::RowReservoir;

pub const DEFAULT_REBUILD_DRIFT: f64 = std::f64::consts::LN_2;
pub const DRIFT_WEIGHT_FLOOR: f64 = 9.210_340_371_976_184;
pub const AUTO_MIN_HELDOUT_GAIN: f64 = 0.0;
pub const AUTO_RESERVOIR_CAPACITY: usize = 1000;
pub const AUTO_K: usize = 10;
pub const AUTO_REFIT_GROWTH: f64 = 1.5;
pub const AUTO_RESCALE_TOL: f64 = 0.01;

/// `None` leaves the metric fixed. `Auto` learns a diagonal metric from a reservoir.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MetricLearning {
    #[default]
    None,
    Auto,
}

impl MetricLearning {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "none" | "NONE" | "None" => Some(Self::None),
            "auto" | "AUTO" | "Auto" => Some(Self::Auto),
            _ => None,
        }
    }
}

fn floored_log(w: &[f64]) -> Vec<f64> {
    let log_w: Vec<f64> = w.iter().map(|v| v.ln()).collect();
    let max = log_w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    log_w
        .into_iter()
        .map(|v| v.max(max - DRIFT_WEIGHT_FLOOR))
        .collect()
}

/// The leave-one-out Gram matrix is temporary. Return those pages so a later
/// resident-set reading does not keep them.
fn release_loo_pages() {
    #[cfg(target_os = "linux")]
    {
        extern "C" {
            fn malloc_trim(pad: usize) -> i32;
        }
        unsafe {
            malloc_trim(0);
        }
    }
}

pub fn weight_drift(weights: &[f64], built: &[f64]) -> f64 {
    let a = floored_log(weights);
    let b = floored_log(built);
    0.5 * a
        .iter()
        .zip(b.iter())
        .map(|(u, v)| (u - v).abs())
        .fold(0.0, f64::max)
}

pub fn auto_uses_learned_metric(heldout_gain: f64) -> bool {
    heldout_gain.is_finite() && heldout_gain > AUTO_MIN_HELDOUT_GAIN
}

pub fn auto_weights(
    x: &[f64],
    n: usize,
    d: usize,
    y: &[f64],
    m: usize,
    k: usize,
    tied: &[Vec<usize>],
) -> Result<(Vec<f64>, f64), ENNError> {
    validate_tied_dims(tied, d)?;
    if too_few_rows(n) {
        return Ok((vec![1.0; d], f64::NEG_INFINITY));
    }
    let w = dependence_weights(x, n, d, y, m, tied, crate::metric_weights::DEPENDENCE_FLOOR)?;
    let ones = vec![1.0; d];
    let gain = loo_loglik(x, n, d, y, m, &w, k) - loo_loglik(x, n, d, y, m, &ones, k);
    Ok((w, gain))
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MetricCounters {
    pub num_refits: usize,
    pub num_rescales: usize,
    pub num_rebuilds: usize,
}

#[derive(Clone, Debug)]
pub struct MetricSnapshot {
    pub num_seen: usize,
    pub counters: MetricCounters,
    pub heldout_gain: Option<f64>,
    pub uses_learned: bool,
    pub weights: Vec<f64>,
    pub built: Vec<f64>,
}

pub struct ScaleUpdate {
    pub x_scale: Vec<f64>,
    pub rebuild: bool,
}

pub struct AutoMetric {
    rows: RowReservoir,
    tied: Vec<Vec<usize>>,
    weights: Vec<f64>,
    built: Vec<f64>,
    pub rebuild_drift: f64,
    refit_growth: f64,
    next_refit: usize,
    pub heldout_gain: Option<f64>,
    counters: MetricCounters,
}

impl AutoMetric {
    pub fn new(
        num_dim: usize,
        num_outputs: usize,
        tied: Vec<Vec<usize>>,
        seed: u64,
    ) -> Result<Self, ENNError> {
        if num_dim == 0 || num_outputs == 0 {
            return Err(ENNError::InvalidParameter(
                "metric learning needs a positive shape".into(),
            ));
        }
        Ok(Self {
            rows: RowReservoir::new(AUTO_RESERVOIR_CAPACITY, num_dim, num_outputs, seed)?,
            tied,
            weights: vec![1.0; num_dim],
            built: vec![1.0; num_dim],
            rebuild_drift: DEFAULT_REBUILD_DRIFT,
            refit_growth: AUTO_REFIT_GROWTH,
            next_refit: MIN_DEPENDENCE_ROWS,
            heldout_gain: None,
            counters: MetricCounters::default(),
        })
    }

    pub fn tied(&self) -> &[Vec<usize>] {
        &self.tied
    }

    pub fn configure(
        &mut self,
        refit_growth: f64,
        rebuild_drift: f64,
        seed: u64,
        capacity: usize,
    ) -> Result<(), ENNError> {
        if refit_growth <= 1.0 {
            return Err(ENNError::InvalidParameter(format!(
                "refit_growth must be > 1, got {refit_growth}"
            )));
        }
        if rebuild_drift.is_nan() || rebuild_drift < 0.0 {
            return Err(ENNError::InvalidParameter(format!(
                "rebuild_drift must be >= 0, got {rebuild_drift}"
            )));
        }
        self.rows.configure(seed, capacity)?;
        self.refit_growth = refit_growth;
        self.rebuild_drift = rebuild_drift;
        Ok(())
    }

    pub fn seed(&self) -> u64 {
        self.rows.seed()
    }

    pub fn capacity(&self) -> usize {
        self.rows.capacity()
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    pub fn built(&self) -> &[f64] {
        &self.built
    }

    pub fn num_seen(&self) -> usize {
        self.rows.num_seen()
    }

    pub fn snapshot(&self) -> MetricSnapshot {
        MetricSnapshot {
            num_seen: self.num_seen(),
            counters: self.counters,
            heldout_gain: self.heldout_gain,
            uses_learned: self.uses_learned_metric(),
            weights: self.weights.clone(),
            built: self.built.clone(),
        }
    }

    pub fn uses_learned_metric(&self) -> bool {
        self.heldout_gain.is_some_and(auto_uses_learned_metric)
    }

    pub fn set_weights(&mut self, weights: &[f64]) -> Result<ScaleUpdate, ENNError> {
        if weights.len() != self.rows.num_dim()
            || !weights.iter().all(|w| w.is_finite() && *w > 0.0)
        {
            return Err(ENNError::InvalidParameter(
                "weights must be finite, > 0, and match the dimension".into(),
            ));
        }
        let rebuild = weight_drift(weights, &self.built) > self.rebuild_drift;
        let x_scale: Vec<f64> = weights.iter().map(|w| 1.0 / w.sqrt()).collect();
        self.weights = weights.to_vec();
        if rebuild {
            self.built = weights.to_vec();
            self.counters.num_rebuilds += 1;
        } else {
            self.counters.num_rescales += 1;
        }
        Ok(ScaleUpdate { x_scale, rebuild })
    }

    pub fn refit(&mut self) -> Result<Option<ScaleUpdate>, ENNError> {
        let n = self.rows.len();
        let (w, gain) = auto_weights(
            self.rows.x(),
            n,
            self.rows.num_dim(),
            self.rows.y(),
            self.rows.num_outputs(),
            AUTO_K,
            &self.tied,
        )?;
        self.heldout_gain = Some(gain);
        self.counters.num_refits += 1;
        let grown = (self.refit_growth * self.num_seen() as f64).ceil() as usize;
        self.next_refit = grown.max(MIN_DEPENDENCE_ROWS);
        let target = if auto_uses_learned_metric(gain) {
            w
        } else {
            vec![1.0; self.rows.num_dim()]
        };
        let change = 0.5
            * target
                .iter()
                .zip(self.weights.iter())
                .map(|(t, w)| (t / w).ln().abs())
                .fold(0.0, f64::max);
        if n >= 512 {
            release_loo_pages();
        }
        if change > AUTO_RESCALE_TOL {
            return Ok(Some(self.set_weights(&target)?));
        }
        Ok(None)
    }

    pub fn observe(
        &mut self,
        x: &[f64],
        y: &[f64],
        n: usize,
    ) -> Result<Option<ScaleUpdate>, ENNError> {
        for i in 0..n {
            let xr = &x[i * self.rows.num_dim()..(i + 1) * self.rows.num_dim()];
            let yr = &y[i * self.rows.num_outputs()..(i + 1) * self.rows.num_outputs()];
            self.rows.push_row(xr, yr);
        }
        if self.num_seen() >= self.next_refit {
            return self.refit();
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configure_updates_growth_seed_and_capacity_before_replacement() {
        let mut metric = AutoMetric::new(2, 1, vec![], 0).unwrap();
        metric.configure(3.0, 0.25, 7, 4).unwrap();
        assert_eq!(metric.refit_growth, 3.0);
        assert_eq!(metric.rebuild_drift, 0.25);
        assert_eq!(metric.seed(), 7);
        assert_eq!(metric.capacity(), 4);
        assert!(metric.configure(3.0, 0.25, 7, 0).is_err());
        metric
            .observe(&[0.0, 1.0, 0.2, 0.3], &[0.0, 1.0], 2)
            .unwrap();
        assert!(metric.configure(3.0, 0.25, 7, 1).is_err());
    }

    #[test]
    fn auto_weights_k_changes_heldout_gain_only() {
        let n = 120;
        let d = 3;
        let mut x = vec![0.0; n * d];
        let mut y = vec![0.0; n];
        for i in 0..n {
            for j in 0..d {
                x[i * d + j] = ((i + 1) as f64 * (j + 3) as f64 * 0.017 + 0.001 * i as f64) % 1.0;
            }
            y[i] = (6.0 * std::f64::consts::PI * x[i * d]).sin() + 0.01 * x[i * d + 1];
        }
        let (w10, g10) = auto_weights(&x, n, d, &y, 1, 10, &[]).unwrap();
        let (w1, g1) = auto_weights(&x, n, d, &y, 1, 1, &[]).unwrap();
        assert_eq!(w10, w1);
        assert_ne!(g10, g1);
    }
}
