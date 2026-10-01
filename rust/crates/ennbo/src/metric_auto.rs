//! AUTO metric policy: reservoir, refit schedule, rescale versus re-partition.

use crate::error::ENNError;
use crate::metric_loo::loo_loglik;
use crate::metric_sobol::MIN_DEPENDENCE_ROWS;
use crate::metric_weights::{dependence_weights, too_few_rows};
use crate::numpy_pcg::NumpyPcg64;

pub const DEFAULT_REBUILD_DRIFT: f64 = std::f64::consts::LN_2;
pub const DRIFT_WEIGHT_FLOOR: f64 = 9.210_340_371_976_184;
pub const AUTO_MIN_HELDOUT_GAIN: f64 = 0.0;
pub const AUTO_RESERVOIR_CAPACITY: usize = 1000;
pub const AUTO_K: usize = 10;
pub const AUTO_REFIT_GROWTH: f64 = 1.5;
pub const AUTO_RESCALE_TOL: f64 = 0.01;

fn floored_log(w: &[f64]) -> Vec<f64> {
    let log_w: Vec<f64> = w.iter().map(|v| v.ln()).collect();
    let max = log_w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    log_w.into_iter().map(|v| v.max(max - DRIFT_WEIGHT_FLOOR)).collect()
}

pub fn weight_drift(weights: &[f64], built: &[f64]) -> f64 {
    let a = floored_log(weights);
    let b = floored_log(built);
    0.5 * a.iter().zip(b.iter()).map(|(u, v)| (u - v).abs()).fold(0.0, f64::max)
}

pub fn auto_uses_learned_metric(heldout_gain: f64) -> bool {
    heldout_gain.is_finite() && heldout_gain > AUTO_MIN_HELDOUT_GAIN
}

pub fn auto_weights(x: &[f64], n: usize, d: usize, y: &[f64], m: usize, tied: &[Vec<usize>]) -> (Vec<f64>, f64) {
    if too_few_rows(n) {
        return (vec![1.0; d], f64::NEG_INFINITY);
    }
    let w = dependence_weights(x, n, d, y, m, tied);
    let ones = vec![1.0; d];
    let gain = loo_loglik(x, n, d, y, m, &w, AUTO_K) - loo_loglik(x, n, d, y, m, &ones, AUTO_K);
    (w, gain)
}

pub struct ScaleUpdate {
    pub x_scale: Vec<f64>,
    pub rebuild: bool,
}

pub struct AutoMetric {
    num_dim: usize,
    num_outputs: usize,
    tied: Vec<Vec<usize>>,
    capacity: usize,
    rng: NumpyPcg64,
    xs: Vec<f64>,
    ys: Vec<f64>,
    num_seen: usize,
    len: usize,
    weights: Vec<f64>,
    built: Vec<f64>,
    pub rebuild_drift: f64,
    refit_growth: f64,
    next_refit: usize,
    pub heldout_gain: Option<f64>,
    pub num_refits: usize,
    pub num_rescales: usize,
    pub num_rebuilds: usize,
}

impl AutoMetric {
    pub fn new(num_dim: usize, num_outputs: usize, tied: Vec<Vec<usize>>, seed: u64) -> Result<Self, ENNError> {
        if num_dim == 0 || num_outputs == 0 {
            return Err(ENNError::InvalidParameter("metric learning needs a positive shape".into()));
        }
        Ok(Self {
            num_dim,
            num_outputs,
            tied,
            capacity: AUTO_RESERVOIR_CAPACITY,
            rng: NumpyPcg64::from_seed(seed),
            xs: vec![0.0; AUTO_RESERVOIR_CAPACITY * num_dim],
            ys: vec![0.0; AUTO_RESERVOIR_CAPACITY * num_outputs],
            num_seen: 0,
            len: 0,
            weights: vec![1.0; num_dim],
            built: vec![1.0; num_dim],
            rebuild_drift: DEFAULT_REBUILD_DRIFT,
            refit_growth: AUTO_REFIT_GROWTH,
            next_refit: MIN_DEPENDENCE_ROWS,
            heldout_gain: None,
            num_refits: 0,
            num_rescales: 0,
            num_rebuilds: 0,
        })
    }

    pub fn weights(&self) -> &[f64] {
        &self.weights
    }

    pub fn built(&self) -> &[f64] {
        &self.built
    }

    pub fn num_seen(&self) -> usize {
        self.num_seen
    }

    pub fn uses_learned_metric(&self) -> bool {
        self.heldout_gain.is_some_and(auto_uses_learned_metric)
    }

    fn push_row(&mut self, x: &[f64], y: &[f64]) {
        if self.len < self.capacity {
            let i = self.len;
            self.xs[i * self.num_dim..(i + 1) * self.num_dim].copy_from_slice(x);
            self.ys[i * self.num_outputs..(i + 1) * self.num_outputs].copy_from_slice(y);
            self.len += 1;
        } else {
            let slot = self.rng.integers_high(self.num_seen as u64 + 1) as usize;
            if slot < self.capacity {
                self.xs[slot * self.num_dim..(slot + 1) * self.num_dim].copy_from_slice(x);
                self.ys[slot * self.num_outputs..(slot + 1) * self.num_outputs].copy_from_slice(y);
            }
        }
        self.num_seen += 1;
    }

    pub fn set_weights(&mut self, weights: &[f64]) -> Result<ScaleUpdate, ENNError> {
        if weights.len() != self.num_dim || !weights.iter().all(|w| w.is_finite() && *w > 0.0) {
            return Err(ENNError::InvalidParameter(
                "weights must be finite, > 0, and match the dimension".into(),
            ));
        }
        let rebuild = weight_drift(weights, &self.built) > self.rebuild_drift;
        let x_scale: Vec<f64> = weights.iter().map(|w| 1.0 / w.sqrt()).collect();
        self.weights = weights.to_vec();
        if rebuild {
            self.built = weights.to_vec();
            self.num_rebuilds += 1;
        } else {
            self.num_rescales += 1;
        }
        Ok(ScaleUpdate { x_scale, rebuild })
    }

    pub fn refit(&mut self) -> Result<Option<ScaleUpdate>, ENNError> {
        let n = self.len;
        let (w, gain) = auto_weights(
            &self.xs[..n * self.num_dim],
            n,
            self.num_dim,
            &self.ys[..n * self.num_outputs],
            self.num_outputs,
            &self.tied,
        );
        self.heldout_gain = Some(gain);
        self.num_refits += 1;
        let grown = (self.refit_growth * self.num_seen as f64).ceil() as usize;
        self.next_refit = grown.max(MIN_DEPENDENCE_ROWS);
        let target = if auto_uses_learned_metric(gain) { w } else { vec![1.0; self.num_dim] };
        let change = 0.5
            * target
                .iter()
                .zip(self.weights.iter())
                .map(|(t, w)| (t / w).ln().abs())
                .fold(0.0, f64::max);
        if change > AUTO_RESCALE_TOL {
            return Ok(Some(self.set_weights(&target)?));
        }
        Ok(None)
    }

    pub fn observe(&mut self, x: &[f64], y: &[f64], n: usize) -> Result<Option<ScaleUpdate>, ENNError> {
        for i in 0..n {
            let xr = &x[i * self.num_dim..(i + 1) * self.num_dim];
            let yr = &y[i * self.num_outputs..(i + 1) * self.num_outputs];
            self.push_row(xr, yr);
        }
        if self.num_seen >= self.next_refit {
            return self.refit();
        }
        Ok(None)
    }
}
