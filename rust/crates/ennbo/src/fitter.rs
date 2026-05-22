//! Stateful ENN hyperparameter fitting with incremental statistics and fit policy.

use ndarray::{Array1, ArrayView2, Axis};
use rand::Rng;

use crate::error::ENNError;
use crate::fit::subsample_loglik;
use crate::model::EpistemicNearestNeighbors;
use crate::params::ENNParams;

/// Fit probability `p = min(1, 100/N)` for `N` training observations.
#[must_use]
pub fn fit_probability(n: usize) -> f64 {
    if n == 0 {
        return 1.0;
    }
    (100.0 / n as f64).min(1.0)
}

/// Random candidate count `max(1, int(100/N + 0.5))` (warm-start is added separately).
#[must_use]
pub fn num_random_fit_candidates(n: usize) -> usize {
    if n == 0 {
        return 1;
    }
    (((100.0 / n as f64) + 0.5).floor() as usize).max(1)
}

/// Stateful ENN fitter: running `y` moments, warm-start params, and fit policy.
pub struct ENNFitter {
    k: i32,
    num_fit_samples: usize,
    infer_aleatoric_variance: bool,
    params: Option<ENNParams>,
    y_sum: Array1<f64>,
    y_sumsq: Array1<f64>,
    y_count: usize,
    num_metrics: usize,
}

impl ENNFitter {
    pub fn new(
        k: i32,
        num_fit_samples: usize,
        infer_aleatoric_variance: bool,
        num_metrics: usize,
    ) -> Self {
        Self {
            k,
            num_fit_samples,
            infer_aleatoric_variance,
            params: None,
            y_sum: Array1::zeros(num_metrics),
            y_sumsq: Array1::zeros(num_metrics),
            y_count: 0,
            num_metrics,
        }
    }

    pub fn params(&self) -> Option<&ENNParams> {
        self.params.as_ref()
    }

    pub fn set_params(&mut self, params: ENNParams) {
        self.params = Some(params);
    }

    pub fn reset_y_stats(&mut self, y: &ArrayView2<f64>) {
        self.y_count = y.nrows();
        self.num_metrics = y.ncols();
        self.y_sum = y.sum_axis(Axis(0));
        self.y_sumsq = y.mapv(|v| v * v).sum_axis(Axis(0));
    }

    pub fn update_y(&mut self, y_new: &ArrayView2<f64>) {
        if y_new.nrows() == 0 {
            return;
        }
        self.num_metrics = y_new.ncols();
        if self.y_count == 0 {
            self.reset_y_stats(y_new);
            return;
        }
        for row in y_new.axis_iter(Axis(0)) {
            for m in 0..self.num_metrics {
                let v = row[m];
                self.y_sum[m] += v;
                self.y_sumsq[m] += v * v;
            }
            self.y_count += 1;
        }
    }

    fn y_std(&self) -> Array1<f64> {
        if self.y_count == 0 {
            return Array1::ones(self.num_metrics.max(1));
        }
        let n = self.y_count as f64;
        let mut std = Array1::zeros(self.num_metrics);
        for m in 0..self.num_metrics {
            let mean = self.y_sum[m] / n;
            let var = (self.y_sumsq[m] / n - mean * mean).max(0.0);
            std[m] = var.sqrt();
        }
        std.mapv(|v| if v.is_finite() && v > 0.0 { v } else { 1.0 })
    }

    pub(crate) fn build_param_candidates<R: Rng>(
        &self,
        num_random: usize,
        rng: &mut R,
    ) -> Result<Vec<ENNParams>, ENNError> {
        let log_min = -3.0;
        let log_max = 3.0;
        let epi: Vec<f64> = (0..num_random)
            .map(|_| 10f64.powf(rng.gen_range(log_min..=log_max)))
            .collect();
        let ale: Vec<f64> = if self.infer_aleatoric_variance {
            (0..num_random)
                .map(|_| 10f64.powf(rng.gen_range(log_min..=log_max)))
                .collect()
        } else {
            vec![0.0; num_random]
        };
        let mut paramss: Vec<ENNParams> = epi
            .iter()
            .zip(ale.iter())
            .filter_map(|(&e, &a)| ENNParams::new(self.k, e, a).ok())
            .collect();
        if let Some(warm) = self.params.as_ref() {
            let warm_params = ENNParams::new(
                self.k,
                warm.epistemic_variance_scale,
                if self.infer_aleatoric_variance {
                    warm.aleatoric_variance_scale
                } else {
                    0.0
                },
            )
            .map_err(|e| ENNError::InvalidParameter(format!("Invalid warm-start params: {e}")))?;
            paramss.push(warm_params);
        }
        if paramss.is_empty() {
            return ENNParams::new(self.k, 1.0, 0.0)
                .map(|p| vec![p])
                .map_err(|e| ENNError::InvalidParameter(format!("Failed to create default params: {e}")));
        }
        Ok(paramss)
    }

    /// Fit with policy gate. `fit_prob_override` of `1.0` forces a fit (used by `enn_fit`).
    pub fn maybe_fit<R: Rng>(
        &mut self,
        model: &EpistemicNearestNeighbors,
        rng: &mut R,
        fit_prob_override: Option<f64>,
    ) -> Result<Option<ENNParams>, ENNError> {
        let n = model.num_obs();
        if n == 0 {
            return Ok(None);
        }
        let p = fit_prob_override.unwrap_or_else(|| fit_probability(n));
        let has_params = self.params.is_some();
        if has_params && rng.gen::<f64>() > p {
            return Ok(None);
        }
        let num_random = num_random_fit_candidates(n);
        let paramss = self.build_param_candidates(num_random, rng)?;
        let train_x = model.train_x();
        let train_y = model.train_y();
        let y_std = self.y_std();
        let logliks = subsample_loglik(
            model,
            &train_x,
            &train_y,
            &paramss,
            self.num_fit_samples,
            rng,
            Some(&y_std.view()),
        )?;
        let best_idx = logliks
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx)
            .unwrap_or(0);
        let best = paramss[best_idx];
        self.params = Some(best);
        Ok(Some(best))
    }
}
