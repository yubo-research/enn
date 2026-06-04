//! Stateful ENN hyperparameter fitting with incremental statistics.

use ndarray::{Array1, ArrayView2, Axis};
use rand::Rng;

use crate::error::ENNError;
use crate::model::EpistemicNearestNeighbors;
use crate::params::ENNParams;

/// Stateful ENN fitter: running `y` moments and warm-start params.
pub struct ENNFitter {
    k: i32,
    infer_aleatoric_variance: bool,
    params: Option<ENNParams>,
    y_sum: Array1<f64>,
    y_sumsq: Array1<f64>,
    y_count: usize,
    num_metrics: usize,
}

impl ENNFitter {
    pub fn new(k: i32, infer_aleatoric_variance: bool) -> Self {
        Self {
            k,
            infer_aleatoric_variance,
            params: None,
            y_sum: Array1::zeros(0),
            y_sumsq: Array1::zeros(0),
            y_count: 0,
            num_metrics: 0,
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

    pub fn tell(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Result<(), ENNError> {
        if x.iter().any(|v| !v.is_finite()) {
            return Err(ENNError::InvalidParameter(
                "x must contain only finite values".to_string(),
            ));
        }
        if y.iter().any(|v| !v.is_finite()) {
            return Err(ENNError::InvalidParameter(
                "y must contain only finite values".to_string(),
            ));
        }
        if x.nrows() != y.nrows() {
            return Err(ENNError::InvalidParameter(format!(
                "x and y must have same number of rows: {} vs {}",
                x.nrows(),
                y.nrows()
            )));
        }
        if let Some(yv) = yvar {
            if yv.iter().any(|v| !v.is_finite()) {
                return Err(ENNError::InvalidParameter(
                    "yvar must contain only finite values".to_string(),
                ));
            }
            if yv.nrows() != y.nrows() {
                return Err(ENNError::InvalidParameter(format!(
                    "yvar and y must have same number of rows: {} vs {}",
                    yv.nrows(),
                    y.nrows()
                )));
            }
            if yv.ncols() != y.ncols() {
                return Err(ENNError::InvalidParameter(format!(
                    "yvar and y must have same number of columns: {} vs {}",
                    yv.ncols(),
                    y.ncols()
                )));
            }
        }
        self.update_y(y);
        Ok(())
    }

    pub fn y_std(&self) -> Array1<f64> {
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

    pub(crate) fn build_random_param_candidates<R: Rng>(
        &self,
        num_random: usize,
        rng: &mut R,
    ) -> Result<Vec<ENNParams>, ENNError> {
        if num_random == 0 {
            return Ok(vec![]);
        }
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
        let paramss: Vec<ENNParams> = epi
            .iter()
            .zip(ale.iter())
            .filter_map(|(&e, &a)| ENNParams::new(self.k, e, a).ok())
            .collect();
        if paramss.is_empty() {
            return ENNParams::new(self.k, 1.0, 0.0)
                .map(|p| vec![p])
                .map_err(|e| {
                    ENNError::InvalidParameter(format!("Failed to create default params: {e}"))
                });
        }
        Ok(paramss)
    }

    pub fn ask<R: Rng>(
        &mut self,
        model: &EpistemicNearestNeighbors,
        num_fit_candidates: usize,
        num_fit_samples: usize,
        params_warm_start: Option<&ENNParams>,
        rng: &mut R,
    ) -> Result<ENNParams, ENNError> {
        if model.num_obs() < 2 {
            let best = ENNParams::new(self.k, 1.0, 0.0).map_err(|e| {
                ENNError::InvalidParameter(format!("Failed to create default params: {e}"))
            })?;
            self.params = Some(best);
            return Ok(best);
        }
        if self.y_count == 0 {
            return Err(ENNError::InvalidParameter(
                "tell must be called before ask to initialize incremental y statistics"
                    .to_string(),
            ));
        }
        let mut paramss = self.build_random_param_candidates(num_fit_candidates, rng)?;
        let warm = params_warm_start.or(self.params.as_ref());
        if let Some(warm) = warm {
            let warm_params = ENNParams::new(
                self.k,
                warm.epistemic_variance_scale,
                if self.infer_aleatoric_variance {
                    warm.aleatoric_variance_scale
                } else {
                    0.0
                },
            )
            .map_err(|e| {
                ENNError::InvalidParameter(format!("Invalid warm-start params: {e}"))
            })?;
            paramss.push(warm_params);
        }
        let indices: Vec<usize> = {
            let n = model.len();
            let p_actual = num_fit_samples.min(n);
            if p_actual == n {
                (0..n).collect()
            } else {
                use rand::seq::index::sample;
                sample(rng, n, p_actual).into_iter().collect()
            }
        };
        let (train_x, train_y, _) = model.rows().train_rows_at(&indices)?;
        let y_std = self.y_std();
        let logliks = crate::fit::subsample_loglik(
            model,
            &train_x.view(),
            &train_y.view(),
            &paramss,
            num_fit_samples,
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
        Ok(best)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::IndexDriver;
    use ndarray::{array, s};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn tell_rejects_non_finite_y() {
        let mut fitter = ENNFitter::new(2, true);
        let x = array![[0.0, 0.0]];
        let y = array![[f64::NAN]];
        assert!(fitter.tell(&x.view(), &y.view(), None).is_err());
    }

    #[test]
    fn tell_rejects_non_finite_x() {
        let mut fitter = ENNFitter::new(2, true);
        let x = array![[f64::NAN, 0.0]];
        let y = array![[0.0]];
        assert!(fitter.tell(&x.view(), &y.view(), None).is_err());
    }

    #[test]
    fn tell_rejects_shape_mismatch_and_bad_yvar() {
        let mut fitter = ENNFitter::new(2, true);
        let x = array![[0.0, 0.0], [1.0, 0.0]];
        let y = array![[0.0]];
        assert!(fitter.tell(&x.view(), &y.view(), None).is_err());
        let yvar = array![[0.1, 0.2]];
        assert!(fitter.tell(&x.view(), &y.view(), Some(&yvar.view())).is_err());
        let yvar_bad = array![[f64::INFINITY]];
        assert!(fitter.tell(&x.view(), &y.view(), Some(&yvar_bad.view())).is_err());
    }

    #[test]
    fn ask_uses_explicit_warm_start() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let train_y = array![[0.0], [1.0], [1.0], [2.0]];
        let model =
            EpistemicNearestNeighbors::new(train_x.clone(), train_y.clone(), None, false, IndexDriver::Exact)
                .unwrap();
        let mut fitter = ENNFitter::new(2, true);
        fitter.tell(&train_x.view(), &train_y.view(), None).unwrap();
        let warm = ENNParams::new(2, 2.5, 0.3).unwrap();
        let mut rng = StdRng::seed_from_u64(7);
        let p = fitter
            .ask(&model, 0, 2, Some(&warm), &mut rng)
            .unwrap();
        assert_eq!(p.k_num_neighbors, 2);
        assert!((p.epistemic_variance_scale - 2.5).abs() < 1e-12);
    }

    #[test]
    fn ask_warm_start_zeros_aleatoric_when_not_inferred() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let train_y = array![[0.0], [1.0], [1.0], [2.0]];
        let model =
            EpistemicNearestNeighbors::new(train_x, train_y, None, false, IndexDriver::Exact)
                .unwrap();
        let mut fitter = ENNFitter::new(2, false);
        let all: Vec<usize> = (0..model.len()).collect();
        let (_, ty, _) = model.rows().train_rows_at(&all).unwrap();
        fitter.reset_y_stats(&ty.view());
        let warm = ENNParams::new(2, 2.5, 9.9).unwrap();
        let mut rng = StdRng::seed_from_u64(8);
        let p = fitter
            .ask(&model, 2, 2, Some(&warm), &mut rng)
            .unwrap();
        assert_eq!(p.aleatoric_variance_scale, 0.0);
    }

    #[test]
    fn ask_returns_defaults_when_num_obs_lt_2() {
        let train_x = array![[0.0, 0.0]];
        let train_y = array![[0.0]];
        let model =
            EpistemicNearestNeighbors::new(train_x, train_y, None, false, IndexDriver::Exact)
                .unwrap();
        let mut fitter = ENNFitter::new(3, true);
        let mut rng = StdRng::seed_from_u64(1);
        let p = fitter.ask(&model, 5, 3, None, &mut rng).unwrap();
        assert_eq!(p.k_num_neighbors, 3);
        assert!((p.epistemic_variance_scale - 1.0).abs() < 1e-12);
        assert!((p.aleatoric_variance_scale - 0.0).abs() < 1e-12);
    }

    #[test]
    fn incremental_y_std_matches_batch_std() {
        let train_x = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5]
        ];
        let train_y = array![[0.0], [1.0], [1.0], [2.0], [1.5]];
        let _model = EpistemicNearestNeighbors::new(
            train_x.clone(),
            train_y.clone(),
            None,
            false,
            IndexDriver::Exact,
        )
        .unwrap();
        let mut fitter = ENNFitter::new(2, true);
        for (i, row) in train_y.axis_iter(Axis(0)).enumerate() {
            let y_row = row.insert_axis(Axis(0));
            let x_row = train_x.slice(s![i..i + 1, ..]);
            fitter.tell(&x_row, &y_row, None).unwrap();
        }
        let batch_std = train_y.std_axis(Axis(0), 0.0);
        let inc_std = fitter.y_std();
        for (a, b) in inc_std.iter().zip(batch_std.iter()) {
            if *b > 1e-10 {
                assert!((a - b).abs() < 1e-10, "incremental std {a} vs batch std {b}");
            } else {
                assert!((*a - 1.0).abs() < 1e-10, "zero-variance metric should clamp to 1.0");
            }
        }
    }
}
