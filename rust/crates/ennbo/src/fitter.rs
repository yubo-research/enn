//! Stateful ENN hyperparameter fitting with incremental statistics.

use ndarray::{Array1, Array2, ArrayView2, Axis};
use rand::Rng;

use crate::error::ENNError;
use crate::model::EpistemicNearestNeighbors;
use crate::params::ENNParams;
use crate::y_bounds::{bounds_match, is_identity_bounds, validate_bounds, warp_y, warp_yvar};

/// Stateful ENN fitter: running `y` moments and warm-start params.
pub struct ENNFitter {
    k: i32,
    infer_aleatoric_variance: bool,
    params: Option<ENNParams>,
    y_sum: Array1<f64>,
    y_sumsq: Array1<f64>,
    y_count: usize,
    num_metrics: usize,
    /// Feature width from the first `tell`; must match the model at `ask`.
    num_dim: Option<usize>,
    /// Optional output bounds; when set, `tell` accumulates warped-z moments.
    y_bounds: Option<Array2<f64>>,
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
            num_dim: None,
            y_bounds: None,
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

    fn tell_input_error(
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        expected_num_dim: Option<usize>,
    ) -> Option<String> {
        if x.iter().any(|v| !v.is_finite()) {
            return Some("x must contain only finite values".to_string());
        }
        if y.iter().any(|v| !v.is_finite()) {
            return Some("y must contain only finite values".to_string());
        }
        if x.nrows() != y.nrows() {
            return Some(format!(
                "x and y must have same number of rows: {} vs {}",
                x.nrows(),
                y.nrows()
            ));
        }
        if let Some(dim) = expected_num_dim {
            if x.ncols() != dim {
                return Some(format!(
                    "x has {} columns but fitter expects {} feature dimensions",
                    x.ncols(),
                    dim
                ));
            }
        }
        if let Some(yv) = yvar {
            if yv.iter().any(|v| !v.is_finite()) {
                return Some("yvar must contain only finite values".to_string());
            }
            if yv.iter().any(|&v| v < 0.0) || yv.shape() != y.shape() {
                return Some(if yv.iter().any(|&v| v < 0.0) {
                    "yvar must be non-negative".to_string()
                } else {
                    format!(
                        "yvar shape {:?} must match y shape {:?}",
                        yv.shape(),
                        y.shape()
                    )
                });
            }
        }
        None
    }

    /// Register a batch for incremental `y_std`.
    ///
    /// Pass natural-unit `y` / `yvar`. When `y_bounds` is non-identity, values are
    /// warped to z before updating moments (fit stays in warped space).
    pub fn tell(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
        y_bounds: Option<&Array2<f64>>,
    ) -> Result<(), ENNError> {
        if let Some(msg) = Self::tell_input_error(x, y, yvar, self.num_dim) {
            return Err(ENNError::InvalidParameter(msg));
        }
        if let Some(b) = y_bounds {
            validate_bounds(b, y.ncols())?;
            if let Some(prev) = &self.y_bounds {
                if !bounds_match(prev, b) {
                    return Err(ENNError::InvalidParameter(
                        "y_bounds must match across tell() calls".to_string(),
                    ));
                }
            } else {
                self.y_bounds = Some(b.clone());
            }
        }
        if self.num_dim.is_none() {
            self.num_dim = Some(x.ncols());
        }
        let bounds = y_bounds.or(self.y_bounds.as_ref());
        if let Some(b) = bounds {
            if !is_identity_bounds(b) {
                let y_z = warp_y(*y, b)?;
                let yvar_z = match yvar {
                    Some(yv) => Some(warp_yvar(*y, *yv, b)?),
                    None => None,
                };
                let _ = yvar_z;
                self.update_y(&y_z.view());
                return Ok(());
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
        if let Some(dim) = self.num_dim {
            if dim != model.num_dim() {
                return Err(ENNError::InvalidParameter(format!(
                    "x has {} columns but model expects {} feature dimensions",
                    dim,
                    model.num_dim()
                )));
            }
        }
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
        assert!(fitter.tell(&x.view(), &y.view(), None, None).is_err());
    }

    #[test]
    fn tell_rejects_non_finite_x() {
        let mut fitter = ENNFitter::new(2, true);
        let x = array![[f64::NAN, 0.0]];
        let y = array![[0.0]];
        assert!(fitter.tell(&x.view(), &y.view(), None, None).is_err());
    }

    #[test]
    fn tell_rejects_shape_mismatch_and_bad_yvar() {
        let mut fitter = ENNFitter::new(2, true);
        let x = array![[0.0, 0.0], [1.0, 0.0]];
        let y = array![[0.0]];
        assert!(fitter.tell(&x.view(), &y.view(), None, None).is_err());
        let yvar = array![[0.1, 0.2]];
        assert!(fitter.tell(&x.view(), &y.view(), Some(&yvar.view()), None).is_err());
        let yvar_bad = array![[f64::INFINITY]];
        assert!(fitter.tell(&x.view(), &y.view(), Some(&yvar_bad.view()), None).is_err());
        let yvar_neg = array![[-0.1], [-0.1]];
        assert!(fitter.tell(&x.view(), &y.view(), Some(&yvar_neg.view()), None).is_err());
    }

    #[test]
    fn ask_rejects_x_ncols_mismatch_vs_model() {
        let train_x = array![[0.0, 0.0], [1.0, 1.0]];
        let train_y = array![[0.0], [1.0]];
        let model =
            EpistemicNearestNeighbors::new(train_x, train_y, None, false, IndexDriver::Exact)
                .unwrap();
        let mut fitter = ENNFitter::new(2, true);
        let x_bad = array![[0.0], [1.0]];
        let y = array![[0.0], [1.0]];
        fitter.tell(&x_bad.view(), &y.view(), None, None).unwrap();
        let mut rng = StdRng::seed_from_u64(0);
        let err = fitter.ask(&model, 5, 3, None, &mut rng).unwrap_err();
        assert!(
            err.to_string().contains("feature dimensions"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn tell_rejects_inconsistent_x_ncols() {
        let mut fitter = ENNFitter::new(2, true);
        let x2 = array![[0.0, 0.0]];
        let y = array![[0.0]];
        fitter.tell(&x2.view(), &y.view(), None, None).unwrap();
        let x1 = array![[1.0]];
        assert!(fitter.tell(&x1.view(), &y.view(), None, None).is_err());
    }

    #[test]
    fn ask_uses_explicit_warm_start() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let train_y = array![[0.0], [1.0], [1.0], [2.0]];
        let model =
            EpistemicNearestNeighbors::new(train_x.clone(), train_y.clone(), None, false, IndexDriver::Exact)
                .unwrap();
        let mut fitter = ENNFitter::new(2, true);
        fitter.tell(&train_x.view(), &train_y.view(), None, None).unwrap();
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
            fitter.tell(&x_row, &y_row, None, None).unwrap();
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

    #[test]
    fn tell_with_y_bounds_accumulates_warped_y_std() {
        let x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let y = array![[0.1], [0.2], [0.8], [0.9]];
        let bounds = array![[0.0, 1.0]];
        let y_z = crate::y_bounds::warp_y(y.view(), &bounds).unwrap();
        let mut fitter = ENNFitter::new(2, true);
        fitter
            .tell(&x.view(), &y.view(), None, Some(&bounds))
            .unwrap();
        let got = fitter.y_std()[0];
        let want = y_z.std_axis(Axis(0), 0.0)[0];
        assert!(
            (got - want).abs() < 1e-12,
            "tell must warp under y_bounds: got {got} want {want}"
        );
        assert!(got > 1.5, "warped std should exceed natural (~0.35), got {got}");
    }

    #[test]
    fn tell_rejects_out_of_bounds_y_when_y_bounds_set() {
        let mut fitter = ENNFitter::new(2, true);
        let x = array![[0.0, 0.0]];
        let y = array![[1.5]];
        let bounds = array![[0.0, 1.0]];
        assert!(fitter.tell(&x.view(), &y.view(), None, Some(&bounds)).is_err());
    }

    #[test]
    fn kiss_fitter_update_y_and_random_params() {
        let mut fitter = ENNFitter::new(2, true);
        let y = array![[1.0], [2.0]];
        fitter.update_y(&y.view());
        let mut rng = StdRng::seed_from_u64(0);
        let cands = fitter.build_random_param_candidates(3, &mut rng).unwrap();
        assert_eq!(cands.len(), 3);
    }
}
