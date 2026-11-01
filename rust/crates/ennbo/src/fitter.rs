use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::seq::index::sample;
use rand::Rng;
use std::sync::Arc;

use crate::calibration::AffineCalibrator;
use crate::error::ENNError;
use crate::fit_samples::ScaleSearch;
use crate::model::EpistemicNearestNeighbors;
use crate::output_blend::OutputBlend;
use crate::output_blend_fit::OutputBlendFit;
use crate::params::ENNParams;
use crate::y_bounds::{bounds_match, is_identity_bounds, validate_bounds, warp_y, warp_yvar};

pub struct ENNFitter {
    pub(crate) k: i32,
    pub(crate) params: Option<ENNParams>,
    y_sum: Array1<f64>,
    y_sumsq: Array1<f64>,
    y_count: usize,
    num_metrics: usize,
    num_dim: Option<usize>,
    y_bounds: Option<Array2<f64>>,
    calibrator: Option<AffineCalibrator>,
    output_fit: Option<OutputBlendFit>,
}

impl ENNFitter {
    pub fn new(k: i32) -> Self {
        Self {
            k,
            params: None,
            y_sum: Array1::zeros(0),
            y_sumsq: Array1::zeros(0),
            y_count: 0,
            num_metrics: 0,
            num_dim: None,
            y_bounds: None,
            calibrator: None,
            output_fit: None,
        }
    }

    pub fn calibrator(&self) -> Option<&AffineCalibrator> {
        self.calibrator.as_ref()
    }

    pub fn set_calibrator_rows(&mut self, rows: ArrayView2<f64>) -> Result<(), ENNError> {
        self.calibrator = Some(AffineCalibrator::from_rows(rows)?);
        Ok(())
    }

    pub fn set_calibrator_abc(
        &mut self,
        a: ArrayView1<f64>,
        b: ArrayView1<f64>,
        c: ArrayView1<f64>,
    ) -> Result<(), ENNError> {
        if a.len() != b.len() || b.len() != c.len() {
            return Err(ENNError::InvalidShape {
                expected: vec![a.len()],
                got: vec![b.len(), c.len()],
            });
        }
        let mut rows = Array2::zeros((3, a.len()));
        rows.row_mut(0).assign(&a);
        rows.row_mut(1).assign(&b);
        rows.row_mut(2).assign(&c);
        self.set_calibrator_rows(rows.view())
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

    pub(crate) fn seed_y_stats(&mut self, model: &EpistemicNearestNeighbors) {
        let (sum, sumsq) = model.warped_y_sums();
        self.y_sum = sum.clone();
        self.y_sumsq = sumsq.clone();
        self.y_count = model.len();
        self.num_metrics = model.num_metrics();
        self.num_dim = Some(model.num_dim());
        self.y_bounds = Some(model.y_bounds().clone());
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
        self.calibrator = None;
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
        infer_aleatoric_variance: bool,
        rng: &mut R,
    ) -> Result<Vec<ENNParams>, ENNError> {
        let log_min = -3.0;
        let log_max = 3.0;
        let epi: Vec<f64> = (0..num_random)
            .map(|_| 10f64.powf(rng.gen_range(log_min..=log_max)))
            .collect();
        let ale: Vec<f64> = if infer_aleatoric_variance {
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

    fn commit_ask<R: Rng>(
        &mut self,
        model: &EpistemicNearestNeighbors,
        best: ENNParams,
        num_fit_samples: usize,
        rng: &mut R,
        affine_calibrate: bool,
    ) -> Result<ENNParams, ENNError> {
        self.params = Some(best.clone());
        self.calibrator = if affine_calibrate {
            Some(crate::surrogate_affine::fit_calibrator(
                model,
                &best,
                num_fit_samples,
                rng,
                false,
            )?)
        } else {
            None
        };
        Ok(best)
    }

    fn fit_output_blend<R: Rng>(
        &mut self,
        model: &EpistemicNearestNeighbors,
        warm: Option<&Arc<OutputBlend>>,
        rng: &mut R,
    ) -> Result<Option<Arc<OutputBlend>>, ENNError> {
        let Some(outputs) = model.semantic_outputs() else {
            return Ok(None);
        };
        let k = self.k;
        self.output_fit
            .get_or_insert_with(|| OutputBlendFit::new(outputs))
            .fit(model, outputs, k, warm, rng)
            .map(Some)
    }

    pub fn ask<R: Rng>(
        &mut self,
        model: &EpistemicNearestNeighbors,
        search: &ScaleSearch,
        params_warm_start: Option<&ENNParams>,
        rng: &mut R,
    ) -> Result<ENNParams, ENNError> {
        let num_fit_samples = search.num_fit_samples.get();
        let affine_calibrate = search.affine_calibrate;
        if let Some(dim) = self.num_dim {
            if dim != model.num_dim() {
                return Err(ENNError::InvalidParameter(format!(
                    "x has {} columns but model expects {} feature dimensions",
                    dim,
                    model.num_dim()
                )));
            }
        }
        if model.num_obs() >= 2 && self.y_count == 0 {
            return Err(ENNError::InvalidParameter(
                "tell must be called before ask to initialize incremental y statistics".to_string(),
            ));
        }
        let warm = params_warm_start.or(self.params.as_ref()).cloned();
        let warm_blend = params_warm_start.and_then(|w| w.output_blend.as_ref());
        let blend = self.fit_output_blend(model, warm_blend, rng)?;
        if model.num_obs() < 2 {
            let best = ENNParams::new(self.k, 1.0, 0.0)
                .map_err(|e| {
                    ENNError::InvalidParameter(format!("Failed to create default params: {e}"))
                })?
                .with_output_blend(blend);
            return self.commit_ask(model, best, num_fit_samples, rng, affine_calibrate);
        }
        let mut paramss = self.build_random_param_candidates(
            search.num_fit_candidates.get(),
            search.infer_aleatoric_variance,
            rng,
        )?;
        if let Some(warm) = warm {
            let warm_params = ENNParams::new(
                self.k,
                warm.epistemic_variance_scale,
                if search.infer_aleatoric_variance {
                    warm.aleatoric_variance_scale
                } else {
                    0.0
                },
            )
            .map_err(|e| ENNError::InvalidParameter(format!("Invalid warm-start params: {e}")))?;
            paramss.push(warm_params);
        }
        for params in &mut paramss {
            params.output_blend.clone_from(&blend);
        }
        let indices: Vec<usize> = {
            let n = model.len();
            let p_actual = num_fit_samples.min(n);
            if p_actual == n {
                (0..n).collect()
            } else {
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
        let best = paramss.swap_remove(best_idx);
        self.commit_ask(model, best, num_fit_samples, rng, affine_calibrate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, s};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn tell_rejects_non_finite_y() {
        let mut fitter = ENNFitter::new(2);
        let x = array![[0.0, 0.0]];
        let y = array![[f64::NAN]];
        assert!(fitter.tell(&x.view(), &y.view(), None, None).is_err());
    }

    #[test]
    fn tell_rejects_non_finite_x() {
        let mut fitter = ENNFitter::new(2);
        let x = array![[f64::NAN, 0.0]];
        let y = array![[0.0]];
        assert!(fitter.tell(&x.view(), &y.view(), None, None).is_err());
    }

    #[test]
    fn tell_rejects_shape_mismatch_and_bad_yvar() {
        let mut fitter = ENNFitter::new(2);
        let x = array![[0.0, 0.0], [1.0, 0.0]];
        let y = array![[0.0]];
        assert!(fitter.tell(&x.view(), &y.view(), None, None).is_err());
        let yvar = array![[0.1, 0.2]];
        assert!(fitter
            .tell(&x.view(), &y.view(), Some(&yvar.view()), None)
            .is_err());
        let yvar_bad = array![[f64::INFINITY]];
        assert!(fitter
            .tell(&x.view(), &y.view(), Some(&yvar_bad.view()), None)
            .is_err());
        let yvar_neg = array![[-0.1], [-0.1]];
        assert!(fitter
            .tell(&x.view(), &y.view(), Some(&yvar_neg.view()), None)
            .is_err());
    }

    #[test]
    fn ask_rejects_x_ncols_mismatch_vs_model() {
        let train_x = array![[0.0, 0.0], [1.0, 1.0]];
        let train_y = array![[0.0], [1.0]];
        let model = crate::test_helpers::test_model(train_x, train_y, None, false);
        let mut fitter = ENNFitter::new(2);
        let x_bad = array![[0.0], [1.0]];
        let y = array![[0.0], [1.0]];
        fitter.tell(&x_bad.view(), &y.view(), None, None).unwrap();
        let mut rng = StdRng::seed_from_u64(0);
        let err = fitter
            .ask(
                &model,
                &crate::fit_samples::test_search(3, 5, true),
                None,
                &mut rng,
            )
            .unwrap_err();
        assert!(
            err.to_string().contains("feature dimensions"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn tell_rejects_inconsistent_x_ncols() {
        let mut fitter = ENNFitter::new(2);
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
        let model = crate::test_helpers::test_model(train_x.clone(), train_y.clone(), None, false);
        let warm = ENNParams::new(2, 2.5, 0.3).unwrap();
        let search = crate::fit_samples::test_search(2, 1, true);
        let ask = |seed: u64, warm: Option<&ENNParams>| {
            let mut fitter = ENNFitter::new(2);
            fitter
                .tell(&train_x.view(), &train_y.view(), None, None)
                .unwrap();
            fitter
                .ask(&model, &search, warm, &mut StdRng::seed_from_u64(seed))
                .unwrap()
        };
        let mut warm_won = false;
        for seed in 0..20 {
            let cold = ask(seed, None);
            let p = ask(seed, Some(&warm));
            assert_eq!(p.k_num_neighbors, 2);
            assert!(
                p == cold || p == warm,
                "seed {seed}: {p:?} is neither {cold:?} nor {warm:?}"
            );
            warm_won |= p == warm;
        }
        assert!(
            warm_won,
            "the warm start never beat the single random candidate"
        );
    }

    #[test]
    fn ask_warm_start_zeros_aleatoric_when_not_inferred() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let train_y = array![[0.0], [1.0], [1.0], [2.0]];
        let model = crate::test_helpers::test_model(train_x, train_y, None, false);
        let mut fitter = ENNFitter::new(2);
        let all: Vec<usize> = (0..model.len()).collect();
        let (_, ty, _) = model.rows().train_rows_at(&all).unwrap();
        fitter.reset_y_stats(&ty.view());
        let warm = ENNParams::new(2, 2.5, 9.9).unwrap();
        let mut rng = StdRng::seed_from_u64(8);
        let p = fitter
            .ask(
                &model,
                &crate::fit_samples::test_search(2, 2, false),
                Some(&warm),
                &mut rng,
            )
            .unwrap();
        assert_eq!(p.aleatoric_variance_scale, 0.0);
    }

    #[test]
    fn ask_returns_defaults_when_num_obs_lt_2() {
        let train_x = array![[0.0, 0.0]];
        let train_y = array![[0.0]];
        let model = crate::test_helpers::test_model(train_x, train_y, None, false);
        let mut fitter = ENNFitter::new(3);
        let mut rng = StdRng::seed_from_u64(1);
        let p = fitter
            .ask(
                &model,
                &crate::fit_samples::test_search(3, 5, true),
                None,
                &mut rng,
            )
            .unwrap();
        assert_eq!(p.k_num_neighbors, 3);
        assert!((p.epistemic_variance_scale - 1.0).abs() < 1e-12);
        assert!((p.aleatoric_variance_scale - 0.0).abs() < 1e-12);
    }

    #[test]
    fn incremental_y_std_matches_batch_std() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];
        let train_y = array![[0.0], [1.0], [1.0], [2.0], [1.5]];
        let _model = crate::test_helpers::test_model(train_x.clone(), train_y.clone(), None, false);
        let mut fitter = ENNFitter::new(2);
        for (i, row) in train_y.axis_iter(Axis(0)).enumerate() {
            let y_row = row.insert_axis(Axis(0));
            let x_row = train_x.slice(s![i..i + 1, ..]);
            fitter.tell(&x_row, &y_row, None, None).unwrap();
        }
        let batch_std = train_y.std_axis(Axis(0), 0.0);
        let inc_std = fitter.y_std();
        for (a, b) in inc_std.iter().zip(batch_std.iter()) {
            if *b > 1e-10 {
                assert!(
                    (a - b).abs() < 1e-10,
                    "incremental std {a} vs batch std {b}"
                );
            } else {
                assert!(
                    (*a - 1.0).abs() < 1e-10,
                    "zero-variance metric should clamp to 1.0"
                );
            }
        }
    }

    #[test]
    fn tell_with_y_bounds_accumulates_warped_y_std() {
        let x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let y = array![[0.1], [0.2], [0.8], [0.9]];
        let bounds = array![[0.0, 1.0]];
        let y_z = crate::y_bounds::warp_y(y.view(), &bounds).unwrap();
        let mut fitter = ENNFitter::new(2);
        fitter
            .tell(&x.view(), &y.view(), None, Some(&bounds))
            .unwrap();
        let got = fitter.y_std()[0];
        let want = y_z.std_axis(Axis(0), 0.0)[0];
        assert!(
            (got - want).abs() < 1e-12,
            "tell must warp under y_bounds: got {got} want {want}"
        );
        assert!(
            got > 1.5,
            "warped std should exceed natural (~0.35), got {got}"
        );
    }

    #[test]
    fn tell_rejects_out_of_bounds_y_when_y_bounds_set() {
        let mut fitter = ENNFitter::new(2);
        let x = array![[0.0, 0.0]];
        let y = array![[1.5]];
        let bounds = array![[0.0, 1.0]];
        assert!(fitter
            .tell(&x.view(), &y.view(), None, Some(&bounds))
            .is_err());
    }

    fn yy_model(alpha: Option<f64>) -> crate::test_helpers::TestModel {
        let x = Array2::from_shape_fn((30, 1), |(i, _)| i as f64 * 0.1);
        let y = Array2::from_shape_fn((30, 2), |(i, j)| (i as f64 * 0.1 + j as f64).sin());
        let mut model = crate::test_helpers::test_model(x, y, None, false);
        let loading = 0.5f64.sqrt();
        model
            .set_semantic_outputs(array![[loading], [loading]].into_dyn().view(), alpha)
            .unwrap();
        model
    }

    fn seeded_fitter(model: &EpistemicNearestNeighbors) -> ENNFitter {
        let mut fitter = ENNFitter::new(3);
        fitter.seed_y_stats(model);
        fitter
    }

    #[test]
    fn yy_model_returns_a_blend_on_every_candidate_path() {
        let model = yy_model(Some(0.3));
        let mut fitter = seeded_fitter(&model);
        let search = crate::fit_samples::test_search(5, 3, true);
        let mut rng = StdRng::seed_from_u64(0);
        let p = fitter.ask(&model, &search, None, &mut rng).unwrap();
        let blend = p.output_blend.as_ref().expect("blend");
        assert_eq!(blend.alpha, array![0.3, 0.3]);
        assert_eq!(fitter.params().unwrap(), &p);
    }

    #[test]
    fn yy_model_with_one_row_returns_defaults_with_a_blend() {
        let x = array![[0.0]];
        let y = array![[1.0, 2.0]];
        let mut model = crate::test_helpers::test_model(x, y, None, false);
        let loading = 0.5f64.sqrt();
        model
            .set_semantic_outputs(array![[loading], [loading]].into_dyn().view(), None)
            .unwrap();
        let mut rng = StdRng::seed_from_u64(0);
        let p = ENNFitter::new(3)
            .ask(
                &model,
                &crate::fit_samples::test_search(3, 5, true),
                None,
                &mut rng,
            )
            .unwrap();
        assert_eq!(p.epistemic_variance_scale, 1.0);
        assert_eq!(p.output_blend.unwrap().alpha, array![0.3, 0.3]);
    }

    fn grow(fitter: &mut ENNFitter, model: &mut EpistemicNearestNeighbors) {
        let x = array![[10.0], [11.0], [12.0]];
        let y = array![[5.0, 7.0], [6.0, 8.0], [7.0, 9.0]];
        model.add(&x.view(), &y.view(), None).unwrap();
        fitter.tell(&x.view(), &y.view(), None, None).unwrap();
    }

    fn short_yy_model() -> crate::test_helpers::TestModel {
        let x = Array2::from_shape_fn((4, 1), |(i, _)| i as f64);
        let y = Array2::from_shape_fn((4, 2), |(i, j)| (i * (j + 1)) as f64 * 0.1);
        let mut model = crate::test_helpers::test_model(x, y, None, false);
        let loading = 0.5f64.sqrt();
        model
            .set_semantic_outputs(array![[loading], [loading]].into_dyn().view(), None)
            .unwrap();
        model
    }

    #[test]
    fn prior_blend_keeps_tracking_the_model_before_twelve_rows() {
        let mut model = short_yy_model();
        let mut fitter = seeded_fitter(&model);
        let search = crate::fit_samples::test_search(5, 3, true);
        let mut rng = StdRng::seed_from_u64(0);
        let first = fitter.ask(&model, &search, None, &mut rng).unwrap();
        grow(&mut fitter, &mut model);
        let second = fitter.ask(&model, &search, None, &mut rng).unwrap();
        let center = |p: &ENNParams| p.output_blend.as_ref().unwrap().center.clone();
        assert_ne!(center(&first), center(&second));
        let (sum, _) = model.warped_y_sums();
        assert_eq!(center(&second), sum / model.len() as f64);
    }

    #[test]
    fn warm_blend_given_once_is_kept_until_twelve_rows() {
        let mut model = short_yy_model();
        let mut fitter = seeded_fitter(&model);
        let search = crate::fit_samples::test_search(5, 3, true);
        let mut rng = StdRng::seed_from_u64(0);
        let warm_blend = Arc::new(
            OutputBlend::new(
                array![0.7, 0.2],
                Array2::eye(2),
                array![0.0, 0.0],
                array![1.0, 1.0],
            )
            .unwrap(),
        );
        let warm = ENNParams::new(3, 1.0, 0.1)
            .unwrap()
            .with_output_blend(Some(warm_blend.clone()));
        fitter.ask(&model, &search, Some(&warm), &mut rng).unwrap();
        grow(&mut fitter, &mut model);
        let later = fitter.ask(&model, &search, None, &mut rng).unwrap();
        assert!(Arc::ptr_eq(
            later.output_blend.as_ref().unwrap(),
            &warm_blend
        ));
    }

    #[test]
    fn plain_model_draws_the_same_candidates_and_returns_no_blend() {
        let plain = {
            let x = Array2::from_shape_fn((30, 1), |(i, _)| i as f64 * 0.1);
            let y = Array2::from_shape_fn((30, 2), |(i, j)| (i as f64 * 0.1 + j as f64).sin());
            crate::test_helpers::test_model(x, y, None, false)
        };
        let pinned_zero = yy_model(Some(0.0));
        let search = crate::fit_samples::test_search(5, 3, true);
        let ask = |model| {
            seeded_fitter(model)
                .ask(model, &search, None, &mut StdRng::seed_from_u64(4))
                .unwrap()
        };
        let got = ask(&plain);
        let zero = ask(&pinned_zero);
        assert!(got.output_blend.is_none());
        assert_eq!(got.epistemic_variance_scale, zero.epistemic_variance_scale);
        assert_eq!(got.aleatoric_variance_scale, zero.aleatoric_variance_scale);
    }

    #[test]
    fn kiss_fitter_update_y_and_random_params() {
        let mut fitter = ENNFitter::new(2);
        let y = array![[1.0], [2.0]];
        fitter.update_y(&y.view());
        let mut rng = StdRng::seed_from_u64(0);
        let cands = fitter
            .build_random_param_candidates(3, true, &mut rng)
            .unwrap();
        assert_eq!(cands.len(), 3);
    }
}
