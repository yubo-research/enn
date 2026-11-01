use ndarray::{Array2, ArrayView1, ArrayView2};
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

pub struct ENNFitter {
    pub(crate) k: i32,
    pub(crate) params: Option<ENNParams>,
    calibrator: Option<AffineCalibrator>,
    output_fit: Option<OutputBlendFit>,
}

impl ENNFitter {
    pub fn new(k: i32) -> Self {
        Self {
            k,
            params: None,
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
        let logliks = crate::fit::subsample_loglik(
            model,
            &train_x.view(),
            &train_y.view(),
            &paramss,
            num_fit_samples,
            rng,
            Some(&model.y_scale().view()),
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
    use ndarray::{array, Axis};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn ask_uses_explicit_warm_start() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let train_y = array![[0.0], [1.0], [1.0], [2.0]];
        let model = crate::test_helpers::test_model(train_x, train_y, None, false);
        let warm = ENNParams::new(2, 2.5, 0.3).unwrap();
        let search = crate::fit_samples::test_search(2, 1, true);
        let ask = |seed: u64, warm: Option<&ENNParams>| {
            ENNFitter::new(2)
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

    #[test]
    fn yy_model_returns_a_blend_on_every_candidate_path() {
        let model = yy_model(Some(0.3));
        let mut fitter = ENNFitter::new(3);
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

    fn grow(model: &mut EpistemicNearestNeighbors) {
        let x = array![[10.0], [11.0], [12.0]];
        let y = array![[5.0, 7.0], [6.0, 8.0], [7.0, 9.0]];
        model.add(&x.view(), &y.view(), None).unwrap();
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
        let mut fitter = ENNFitter::new(3);
        let search = crate::fit_samples::test_search(5, 3, true);
        let mut rng = StdRng::seed_from_u64(0);
        let first = fitter.ask(&model, &search, None, &mut rng).unwrap();
        grow(&mut model);
        let second = fitter.ask(&model, &search, None, &mut rng).unwrap();
        let center = |p: &ENNParams| p.output_blend.as_ref().unwrap().center.clone();
        assert_ne!(center(&first), center(&second));
        let indices: Vec<usize> = (0..model.len()).collect();
        let (_, y, _) = model.train_rows_at(&indices).unwrap();
        let want = y.mean_axis(Axis(0)).unwrap();
        assert!((center(&second) - want).iter().all(|d| d.abs() < 1e-12));
    }

    #[test]
    fn warm_blend_given_once_is_kept_until_twelve_rows() {
        let mut model = short_yy_model();
        let mut fitter = ENNFitter::new(3);
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
        grow(&mut model);
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
            ENNFitter::new(3)
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
    fn kiss_fitter_random_params() {
        let fitter = ENNFitter::new(2);
        let mut rng = StdRng::seed_from_u64(0);
        let cands = fitter
            .build_random_param_candidates(3, true, &mut rng)
            .unwrap();
        assert_eq!(cands.len(), 3);
    }
}
