use rand::seq::index::sample;
use rand::Rng;
use std::sync::Arc;

use crate::error::ENNError;
use crate::fit_samples::ScaleSearch;
use crate::model::EpistemicNearestNeighbors;
use crate::params::ENNParams;

pub struct ENNFitter {
    pub(crate) k: i32,
    pub(crate) params: Option<ENNParams>,
}

impl ENNFitter {
    pub fn new(k: i32) -> Self {
        Self { k, params: None }
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
        let calibrator = if affine_calibrate {
            Some(Arc::new(crate::surrogate_affine::fit_calibrator(
                model,
                &best,
                num_fit_samples,
                rng,
                false,
            )?))
        } else {
            None
        };
        let best = best.with_calibrator(calibrator);
        self.params = Some(best.clone());
        Ok(best)
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
        if model.num_obs() < 2 {
            let best = ENNParams::new(self.k, 1.0, 0.0).map_err(|e| {
                ENNError::InvalidParameter(format!("Failed to create default params: {e}"))
            })?;
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
    use ndarray::array;
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
