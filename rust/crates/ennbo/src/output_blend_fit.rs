use ndarray::{Array1, Array2, Axis, Ix2};
use rand::seq::index::sample;
use rand::Rng;
use std::sync::Arc;

use crate::error::{invalid, ENNError};
use crate::model::{EpistemicNearestNeighbors, OutputMoments};
use crate::online_output_weights::{OutputWeights, PRIOR_ALPHA};
use crate::output_blend::{OutputBlend, SemanticOutputs};
use crate::params::{ENNParams, PosteriorFlags};

const MAX_ROWS_PER_ASK: usize = 1000;
const MIN_LEARNED_ROWS: usize = 12;

#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub(crate) struct OutputBlendFit {
    weights: OutputWeights,
    cursor: usize,
    #[serde(with = "crate::exact")]
    warm: Option<Arc<OutputBlend>>,
}

impl OutputBlendFit {
    pub(crate) fn new(outputs: &SemanticOutputs) -> Self {
        Self {
            weights: OutputWeights::new(outputs.loadings.clone()),
            cursor: 0,
            warm: None,
        }
    }

    pub(crate) fn fit<R: Rng>(
        &mut self,
        model: &EpistemicNearestNeighbors,
        outputs: &SemanticOutputs,
        k: i32,
        warm: Option<&Arc<OutputBlend>>,
        rng: &mut R,
    ) -> Result<Arc<OutputBlend>, ENNError> {
        if self.weights.loadings() != outputs.loadings {
            return Err(invalid("fitter was used with different output loadings"));
        }
        if warm.is_some_and(|w| w.num_outputs() != model.num_metrics()) {
            return Err(invalid(format!(
                "warm output blend must have {} outputs",
                model.num_metrics()
            )));
        }
        if warm.is_some() {
            self.warm = warm.cloned();
        }
        let moments = model.output_moments();
        self.observe_new_rows(model, k, &moments, rng)?;
        self.choose(outputs, moments)
    }

    fn observe_new_rows<R: Rng>(
        &mut self,
        model: &EpistemicNearestNeighbors,
        k: i32,
        moments: &OutputMoments,
        rng: &mut R,
    ) -> Result<(), ENNError> {
        let len = model.len();
        if self.cursor > len {
            return Err(invalid("fitter has seen more rows than the model holds"));
        }
        if len < 2 || self.cursor == len {
            return Ok(());
        }
        let indices = capped_rows(self.cursor, len, rng);
        self.cursor = len;
        let (x, y, _) = model.rows().train_rows_at(&indices)?;
        let neighbors = (k as usize).min(len - 1) as i32;
        let params = ENNParams::new(neighbors, 1.0, 0.0).map_err(invalid)?;
        let flags = PosteriorFlags::new().with_exclude_nearest(true);
        let post = model.posterior_warped(&x.view(), &params, &flags)?;
        let mean = post.mu.into_dimensionality::<Ix2>()?;
        let predicted = moments.standardized(&mean);
        let target = moments.standardized(&y);
        for (a, t) in predicted.axis_iter(Axis(0)).zip(target.axis_iter(Axis(0))) {
            self.weights.observe(a, t);
        }
        Ok(())
    }

    fn choose(
        &self,
        outputs: &SemanticOutputs,
        moments: OutputMoments,
    ) -> Result<Arc<OutputBlend>, ENNError> {
        let (alpha, projector) = match (outputs.alpha, &self.warm) {
            (Some(pinned), _) => self.prior_with_alpha(pinned),
            _ if self.weights.rows() >= MIN_LEARNED_ROWS => {
                (self.weights.alpha(), self.weights.weighted_projector())
            }
            (None, Some(warm)) => return Ok(warm.clone()),
            (None, None) => self.prior_with_alpha(PRIOR_ALPHA),
        };
        moments.into_blend(alpha, projector).map(Arc::new)
    }

    fn prior_with_alpha(&self, alpha: f64) -> (Array1<f64>, Array2<f64>) {
        let projector = self.weights.projector().clone();
        (Array1::from_elem(projector.nrows(), alpha), projector)
    }
}

fn capped_rows<R: Rng>(start: usize, end: usize, rng: &mut R) -> Vec<usize> {
    let backlog = end - start;
    if backlog <= MAX_ROWS_PER_ASK {
        return (start..end).collect();
    }
    sample(rng, backlog, MAX_ROWS_PER_ASK)
        .into_iter()
        .map(|i| start + i)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn group_loadings() -> Array2<f64> {
        Array2::from_elem((2, 1), 0.5f64.sqrt())
    }

    fn pinned(alpha: Option<f64>) -> SemanticOutputs {
        SemanticOutputs {
            loadings: group_loadings(),
            alpha,
        }
    }

    fn rows_model(n: usize, sign: f64) -> crate::test_helpers::TestModel {
        let x = Array2::from_shape_fn((n, 1), |(i, _)| i as f64 * 0.1);
        let y = Array2::from_shape_fn((n, 2), |(i, j)| {
            let t = (i as f64 * 0.1).sin();
            if j == 0 {
                t
            } else {
                sign * t + 0.01 * ((i * 7) % 5) as f64
            }
        });
        crate::test_helpers::test_model(x, y, None, false)
    }

    fn run(
        fit: &mut OutputBlendFit,
        model: &EpistemicNearestNeighbors,
        outputs: &SemanticOutputs,
        warm: Option<&Arc<OutputBlend>>,
    ) -> Result<Arc<OutputBlend>, ENNError> {
        fit.fit(model, outputs, 3, warm, &mut StdRng::seed_from_u64(0))
    }

    #[test]
    fn capped_rows_stay_in_range_and_are_unique() {
        let mut rng = StdRng::seed_from_u64(0);
        assert_eq!(capped_rows(3, 6, &mut rng), vec![3, 4, 5]);
        let mut rows = capped_rows(5, 5005, &mut rng);
        assert_eq!(rows.len(), MAX_ROWS_PER_ASK);
        rows.sort_unstable();
        rows.dedup();
        assert_eq!(rows.len(), MAX_ROWS_PER_ASK);
        assert!(rows[0] >= 5 && rows[rows.len() - 1] < 5005);
    }

    #[test]
    fn prior_is_used_until_twelve_rows_then_learning_takes_over() {
        let outputs = pinned(None);
        let mut fit = OutputBlendFit::new(&outputs);
        let blend = run(&mut fit, &rows_model(11, 1.0), &outputs, None).unwrap();
        assert_eq!(blend.alpha, array![PRIOR_ALPHA, PRIOR_ALPHA]);
        let blend = run(&mut fit, &rows_model(40, 1.0), &outputs, None).unwrap();
        assert_ne!(blend.alpha, array![PRIOR_ALPHA, PRIOR_ALPHA]);
        assert_eq!(fit.weights.rows(), 39 + 1);
    }

    #[test]
    fn pinned_alpha_wins_over_learning_and_warm_blend() {
        let outputs = pinned(Some(0.1));
        let mut fit = OutputBlendFit::new(&outputs);
        let warm = run(
            &mut OutputBlendFit::new(&pinned(None)),
            &rows_model(4, 1.0),
            &pinned(None),
            None,
        )
        .unwrap();
        let blend = run(&mut fit, &rows_model(40, 1.0), &outputs, Some(&warm)).unwrap();
        assert_eq!(blend.alpha, array![0.1, 0.1]);
        assert_eq!(blend.projector, group_loadings().dot(&group_loadings().t()));
    }

    #[test]
    fn warm_blend_is_kept_before_twelve_rows_and_shape_checked() {
        let outputs = pinned(None);
        let warm = Arc::new(
            OutputBlend::new(
                array![0.7, 0.2],
                Array2::eye(2),
                array![0.0, 0.0],
                array![1.0, 1.0],
            )
            .unwrap(),
        );
        let model = rows_model(5, 1.0);
        let blend = run(
            &mut OutputBlendFit::new(&outputs),
            &model,
            &outputs,
            Some(&warm),
        )
        .unwrap();
        assert!(Arc::ptr_eq(&blend, &warm));
        let wrong = Arc::new(
            OutputBlend::new(array![0.1], Array2::eye(1), array![0.0], array![1.0]).unwrap(),
        );
        let err = run(
            &mut OutputBlendFit::new(&outputs),
            &model,
            &outputs,
            Some(&wrong),
        );
        assert!(err.unwrap_err().to_string().contains("warm output blend"));
    }

    #[test]
    fn one_row_is_not_consumed_until_a_second_arrives() {
        let outputs = pinned(None);
        let mut fit = OutputBlendFit::new(&outputs);
        run(&mut fit, &rows_model(1, 1.0), &outputs, None).unwrap();
        assert_eq!((fit.cursor, fit.weights.rows()), (0, 0));
        run(&mut fit, &rows_model(3, 1.0), &outputs, None).unwrap();
        assert_eq!((fit.cursor, fit.weights.rows()), (3, 3));
    }

    #[test]
    fn fitter_rejects_a_smaller_model_or_other_loadings() {
        let outputs = pinned(None);
        let mut fit = OutputBlendFit::new(&outputs);
        run(&mut fit, &rows_model(5, 1.0), &outputs, None).unwrap();
        assert!(run(&mut fit, &rows_model(3, 1.0), &outputs, None).is_err());
        let other = SemanticOutputs {
            loadings: array![[1.0], [0.0]],
            alpha: None,
        };
        assert!(run(&mut fit, &rows_model(5, 1.0), &other, None).is_err());
    }
}
