use crate::adaptive_basis::Metadata;
use crate::adaptive_geometry::invalid;
use crate::adaptive_predictor::{Choice, Epoch, Predictor};
use crate::adaptive_selection::{allowed_output, geometries};
use crate::error::ENNError;
use crate::fit::compute_single_loglik;
use crate::fit_samples::ScaleSearch;
use crate::fitter::ENNFitter;
use crate::params::{ENNParams, PosteriorFlags};
use ndarray::{Array1, Array2, Axis, Ix2};
use rand::seq::index::sample;
use rand::Rng;

pub(crate) struct AdaptiveInputs<'a> {
    pub x: &'a Array2<f64>,
    pub y: &'a Array2<f64>,
    pub metadata: &'a Metadata,
    pub trend: bool,
    pub adaptation: &'a str,
    pub weight: f64,
    pub column_scale: Option<&'a Array1<f64>>,
}

const MIN_VARIANCE: f64 = 1e-12;

fn score(
    predictor: &Predictor,
    choice: &Choice,
    x: &Array2<f64>,
    y: &Array2<f64>,
    trend: bool,
) -> Result<f64, ENNError> {
    let basis = &predictor.basis;
    let features = basis.features(x, &choice.geometry, choice.weight)?;
    let flags = PosteriorFlags::new()
        .with_exclude_nearest(true)
        .with_observation_noise(true);
    let post = predictor
        .model
        .posterior(&features.view(), &choice.params, &flags)?;
    let mut mean = post.mu.into_dimensionality::<Ix2>().map_err(invalid)?;
    let se = post.se.into_dimensionality::<Ix2>().map_err(invalid)?;
    if trend {
        mean += &basis.trend(x);
    }
    let projector = &basis
        .operators
        .iter()
        .find(|(name, _)| *name == choice.output)
        .ok_or_else(|| invalid("unknown output operator"))?
        .1;
    let operator =
        Array2::<f64>::eye(mean.ncols()) * (1.0 - choice.alpha) + projector * choice.alpha;
    let mut mu = mean.dot(&operator);
    let mut se = se.dot(&operator.mapv(f64::abs));
    let mut observed = (y - &basis.y_mean) / &basis.y_scale;
    for (j, constant) in basis.constant.iter().enumerate() {
        if *constant {
            mu.column_mut(j).fill(0.0);
            observed.column_mut(j).fill(0.0);
            se.column_mut(j).fill(1.0);
        }
    }
    se.mapv_inplace(|v| v.max(MIN_VARIANCE.sqrt()));
    Ok(compute_single_loglik(
        &observed.view(),
        &mu.view(),
        &se.view(),
    ))
}

impl ENNFitter {
    fn propose<R: Rng>(
        &self,
        inputs: &AdaptiveInputs,
        outputs: &[&'static str],
        search: &ScaleSearch,
        rng: &mut R,
    ) -> Result<Vec<Choice>, ENNError> {
        let menu = geometries(
            inputs.metadata,
            inputs.adaptation,
            Some(inputs.weight),
            inputs.column_scale.is_some(),
        );
        let paramss = self.build_random_param_candidates(
            search.num_fit_candidates.get(),
            search.infer_aleatoric_variance,
            rng,
        )?;
        Ok(paramss
            .into_iter()
            .map(|params| {
                let (geometry, weight) = menu[rng.gen_range(0..menu.len())];
                let output = outputs[rng.gen_range(0..outputs.len())];
                let alpha = match output {
                    "independent" => 0.0,
                    "lowrank" => 1.0,
                    _ => rng.gen::<f64>(),
                };
                Choice {
                    geometry: geometry.into(),
                    weight,
                    output: output.into(),
                    params,
                    alpha,
                }
            })
            .collect())
    }

    fn refreshed_incumbent(&self, inputs: &AdaptiveInputs, search: &ScaleSearch) -> Option<Choice> {
        let incumbent = self.incumbent.as_ref()?;
        let menu = geometries(
            inputs.metadata,
            inputs.adaptation,
            Some(inputs.weight),
            inputs.column_scale.is_some(),
        );
        let (_, weight) = menu.iter().find(|(name, _)| *name == incumbent.geometry)?;
        let mut choice = incumbent.clone();
        choice.weight = *weight;
        if !search.infer_aleatoric_variance {
            choice.params = ENNParams::new(
                choice.params.k_num_neighbors,
                choice.params.epistemic_variance_scale,
                0.0,
            )
            .ok()?;
        }
        Some(choice)
    }

    pub(crate) fn ask_adaptive<R: Rng>(
        &mut self,
        inputs: &AdaptiveInputs,
        search: &ScaleSearch,
        rng: &mut R,
    ) -> Result<Epoch, ENNError> {
        let n = inputs.x.nrows();
        let seed = Choice {
            geometry: "fixed".into(),
            weight: 0.0,
            output: "independent".into(),
            params: ENNParams::new(self.k.min(n as i32), 1.0, 0.0).map_err(invalid)?,
            alpha: 0.0,
        };
        let mut predictors = vec![(
            "fixed".to_string(),
            Predictor::new(
                inputs.x,
                inputs.y,
                inputs.metadata,
                seed,
                inputs.trend,
                inputs.column_scale,
            )?,
        )];
        let outputs: Vec<&'static str> = predictors[0]
            .1
            .basis
            .operators
            .iter()
            .map(|(name, _)| *name)
            .filter(|name| {
                allowed_output(name, inputs.adaptation, inputs.metadata.output.is_some())
            })
            .collect();
        let mut candidates: Vec<Choice> = self
            .refreshed_incumbent(inputs, search)
            .into_iter()
            .collect();
        candidates.extend(self.propose(inputs, &outputs, search, rng)?);
        let count = search.num_fit_samples.get().min(n);
        let rows: Vec<usize> = if count == n {
            (0..n).collect()
        } else {
            sample(rng, n, count).into_iter().collect()
        };
        let sub_x = inputs.x.select(Axis(0), &rows);
        let sub_y = inputs.y.select(Axis(0), &rows);
        let mut best: Option<(f64, Choice)> = None;
        for choice in candidates {
            if !predictors.iter().any(|(g, _)| *g == choice.geometry) {
                let predictor = Predictor::new(
                    inputs.x,
                    inputs.y,
                    inputs.metadata,
                    choice.clone(),
                    inputs.trend,
                    inputs.column_scale,
                )?;
                predictors.push((choice.geometry.clone(), predictor));
            }
            let predictor = &predictors
                .iter()
                .find(|(g, _)| *g == choice.geometry)
                .unwrap()
                .1;
            let value = score(predictor, &choice, &sub_x, &sub_y, inputs.trend)?;
            if best.as_ref().is_none_or(|(top, _)| value > *top) {
                best = Some((value, choice));
            }
        }
        let (_, winner) = best.ok_or_else(|| invalid("adaptive fit produced no candidates"))?;
        let index = predictors
            .iter()
            .position(|(g, _)| *g == winner.geometry)
            .ok_or_else(|| invalid("winning geometry has no predictor"))?;
        let mut predictor = predictors.swap_remove(index).1;
        predictor.choice = winner.clone();
        let floor = predictor.basis.y_scale.mapv(|v| (v * 0.03).max(1e-12));
        self.params = Some(winner.params);
        self.incumbent = Some(winner);
        Ok(Epoch {
            predictor,
            blend: None,
            floor,
        })
    }
}
