//! Arm selection from a candidate set.

use ndarray::{Array2, ArrayView2};
use rand::seq::SliceRandom;
use rand::RngCore;

use crate::acquisition::{ParetoAcquisition, RandomAcquisition, UCBAcquisition};
use crate::config::AcquisitionConfig;
use crate::error::ENNError;
use crate::optimizer::Optimizer;
use crate::util::argmax_random_tie;

/// Select arms randomly.
pub(super) fn select_with_random(
    x_cand: &ArrayView2<f64>,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    let random_acq = RandomAcquisition;
    let indices = random_acq
        .select(x_cand.nrows(), num_arms, rng)
        .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
    Ok(select_by_indices(x_cand, &indices))
}

/// Select arms via Thompson sampling (posterior draw).
pub(super) fn select_with_thompson(
    optimizer: &Optimizer,
    surrogate: &(dyn crate::surrogate::Surrogate + Send + Sync),
    x_cand: &ArrayView2<f64>,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    let samples = surrogate.sample(x_cand, num_arms, rng)?;
    let n_candidates = x_cand.nrows();
    if optimizer.trust_region().is_morbo() {
        let num_metrics = samples.shape()[2];
        let mut flat = ndarray::Array2::zeros((num_arms * n_candidates, num_metrics));
        for arm in 0..num_arms {
            for cand in 0..n_candidates {
                for m in 0..num_metrics {
                    flat[[arm * n_candidates + cand, m]] = samples[[arm, cand, m]];
                }
            }
        }

        let flat = surrogate.naturalize_observations_y(flat);
        let flat_scores = optimizer
            .trust_region()
            .morbo_scalarize(&flat.view(), false)
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
        let mut all_scores = ndarray::Array2::zeros((num_arms, n_candidates));
        for arm in 0..num_arms {
            for cand in 0..n_candidates {
                all_scores[[arm, cand]] = flat_scores[arm * n_candidates + cand];
            }
        }
        let mut indices = Vec::with_capacity(num_arms);
        for arm in 0..num_arms {
            let mut arm_scores = vec![f64::NEG_INFINITY; n_candidates];
            for cand in 0..n_candidates {
                arm_scores[cand] = all_scores[[arm, cand]];
            }
            for &prev in &indices {
                arm_scores[prev] = f64::NEG_INFINITY;
            }
            indices.push(argmax_random_tie(&arm_scores, rng));
        }
        return Ok(select_by_indices(x_cand, &indices));
    }


    let mut flat = ndarray::Array2::zeros((num_arms * n_candidates, 1));
    for arm in 0..num_arms {
        for i in 0..n_candidates {
            flat[[arm * n_candidates + i, 0]] = samples[[arm, i, 0]];
        }
    }
    let flat = surrogate.naturalize_observations_y(flat);
    let mut indices = Vec::with_capacity(num_arms);
    for arm in 0..num_arms {
        let arm_scores: Vec<f64> = (0..n_candidates)
            .map(|i| flat[[arm * n_candidates + i, 0]])
            .collect();
        indices.push(argmax_random_tie(&arm_scores, rng));
    }
    Ok(select_by_indices(x_cand, &indices))
}

/// Select arms via UCB (upper confidence bound).
pub(super) fn select_with_ucb(
    optimizer: &Optimizer,
    surrogate: &(dyn crate::surrogate::Surrogate + Send + Sync),
    x_cand: &ArrayView2<f64>,
    num_arms: usize,
    beta: f64,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    let pred = surrogate.naturalize_prediction(surrogate.predict(x_cand)?);
    if optimizer.trust_region().is_morbo() {

        let ucb_vals = &pred.mu + &(pred.se * beta);
        let scores = optimizer
            .trust_region()
            .morbo_scalarize(&ucb_vals.view(), false)
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
        let mut indices: Vec<usize> = (0..scores.len()).collect();
        indices.shuffle(rng);
        indices.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
        let selected: Vec<usize> = indices.into_iter().take(num_arms).collect();
        return Ok(select_by_indices(x_cand, &selected));
    }
    let mu = pred.mu.column(0);
    let sigma = pred.se.column(0);
    let ucb = UCBAcquisition::new(beta);
    let indices = ucb
        .select(&mu, &sigma, num_arms, rng)
        .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
    Ok(select_by_indices(x_cand, &indices))
}

/// Select arms via Pareto frontier.
pub(super) fn select_with_pareto(
    surrogate: &(dyn crate::surrogate::Surrogate + Send + Sync),
    x_cand: &ArrayView2<f64>,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {

    let pred = surrogate.naturalize_prediction(surrogate.predict(x_cand)?);
    let pareto = ParetoAcquisition::new();
    let indices = pareto
        .select(&pred.mu.view(), &pred.se.view(), num_arms, rng)
        .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
    Ok(select_by_indices(x_cand, &indices))
}

/// Select arms using acquisition function.
pub(super) fn select_arms(
    optimizer: &Optimizer,
    x_cand: &ArrayView2<f64>,
    num_arms: usize,
    rng: &mut dyn RngCore,
) -> Result<Array2<f64>, ENNError> {
    let config = optimizer.config().acquisition;

    match config {
        AcquisitionConfig::Random => select_with_random(x_cand, num_arms, rng),
        AcquisitionConfig::Thompson => match optimizer.surrogate() {
            Some(s) => select_with_thompson(optimizer, s, x_cand, num_arms, rng),
            None => select_with_random(x_cand, num_arms, rng),
        },
        AcquisitionConfig::UCB { beta } => match optimizer.surrogate() {
            Some(s) => select_with_ucb(optimizer, s, x_cand, num_arms, beta, rng),
            None => select_with_random(x_cand, num_arms, rng),
        },
        AcquisitionConfig::Pareto => match optimizer.surrogate() {
            Some(s) => select_with_pareto(s, x_cand, num_arms, rng),
            None => select_with_random(x_cand, num_arms, rng),
        },
    }
}

/// Select rows by indices.
pub(super) fn select_by_indices(x: &ArrayView2<f64>, indices: &[usize]) -> Array2<f64> {
    use ndarray::Axis;
    let rows: Vec<_> = indices.iter().map(|&i| x.row(i).to_owned()).collect();
    ndarray::stack(Axis(0), &rows.iter().map(|r| r.view()).collect::<Vec<_>>())
        .expect("stack should succeed for same-shaped rows")
}
