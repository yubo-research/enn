use ndarray::{Array1, Array2, ArrayView2};

use crate::error::ENNError;
use crate::posterior_formula::posterior_from_neighbors;

use super::search::{rank_all, Candidate};
use super::{EnnTs, QueryOptions};

pub struct PosteriorArrays {
    pub mu: Array2<f64>,
    pub se: Array2<f64>,
    pub se_epi: Array2<f64>,
    pub se_ale: Array2<f64>,
}

pub(super) fn posterior(
    model: &EnnTs,
    queries: ArrayView2<f64>,
    cutoff: i64,
    k: usize,
    observation_noise: bool,
) -> Result<PosteriorArrays, ENNError> {
    let options = QueryOptions {
        cutoff,
        k,
        require_target: true,
    };
    let ranked = rank_all(model, queries, &options, false)?;
    let scale = y_scale_vec(model);
    let metrics = model.metrics.max(scale.len());
    let mut out = prior_arrays(ranked.len(), metrics, &scale);
    for (i, row) in ranked.iter().enumerate() {
        if row.is_empty() {
            continue;
        }
        let moments = row_moments(row, &scale, observation_noise, model)?;
        out.mu.row_mut(i).assign(&moments.mu.row(0));
        out.se.row_mut(i).assign(&moments.se.row(0));
        out.se_epi.row_mut(i).assign(&moments.se_epi.row(0));
        out.se_ale.row_mut(i).assign(&moments.se_ale.row(0));
    }
    Ok(out)
}

fn y_scale_vec(model: &EnnTs) -> Vec<f64> {
    if model.metrics == 0 || model.n_y < 2 {
        return vec![1.0; model.metrics];
    }
    let n = model.n_y as f64;
    (0..model.metrics)
        .map(|m| {
            let mean = model.y_sum[m] / n;
            let var = (model.y_sumsq[m] / n - mean * mean).max(0.0);
            let std = var.sqrt();
            if std.is_finite() && std > 0.0 {
                std
            } else {
                1.0
            }
        })
        .collect()
}

fn prior_arrays(n: usize, metrics: usize, scale: &[f64]) -> PosteriorArrays {
    let mut se = Array2::ones((n, metrics));
    for m in 0..metrics.min(scale.len()) {
        for i in 0..n {
            se[[i, m]] = scale[m];
        }
    }
    PosteriorArrays {
        mu: Array2::zeros((n, metrics)),
        se: se.clone(),
        se_epi: se,
        se_ale: Array2::zeros((n, metrics)),
    }
}

fn row_moments(
    row: &[Candidate],
    scale: &[f64],
    observation_noise: bool,
    model: &EnnTs,
) -> Result<crate::posterior_formula::PosteriorMoments, ENNError> {
    let k = row.len();
    let metrics = scale.len();
    let mut dist2 = Array2::zeros((1, k));
    let mut y = Array2::zeros((k, metrics));
    for (j, cand) in row.iter().enumerate() {
        dist2[[0, j]] = cand.dist2;
        for m in 0..metrics.min(cand.target.len()) {
            y[[j, m]] = cand.target[m];
        }
    }
    let y_scale = Array1::from_vec(scale.to_vec());
    posterior_from_neighbors(
        &dist2.view(),
        &y.view(),
        None,
        model.config.epistemic_variance_scale,
        model.config.aleatoric_variance_scale,
        observation_noise,
        &y_scale.view(),
    )
}
