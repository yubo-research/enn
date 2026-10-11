use ndarray::{Array2, Array3, ArrayView1, ArrayView2};

use crate::error::{ENNError, EPS_VAR};

#[derive(Debug, Clone, PartialEq)]
pub struct PosteriorMoments {
    pub w_normalized: Array3<f64>,
    pub l2: Array2<f64>,
    pub mu: Array2<f64>,
    pub se: Array2<f64>,
    pub se_epi: Array2<f64>,
    pub se_ale: Array2<f64>,
}

pub fn noise_field_variance(epistemic_term: f64, aleatoric_scale: f64, scaled_yvar: f64) -> f64 {
    EPS_VAR + epistemic_term + aleatoric_scale + scaled_yvar
}

pub fn se_from_variance_components(
    epistemic_var: f64,
    aleatoric_var: f64,
    y_scale: f64,
) -> (f64, f64, f64) {
    let sum = epistemic_var + aleatoric_var;
    if sum < EPS_VAR {
        let se = EPS_VAR.sqrt() * y_scale;
        (se, se, 0.0)
    } else {
        let se = sum.sqrt() * y_scale;
        let se_epi = epistemic_var.sqrt() * y_scale;
        let se_ale = aleatoric_var.sqrt() * y_scale;
        (se, se_epi, se_ale)
    }
}

pub fn posterior_from_neighbors(
    dist2s: &ArrayView2<f64>,
    y_neighbors: &ArrayView2<f64>,
    yvar_neighbors: Option<ArrayView2<f64>>,
    epistemic_variance_scale: f64,
    aleatoric_variance_scale: f64,
    observation_noise: bool,
    y_scale: &ArrayView1<f64>,
) -> Result<PosteriorMoments, ENNError> {
    let n_query = dist2s.nrows();
    let k = dist2s.ncols();
    let num_metrics = y_scale.len();
    check_neighbor_shapes(y_neighbors, yvar_neighbors, n_query, k, num_metrics)?;
    if n_query == 0 || k == 0 {
        return Ok(empty_moments(n_query, k, num_metrics));
    }
    let y_scale_sq: Vec<f64> = y_scale.iter().map(|v| v * v).collect();
    let weights = noise_field_weights(
        dist2s,
        yvar_neighbors,
        epistemic_variance_scale,
        aleatoric_variance_scale,
        &y_scale_sq,
    );
    Ok(moments_from_weights(
        &weights,
        y_neighbors,
        yvar_neighbors,
        n_query,
        k,
        observation_noise,
        aleatoric_variance_scale,
        y_scale,
        &y_scale_sq,
    ))
}

fn check_neighbor_shapes(
    y_neighbors: &ArrayView2<f64>,
    yvar_neighbors: Option<ArrayView2<f64>>,
    n_query: usize,
    k: usize,
    num_metrics: usize,
) -> Result<(), ENNError> {
    if y_neighbors.ncols() != num_metrics || (k > 0 && y_neighbors.nrows() != n_query * k) {
        return Err(ENNError::InvalidShape {
            expected: vec![n_query * k, num_metrics],
            got: y_neighbors.shape().to_vec(),
        });
    }
    if let Some(yv) = yvar_neighbors.as_ref() {
        if yv.shape() != y_neighbors.shape() {
            return Err(ENNError::InvalidShape {
                expected: y_neighbors.shape().to_vec(),
                got: yv.shape().to_vec(),
            });
        }
    }
    Ok(())
}

fn empty_moments(n_query: usize, k: usize, num_metrics: usize) -> PosteriorMoments {
    PosteriorMoments {
        w_normalized: Array3::zeros((n_query, k, num_metrics)),
        l2: Array2::ones((n_query, num_metrics)),
        mu: Array2::zeros((n_query, num_metrics)),
        se: Array2::ones((n_query, num_metrics)),
        se_epi: Array2::ones((n_query, num_metrics)),
        se_ale: Array2::zeros((n_query, num_metrics)),
    }
}

fn noise_field_weights(
    dist2s: &ArrayView2<f64>,
    yvar_neighbors: Option<ArrayView2<f64>>,
    epistemic_variance_scale: f64,
    aleatoric_variance_scale: f64,
    y_scale_sq: &[f64],
) -> Array2<f64> {
    let n_query = dist2s.nrows();
    let k = dist2s.ncols();
    let num_metrics = y_scale_sq.len();
    let mut w = Array2::zeros((n_query * k, num_metrics));
    for i in 0..n_query {
        for j in 0..k {
            let epistemic_term = epistemic_variance_scale * dist2s[[i, j]];
            let idx = i * k + j;
            for m in 0..num_metrics {
                let scaled_yvar = match yvar_neighbors {
                    Some(yv) => yv[[idx, m]] / y_scale_sq[m],
                    None => 0.0,
                };
                let var_total =
                    noise_field_variance(epistemic_term, aleatoric_variance_scale, scaled_yvar);
                w[[idx, m]] = 1.0 / var_total;
            }
        }
    }
    w
}

#[allow(clippy::too_many_arguments)]
fn moments_from_weights(
    w: &Array2<f64>,
    y_neighbors: &ArrayView2<f64>,
    yvar_neighbors: Option<ArrayView2<f64>>,
    n_query: usize,
    k: usize,
    observation_noise: bool,
    aleatoric_scale: f64,
    y_scale: &ArrayView1<f64>,
    y_scale_sq: &[f64],
) -> PosteriorMoments {
    let num_metrics = y_scale.len();
    let mut out = empty_moments(n_query, k, num_metrics);
    out.l2.fill(0.0);
    out.se.fill(0.0);
    out.se_epi.fill(0.0);
    for i in 0..n_query {
        for m in 0..num_metrics {
            let base_idx = i * k;
            let norm: f64 = (0..k).map(|j| w[[base_idx + j, m]]).sum();
            let inv_norm = 1.0 / norm;
            let mut l2_sq = 0.0;
            let mut mu_val = 0.0;
            for j in 0..k {
                let w_norm = w[[base_idx + j, m]] * inv_norm;
                out.w_normalized[[i, j, m]] = w_norm;
                l2_sq += w_norm * w_norm;
                mu_val += w_norm * y_neighbors[[base_idx + j, m]];
            }
            out.l2[[i, m]] = l2_sq.sqrt();
            out.mu[[i, m]] = mu_val;
            let aleatoric_var = if observation_noise {
                aleatoric_from_weights(
                    &out.w_normalized,
                    yvar_neighbors,
                    i,
                    m,
                    k,
                    aleatoric_scale,
                    y_scale_sq[m],
                )
            } else {
                0.0
            };
            let (se_val, se_epi_val, se_ale_val) =
                se_from_variance_components(inv_norm, aleatoric_var, y_scale[m]);
            out.se[[i, m]] = se_val;
            out.se_epi[[i, m]] = se_epi_val;
            out.se_ale[[i, m]] = se_ale_val;
        }
    }
    out
}

fn aleatoric_from_weights(
    w_normalized: &Array3<f64>,
    yvar_neighbors: Option<ArrayView2<f64>>,
    row: usize,
    metric: usize,
    k: usize,
    aleatoric_scale: f64,
    y_scale_sq: f64,
) -> f64 {
    let mut sum = 0.0;
    for j in 0..k {
        let var_y = match yvar_neighbors {
            Some(yv) => yv[[row * k + j, metric]] / y_scale_sq,
            None => 0.0,
        };
        sum += w_normalized[[row, j, metric]] * (aleatoric_scale + var_y);
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn noise_field_grows_with_squared_distance() {
        let near = noise_field_variance(0.0, 0.1, 0.0);
        let far = noise_field_variance(2.0, 0.1, 0.0);
        assert!(far > near);
        assert!(1.0 / far < 1.0 / near);
    }

    #[test]
    fn one_neighbor_mean_matches_target() {
        let dist2s = array![[0.0]];
        let y_neighbors = array![[4.0]];
        let y_scale = array![2.0];
        let moments = posterior_from_neighbors(
            &dist2s.view(),
            &y_neighbors.view(),
            None,
            1.0,
            0.0,
            false,
            &y_scale.view(),
        )
        .unwrap();
        assert!((moments.mu[[0, 0]] - 4.0).abs() < 1e-12);
        let expected_se = EPS_VAR.sqrt() * 2.0;
        assert!((moments.se[[0, 0]] - expected_se).abs() < 1e-12);
    }
}
