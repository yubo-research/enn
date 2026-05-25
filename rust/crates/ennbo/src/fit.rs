//! Parameter fitting for ENN models via subsample log-likelihood.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use rand::seq::index::sample;
use rand::Rng;

use crate::error::ENNError;
use crate::model::EpistemicNearestNeighbors;
use crate::params::{ENNParams, PosteriorFlags};
use crate::traits::PosteriorComputation;

/// Validates subsample log-likelihood inputs.
fn validate_subsample_inputs(
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    p: usize,
    num_params: usize,
) -> Result<(), ENNError> {
    if x.ndim() != 2 {
        return Err(ENNError::InvalidShape {
            expected: vec![x.nrows(), x.ncols()],
            got: x.shape().to_vec(),
        });
    }
    if y.ndim() != 2 {
        return Err(ENNError::InvalidShape {
            expected: vec![y.nrows(), y.ncols()],
            got: y.shape().to_vec(),
        });
    }
    if x.nrows() != y.nrows() {
        return Err(ENNError::InvalidParameter(format!(
            "x and y must have same number of rows: {} vs {}",
            x.nrows(),
            y.nrows()
        )));
    }
    if p == 0 {
        return Err(ENNError::InvalidParameter(
            "P (num_samples) must be > 0".to_string(),
        ));
    }
    if num_params == 0 {
        return Err(ENNError::InvalidParameter(
            "paramss must be non-empty".to_string(),
        ));
    }
    Ok(())
}

/// Compute single log-likelihood given scaled predictions.
fn compute_single_loglik(
    y_scaled: &ArrayView2<f64>,
    mu_i: &ArrayView2<f64>,
    se_i: &ArrayView2<f64>,
) -> f64 {
    // Check for non-finite values
    if !mu_i.iter().all(|v| v.is_finite()) || !se_i.iter().all(|v| v.is_finite()) {
        return 0.0;
    }
    if se_i.iter().any(|&v| v <= 0.0) {
        return 0.0;
    }

    let mut loglik = 0.0;
    for i in 0..y_scaled.nrows() {
        for j in 0..y_scaled.ncols() {
            let y_ij = y_scaled[[i, j]];
            let mu_ij = mu_i[[i, j]];
            let se_ij = se_i[[i, j]];
            let var_scaled = se_ij * se_ij;

            // Gaussian log-likelihood: -0.5 * (log(2*pi*var) + (y-mu)^2/var)
            loglik += -0.5 * (2.0 * std::f64::consts::PI * var_scaled).ln()
                - 0.5 * (y_ij - mu_ij).powi(2) / var_scaled;
        }
    }

    if loglik.is_finite() {
        loglik
    } else {
        0.0
    }
}

/// Compute subsample log-likelihood for multiple parameter sets.
///
/// # Arguments
/// * `model` - The ENN model
/// * `x` - Input features (n x d)
/// * `y` - Target values (n x m)
/// * `paramss` - Parameter sets to evaluate
/// * `p` - Number of subsamples
/// * `rng` - Random number generator
/// * `y_std` - Optional standardization factors for y
///
/// # Returns
/// Vector of log-likelihoods, one per parameter set
pub fn subsample_loglik<R: Rng>(
    model: &EpistemicNearestNeighbors,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    paramss: &[ENNParams],
    p: usize,
    rng: &mut R,
    y_std: Option<&ArrayView1<f64>>,
) -> Result<Vec<f64>, ENNError> {
    validate_subsample_inputs(x, y, p, paramss.len())?;

    let n = x.nrows();
    if n == 0 || model.num_obs() <= 1 {
        return Ok(vec![0.0; paramss.len()]);
    }

    // Check for non-finite y values
    if !y.iter().all(|v| v.is_finite()) {
        return Ok(vec![0.0; paramss.len()]);
    }

    let p_actual = p.min(n);

    let indices: Vec<usize> = if p_actual == n {
        (0..n).collect()
    } else {
        sample(rng, n, p_actual).into_iter().collect()
    };

    // Select subsampled data
    let num_metrics = y.ncols();
    let num_dim = x.ncols();
    let mut x_sel = Array2::zeros((p_actual, num_dim));
    let mut y_sel = Array2::zeros((p_actual, num_metrics));
    for (new_i, &old_i) in indices.iter().enumerate() {
        x_sel.row_mut(new_i).assign(&x.row(old_i));
        y_sel.row_mut(new_i).assign(&y.row(old_i));
    }

    // Compute posterior for all parameter sets at once
    let flags = PosteriorFlags::new()
        .with_exclude_nearest(true)
        .with_observation_noise(true);
    let post = model.batch_posterior(&x_sel.view(), paramss, &flags)?;

    let num_params = paramss.len();
    let num_outputs = y_sel.ncols();

    // Validate shapes
    let expected_shape = vec![num_params, p_actual, num_outputs];
    let mu_shape = post.mu.shape().to_vec();
    let se_shape = post.se.shape().to_vec();
    if mu_shape != expected_shape || se_shape != expected_shape {
        return Err(ENNError::InvalidShape {
            expected: expected_shape.clone(),
            got: mu_shape,
        });
    }

    // Compute y_std for standardization
    let y_std_computed: Array1<f64> = if let Some(ys) = y_std {
        ys.to_owned()
    } else {
        y.std_axis(Axis(0), 0.0)
    };
    let y_std_safe: Array1<f64> = y_std_computed
        .iter()
        .map(|&v| if v.is_finite() && v > 0.0 { v } else { 1.0 })
        .collect();

    // Scale data
    let mut y_scaled = Array2::zeros(y_sel.raw_dim());
    // mu_scaled and se_scaled are 3D: (num_params, p_actual, num_outputs)
    let mut mu_scaled = Array2::zeros((num_params * p_actual, num_outputs));
    let mut se_scaled = Array2::zeros((num_params * p_actual, num_outputs));

    for i in 0..p_actual {
        for j in 0..num_outputs {
            let std_j = y_std_safe[j];
            y_scaled[[i, j]] = y_sel[[i, j]] / std_j;
            for pi in 0..num_params {
                let idx = pi * p_actual + i;
                mu_scaled[[idx, j]] = post.mu[[pi, i, j]] / std_j;
                se_scaled[[idx, j]] = post.se[[pi, i, j]] / std_j;
            }
        }
    }

    // Compute log-likelihoods
    let mut logliks = Vec::with_capacity(num_params);
    for pi in 0..num_params {
        // Extract the pi-th parameter's predictions (p_actual x num_outputs)
        let start_idx = pi * p_actual;
        let mu_i = mu_scaled.slice(ndarray::s![start_idx..start_idx + p_actual, ..]);
        let se_i = se_scaled.slice(ndarray::s![start_idx..start_idx + p_actual, ..]);
        let ll = compute_single_loglik(&y_scaled.view(), &mu_i, &se_i);
        logliks.push(ll);
    }

    Ok(logliks)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fitter::ENNFitter;
    use crate::index::IndexDriver;
    use crate::test_helpers::test_epistemic_model_exact_unit_square as create_test_model;
    use ndarray::array;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_subsample_loglik_basic() {
        let model = create_test_model();
        let x = array![[0.5, 0.5], [0.2, 0.8]];
        let y = array![[1.0], [1.2]];
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let paramss = vec![params];
        let mut rng = StdRng::seed_from_u64(42);

        let logliks =
            subsample_loglik(&model, &x.view(), &y.view(), &paramss, 2, &mut rng, None).unwrap();

        assert_eq!(logliks.len(), 1);
        assert!(logliks[0].is_finite());
    }

    #[test]
    fn test_subsample_loglik_empty_model() {
        let train_x = array![[0.0, 0.0]];
        let train_y = array![[0.0]];
        let model =
            EpistemicNearestNeighbors::new(train_x, train_y, None, false, IndexDriver::Exact)
                .unwrap();

        let x = array![[0.5, 0.5]];
        let y = array![[1.0]];
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let paramss = vec![params];
        let mut rng = StdRng::seed_from_u64(42);

        let logliks =
            subsample_loglik(&model, &x.view(), &y.view(), &paramss, 2, &mut rng, None).unwrap();

        assert_eq!(logliks, vec![0.0]);
    }

    #[test]
    fn test_enn_fitter_ask_basic() {
        let model = create_test_model();
        let mut rng = StdRng::seed_from_u64(42);
        let mut fitter = ENNFitter::new(2, true);
        fitter.reset_y_stats(&model.train_y());

        let result = fitter.ask(&model, 5, 3, None, &mut rng).unwrap();

        assert_eq!(result.k_num_neighbors, 2);
        assert!(result.epistemic_variance_scale > 0.0);
        assert!(result.aleatoric_variance_scale >= 0.0);
    }

    #[test]
    fn test_enn_fitter_ask_with_warm_start() {
        let model = create_test_model();
        let mut rng = StdRng::seed_from_u64(42);
        let mut fitter = ENNFitter::new(2, true);
        fitter.reset_y_stats(&model.train_y());

        let warm_start = ENNParams::new(2, 1.5, 0.2).unwrap();

        let result = fitter
            .ask(&model, 5, 3, Some(&warm_start), &mut rng)
            .unwrap();

        assert_eq!(result.k_num_neighbors, 2);
        assert!(result.epistemic_variance_scale > 0.0);
    }

    #[test]
    fn test_enn_fitter_ask_disable_aleatoric() {
        let model = create_test_model();
        let mut rng = StdRng::seed_from_u64(42);
        let mut fitter = ENNFitter::new(2, false);
        fitter.reset_y_stats(&model.train_y());

        let result = fitter.ask(&model, 5, 3, None, &mut rng).unwrap();

        assert_eq!(result.k_num_neighbors, 2);
        assert!(result.epistemic_variance_scale > 0.0);
        assert_eq!(result.aleatoric_variance_scale, 0.0);
    }

    #[test]
    fn test_enn_fitter_ask_multioutput() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];
        let train_y = array![[0.0, 1.0], [1.0, 2.0], [1.0, 0.0], [2.0, 1.0], [1.0, 1.5]];
        let model =
            EpistemicNearestNeighbors::new(train_x, train_y, None, false, IndexDriver::Exact)
                .unwrap();

        let mut rng = StdRng::seed_from_u64(42);
        let mut fitter = ENNFitter::new(2, true);
        fitter.reset_y_stats(&model.train_y());

        let result = fitter.ask(&model, 5, 3, None, &mut rng).unwrap();

        assert_eq!(result.k_num_neighbors, 2);
        assert!(result.epistemic_variance_scale > 0.0);
    }

    #[test]
    fn test_subsample_loglik_invalid_p() {
        let model = create_test_model();
        let x = array![[0.5, 0.5]];
        let y = array![[1.0]];
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let paramss = vec![params];
        let mut rng = StdRng::seed_from_u64(42);

        let result = subsample_loglik(&model, &x.view(), &y.view(), &paramss, 0, &mut rng, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_subsample_loglik_mismatched_xy() {
        let model = create_test_model();
        let x = array![[0.5, 0.5], [0.2, 0.8]]; // 2 rows
        let y = array![[1.0]]; // 1 row
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let paramss = vec![params];
        let mut rng = StdRng::seed_from_u64(42);

        let result = subsample_loglik(&model, &x.view(), &y.view(), &paramss, 2, &mut rng, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_subsample_inputs_empty_paramss() {
        let x = array![[0.5, 0.5]];
        let y = array![[1.0]];
        let err = validate_subsample_inputs(&x.view(), &y.view(), 1, 0).unwrap_err();
        assert!(err.to_string().contains("paramss must be non-empty"));
    }

    #[test]
    fn test_compute_single_loglik_gaussian_term_golden() {
        let y_scaled = array![[1.0]];
        let mu_i = array![[0.0]];
        let se_i = array![[1.0]];
        let ll = compute_single_loglik(&y_scaled.view(), &mu_i.view(), &se_i.view());
        let expected = -0.5 * ((2.0 * std::f64::consts::PI).ln() + 1.0);
        assert!((ll - expected).abs() < 1e-12);
    }

    #[test]
    fn test_subsample_loglik_custom_y_std_seed_11() {
        let model = create_test_model();
        let x = array![[0.5, 0.5], [0.2, 0.8]];
        let y = array![[1.0], [1.2]];
        let params = ENNParams::new(2, 1.0, 0.1).unwrap();
        let y_std = array![2.0];
        let mut rng = StdRng::seed_from_u64(11);
        let paramss = vec![params];
        let logliks = subsample_loglik(
            &model,
            &x.view(),
            &y.view(),
            &paramss,
            2,
            &mut rng,
            Some(&y_std.view()),
        )
        .unwrap();
        assert_eq!(logliks.len(), 1);
        assert!(logliks[0].is_finite());
        let again = subsample_loglik(
            &model,
            &x.view(),
            &y.view(),
            &paramss,
            2,
            &mut StdRng::seed_from_u64(11),
            Some(&y_std.view()),
        )
        .unwrap();
        assert!((logliks[0] - again[0]).abs() < 1e-12);
    }

    #[test]
    fn test_compute_single_loglik_nonfinite_and_nonpositive_se() {
        let y = array![[1.0]];
        let mu_bad = array![[f64::NAN]];
        let se_ok = array![[1.0]];
        assert_eq!(
            compute_single_loglik(&y.view(), &mu_bad.view(), &se_ok.view()),
            0.0
        );

        let mu_ok = array![[1.0]];
        let se_bad = array![[0.0]];
        assert_eq!(
            compute_single_loglik(&y.view(), &mu_ok.view(), &se_bad.view()),
            0.0
        );
    }
}
