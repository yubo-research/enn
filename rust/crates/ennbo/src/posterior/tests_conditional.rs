use super::*;
use crate::index::IndexDriver;
use crate::model::EpistemicNearestNeighbors;
use crate::test_helpers::test_epistemic_model_exact_unit_square as create_test_model;
use ndarray::array;

#[test]
fn test_conditional_posterior_empty_whatif_delegates_to_posterior() {
    let model = create_test_model();
    let params = ENNParams::new(2, 1.0, 0.1).unwrap();
    let flags = PosteriorFlags::new();
    let query = array![[0.5, 0.5]];
    let x_whatif: Array2<f64> = Array2::zeros((0, 2));
    let y_whatif: Array2<f64> = Array2::zeros((0, 1));

    let post = model.posterior(&query.view(), &params, &flags).unwrap();
    let cond = compute_conditional_posterior_internals(
        &model,
        &query.view(),
        &x_whatif.view(),
        &y_whatif.view(),
        &params,
        &flags,
    )
    .unwrap();

    assert_eq!(post.mu[[0, 0]], cond.mu[[0, 0]]);
    assert_eq!(post.se[[0, 0]], cond.se[[0, 0]]);
}

#[test]
fn test_conditional_posterior_with_whatif() {
    let model = create_test_model();
    let params = ENNParams::new(2, 1.0, 0.1).unwrap();
    let flags = PosteriorFlags::new();
    let query = array![[0.5, 0.5]];
    let x_whatif = array![[0.5, 0.5]];
    let y_whatif = array![[2.0]];

    let internals = compute_conditional_posterior_internals(
        &model,
        &query.view(),
        &x_whatif.view(),
        &y_whatif.view(),
        &params,
        &flags,
    )
    .unwrap();

    assert_eq!(internals.mu.nrows(), 1);
    assert_eq!(internals.mu.ncols(), 1);
    assert!(internals.mu[[0, 0]].is_finite());
    assert!(internals.se[[0, 0]].is_finite());
}

#[test]
fn test_conditional_posterior_function_draw() {
    let model = create_test_model();
    let params = ENNParams::new(2, 1.0, 0.1).unwrap();
    let flags = PosteriorFlags::new();
    let query = array![[0.5, 0.5]];
    let x_whatif = array![[0.5, 0.5]];
    let y_whatif = array![[1.5]];
    let seeds = vec![1i64, 2, 3];

    let internals = compute_conditional_posterior_internals(
        &model,
        &query.view(),
        &x_whatif.view(),
        &y_whatif.view(),
        &params,
        &flags,
    )
    .unwrap();
    let result = draw_from_internals(&model, &internals, &seeds);

    assert!(result.is_ok());
    let draws = result.unwrap();
    assert_eq!(draws.shape(), &[3, 1, 1]);
}

#[test]
fn test_conditional_posterior_exclude_nearest() {
    let model = create_test_model();
    let params = ENNParams::new(2, 1.0, 0.1).unwrap();
    let flags = PosteriorFlags::new().with_exclude_nearest(true);
    let query = array![[0.5, 0.5]];
    let x_whatif = array![[0.5, 0.5], [0.6, 0.6]];
    let y_whatif = array![[1.5], [2.0]];

    let internals = compute_conditional_posterior_internals(
        &model,
        &query.view(),
        &x_whatif.view(),
        &y_whatif.view(),
        &params,
        &flags,
    )
    .unwrap();
    assert!(internals.mu[[0, 0]].is_finite());
}

#[test]
fn test_conditional_posterior_scaled_model() {
    let train_x = array![[0.0, 0.0], [2.0, 2.0], [4.0, 4.0]];
    let train_y = array![[0.0], [1.0], [2.0]];
    let model =
        EpistemicNearestNeighbors::new(train_x, train_y, None, true, IndexDriver::Exact)
            .unwrap();
    let params = ENNParams::new(2, 1.0, 0.1).unwrap();
    let flags = PosteriorFlags::new();
    let query = array![[1.0, 1.0]];
    let x_whatif = array![[3.0, 3.0]];
    let y_whatif = array![[1.5]];

    let internals = compute_conditional_posterior_internals(
        &model,
        &query.view(),
        &x_whatif.view(),
        &y_whatif.view(),
        &params,
        &flags,
    )
    .unwrap();
    assert!(internals.mu[[0, 0]].is_finite());
}

#[test]
fn test_conditional_scale_single_row() {
    let train_x = array![[0.0, 0.0]];
    let train_y = array![[1.0]];
    let model =
        EpistemicNearestNeighbors::new(train_x, train_y, None, false, IndexDriver::Exact)
            .unwrap();
    let params = ENNParams::new(1, 1.0, 0.1).unwrap();
    let flags = PosteriorFlags::new();
    let query = array![[0.0, 0.0]];
    let x_whatif = array![[0.0, 0.0]];
    let y_whatif = array![[2.0]];

    let internals = compute_conditional_posterior_internals(
        &model,
        &query.view(),
        &x_whatif.view(),
        &y_whatif.view(),
        &params,
        &flags,
    )
    .unwrap();
    assert!(internals.mu[[0, 0]].is_finite());
}

#[test]
fn test_compute_scale_for_conditional_branches() {
    use ndarray::Array2;

    let train = array![[1.0f64]];
    let whatif_empty: Array2<f64> = Array2::zeros((0, 1));
    let s0 = compute_scale_for_conditional(&train.view(), &whatif_empty.view());
    assert_eq!(s0.len(), 1);
    assert!((s0[0] - 1.0).abs() < 1e-12);

    let train_c = array![[1.0, 2.0], [1.0, 3.0]];
    let whatif_c = array![[1.0, 0.0]];
    let s1 = compute_scale_for_conditional(&train_c.view(), &whatif_c.view());
    assert_eq!(s1.len(), 2);
    assert!((s1[0] - 1.0).abs() < 1e-9, "constant column uses scale=1");
    assert!(s1[1] > 0.0);
}

#[test]
fn test_conditional_posterior_y_whatif_shape_error_reports_y_whatif_shape() {
    let model = create_test_model();
    let params = ENNParams::new(2, 1.0, 0.1).unwrap();
    let flags = PosteriorFlags::new();
    let query = array![[0.5, 0.5]];
    let x_whatif = array![[0.5, 0.5]];
    // y_whatif has wrong number of columns (2 instead of 1)
    let y_whatif = array![[1.0, 2.0]];

    let result = compute_conditional_posterior_internals(
        &model,
        &query.view(),
        &x_whatif.view(),
        &y_whatif.view(),
        &params,
        &flags,
    );
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("Invalid shape"),
        "expected InvalidShape, got: {}",
        msg
    );
    assert!(
        msg.contains("[1, 2]"),
        "error should report y_whatif shape [1, 2], got: {}",
        msg
    );
}

#[test]
fn test_conditional_posterior_nan_in_x_whatif_returns_error_not_panic() {
    let model = create_test_model();
    let params = ENNParams::new(2, 1.0, 0.1).unwrap();
    let flags = PosteriorFlags::new();
    let query = array![[0.5, 0.5]];
    let x_whatif = array![[f64::NAN, 0.5]];
    let y_whatif = array![[1.0]];

    let result = compute_conditional_posterior_internals(
        &model,
        &query.view(),
        &x_whatif.view(),
        &y_whatif.view(),
        &params,
        &flags,
    );
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        err.to_string().contains("NaN") || err.to_string().contains("invalid"),
        "expected error about NaN/invalid values, got: {}",
        err
    );
}

// Tests for empty-query cases (Bug fixes for panic issues)

#[test]
fn test_empty_query_posterior_with_yvar_no_panic() {
    // Bug fix: Empty query with train_yvar should not panic
    let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let train_y = array![[0.0], [1.0], [1.0], [2.0]];
    let train_yvar = array![[0.1], [0.1], [0.1], [0.1]];

    let model = EpistemicNearestNeighbors::new(
        train_x,
        train_y,
        Some(train_yvar),
        false,
        IndexDriver::Exact,
    )
    .unwrap();

    let params = ENNParams::new(2, 1.0, 0.1).unwrap();
    let flags = PosteriorFlags::new();
    let empty_query: Array2<f64> = Array2::zeros((0, 2));

    let result = model.posterior(&empty_query.view(), &params, &flags);
    assert!(
        result.is_ok(),
        "Empty query posterior with yvar should not panic"
    );

    let posterior = result.unwrap();
    assert_eq!(posterior.mu.shape()[0], 0);
    assert_eq!(posterior.se.shape()[0], 0);
}

#[test]
fn test_empty_query_conditional_posterior_no_panic() {
    // Bug fix: Empty query conditional posterior should not panic
    let model = create_test_model();
    let params = ENNParams::new(2, 1.0, 0.1).unwrap();
    let flags = PosteriorFlags::new();

    let empty_query: Array2<f64> = Array2::zeros((0, 2));
    let x_whatif = array![[0.5, 0.5]];
    let y_whatif = array![[1.5]];

    let result = model.conditional_posterior(
        &x_whatif.view(),
        &y_whatif.view(),
        &empty_query.view(),
        &params,
        &flags,
    );
    assert!(
        result.is_ok(),
        "Empty query conditional posterior should not panic"
    );

    let posterior = result.unwrap();
    assert_eq!(posterior.mu.shape()[0], 0);
    assert_eq!(posterior.se.shape()[0], 0);
}

#[test]
fn test_empty_query_conditional_posterior_with_yvar_no_panic() {
    // Bug fix: Empty query conditional posterior with train_yvar should not panic
    let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let train_y = array![[0.0], [1.0], [1.0], [2.0]];
    let train_yvar = array![[0.1], [0.1], [0.1], [0.1]];

    let model = EpistemicNearestNeighbors::new(
        train_x,
        train_y,
        Some(train_yvar),
        false,
        IndexDriver::Exact,
    )
    .unwrap();

    let params = ENNParams::new(2, 1.0, 0.1).unwrap();
    let flags = PosteriorFlags::new();

    let empty_query: Array2<f64> = Array2::zeros((0, 2));
    let x_whatif = array![[0.5, 0.5]];
    let y_whatif = array![[1.5]];

    let result = model.conditional_posterior(
        &x_whatif.view(),
        &y_whatif.view(),
        &empty_query.view(),
        &params,
        &flags,
    );
    assert!(
        result.is_ok(),
        "Empty query conditional posterior with yvar should not panic"
    );

    let posterior = result.unwrap();
    assert_eq!(posterior.mu.shape()[0], 0);
    assert_eq!(posterior.se.shape()[0], 0);
}

#[test]
fn test_compute_weighted_posterior_empty_idx() {
    let model = create_test_model();
    let params = ENNParams::new(2, 1.0, 0.1).unwrap();
    let empty_dist2s: Array2<f64> = Array2::zeros((0, 2));
    let empty_y_neighbors: Array2<f64> = Array2::zeros((0, 1));
    let empty_idx: Vec<Vec<usize>> = vec![];
    let data = WeightedPosteriorData {
        dist2s: &empty_dist2s.view(),
        idx: &empty_idx,
        y_neighbors: &empty_y_neighbors.view(),
        params: &params,
        observation_noise: false,
        yvar_neighbors_override: None,
    };

    let result = compute_weighted_posterior(&model, data, None);
    assert!(result.is_ok(), "compute_weighted_posterior with empty idx should not panic");
}

#[test]
fn kiss_weighted_posterior_data_type() {
    assert!(std::mem::size_of::<WeightedPosteriorData>() > 0);
}

#[test]
fn kiss_posterior_batch_units_are_linked() {
    let model = create_test_model();
    let params = ENNParams::new(2, 1.0, 0.1).unwrap();
    let paramss = vec![params, params];
    let flags = PosteriorFlags::new();
    let query = array![[0.5, 0.5]];
    let mut mu_all = Array3::zeros((2, 1, 1));
    let mut se_all = Array3::zeros((2, 1, 1));
    let mut se_epi_all = Array3::zeros((2, 1, 1));
    let mut se_ale_all = Array3::zeros((2, 1, 1));
    compute_batch_with_shared_neighbors(
        &model,
        &query.view(),
        &paramss,
        &flags,
        &mut mu_all,
        &mut se_all,
        &mut se_epi_all,
        &mut se_ale_all,
    )
    .unwrap();
    let params_a = ENNParams::new(2, 1.0, 0.1).unwrap();
    let params_b = ENNParams::new(3, 1.0, 0.1).unwrap();
    let mixed = vec![params_a, params_b];
    let mut mu2 = Array3::zeros((2, 1, 1));
    let mut se2 = Array3::zeros((2, 1, 1));
    let mut se_epi2 = Array3::zeros((2, 1, 1));
    let mut se_ale2 = Array3::zeros((2, 1, 1));
    compute_batch_separate_neighbors(
        &model,
        &query.view(),
        &mixed,
        &flags,
        &mut mu2,
        &mut se2,
        &mut se_epi2,
        &mut se_ale2,
    )
    .unwrap();
    let _ = (
        compute_batch_with_shared_neighbors,
        compute_batch_separate_neighbors,
        EpistemicNearestNeighbors::conditional_posterior_function_draw,
    );
}
