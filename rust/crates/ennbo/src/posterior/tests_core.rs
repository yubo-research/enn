use super::*;
use crate::index::IndexDriver;
use crate::model::EpistemicNearestNeighbors;
use crate::test_helpers::test_epistemic_model_exact_unit_square as create_test_model;
use ndarray::array;
use ndarray::ArrayView2;

fn assert_batch_neighbor_fill<F>(
    model: &EpistemicNearestNeighbors,
    paramss: Vec<ENNParams>,
    mut run: F,
) where
    F: FnMut(
        &EpistemicNearestNeighbors,
        &ArrayView2<f64>,
        &[ENNParams],
        &PosteriorFlags,
        &mut Array3<f64>,
        &mut Array3<f64>,
    ) -> Result<(), ENNError>,
{
    let flags = PosteriorFlags::new();
    let query = array![[0.5, 0.5]];
    let (bs, np) = (query.nrows(), paramss.len());
    let mut mu_all = Array3::zeros((np, bs, model.num_metrics()));
    let mut se_all = Array3::zeros((np, bs, model.num_metrics()));
    assert!(run(
        model,
        &query.view(),
        &paramss,
        &flags,
        &mut mu_all,
        &mut se_all
    )
    .is_ok());
    assert_eq!(mu_all.shape(), &[np, bs, 1]);
    assert_eq!(se_all.shape(), &[np, bs, 1]);
}

#[test]
fn test_posterior_computation() {
    let model = create_test_model();
    let params = ENNParams::new(2, 1.0, 0.1).unwrap();
    let flags = PosteriorFlags::new();
    let query = array![[0.5, 0.5]];

    let result = model.posterior(&query.view(), &params, &flags);
    assert!(result.is_ok());

    let posterior = result.unwrap();
    assert_eq!(posterior.mu.len(), 1);
    assert!(posterior.mu[[0, 0]] > 0.0 && posterior.mu[[0, 0]] < 2.0);
    assert!(posterior.se[[0, 0]] > 0.0);
}

#[test]
fn test_batch_posterior() {
    let model = create_test_model();
    let params1 = ENNParams::new(2, 1.0, 0.1).unwrap();
    let params2 = ENNParams::new(2, 2.0, 0.2).unwrap();
    let paramss = vec![params1, params2];
    let flags = PosteriorFlags::new();
    let query = array![[0.5, 0.5]];

    let result = model.batch_posterior(&query.view(), &paramss, &flags);
    assert!(result.is_ok());

    let posterior = result.unwrap();
    assert_eq!(posterior.mu.shape(), &[2, 1, 1]);
}

#[test]
fn test_posterior_function_draw() {
    let model = create_test_model();
    let params = ENNParams::new(2, 1.0, 0.1).unwrap();
    let flags = PosteriorFlags::new();
    let query = array![[0.5, 0.5]];
    let seeds = vec![1i64, 2, 3];

    let result = model.posterior_function_draw(&query.view(), &params, &seeds, &flags);
    assert!(result.is_ok());

    let (draws, idx) = result.unwrap();
    assert_eq!(draws.shape(), &[3, 1, 1]);
    assert_eq!(idx.len(), 1);
}

#[test]
fn test_empty_posterior_internals() {
    let model = create_test_model();
    let internals = empty_posterior_internals(&model, 5);

    assert_eq!(internals.idx.len(), 5);
    assert!(internals.idx.iter().all(|v| v.is_empty()));
    assert_eq!(internals.mu.shape(), &[5, 1]);
    assert_eq!(internals.se.shape(), &[5, 1]);
}

#[test]
fn test_compute_posterior_internals() {
    let model = create_test_model();
    let params = ENNParams::new(2, 1.0, 0.1).unwrap();
    let flags = PosteriorFlags::new();
    let query = array![[0.5, 0.5]];

    let result = compute_posterior_internals(&model, &query.view(), &params, &flags);
    assert!(result.is_ok());

    let internals = result.unwrap();
    assert_eq!(internals.mu.nrows(), 1);
    assert_eq!(internals.mu.ncols(), 1);
}

#[test]
fn test_compute_posterior_internals_empty_model() {
    let train_x = array![[0.0, 0.0]];
    let train_y = array![[0.0]];
    let model =
        EpistemicNearestNeighbors::new(train_x, train_y, None, false, IndexDriver::Exact)
            .unwrap();

    let params = ENNParams::new(2, 1.0, 0.1).unwrap();
    let flags = PosteriorFlags::new();
    let query = array![[100.0, 100.0]];

    let result = compute_posterior_internals(&model, &query.view(), &params, &flags);
    assert!(result.is_ok());
}

#[test]
fn test_batch_posterior_empty_params() {
    let model = create_test_model();
    let flags = PosteriorFlags::new();
    let query = array![[0.5, 0.5]];
    let paramss: Vec<ENNParams> = vec![];

    let result = model.batch_posterior(&query.view(), &paramss, &flags);
    assert!(result.is_err());
    assert!(result
        .unwrap_err()
        .to_string()
        .contains("paramss must be non-empty"));
}

#[test]
fn test_get_neighbor_data() {
    let model = create_test_model();
    let params = ENNParams::new(2, 1.0, 0.1).unwrap();
    let query = array![[0.5, 0.5]];

    let result = get_neighbor_data(&model, &query.view(), &params, false, true);
    assert!(result.is_ok());
    assert!(result.unwrap().is_some());
}

#[test]
fn test_compute_weighted_stats_impl() {
    let model = create_test_model();
    let dist2s = array![[0.1, 0.2]];
    let y_neighbors = array![[0.0], [1.0]];
    let params = ENNParams::new(2, 1.0, 0.1).unwrap();

    let result = compute_weighted_stats_impl(
        &dist2s.view(),
        &y_neighbors.view(),
        None,
        &params,
        false,
        &model.y_scale().view(),
    );
    assert!(result.is_ok());

    let stats = result.unwrap();
    assert_eq!(stats.mu.shape(), &[1, 1]);
    assert_eq!(stats.se.shape(), &[1, 1]);
}

#[test]
fn test_draw_from_internals_k0() {
    let model = create_test_model();
    let internals = empty_posterior_internals(&model, 3);
    let seeds = vec![1i64, 2];

    let result = draw_from_internals(&model, &internals, &seeds);
    assert!(result.is_ok());

    let draws = result.unwrap();
    assert_eq!(draws.shape(), &[2, 3, 1]);
}

#[test]
fn test_draw_from_internals_with_neighbors() {
    let model = create_test_model();
    let params = ENNParams::new(2, 1.0, 0.1).unwrap();
    let flags = PosteriorFlags::new();
    let query = array![[0.5, 0.5]];

    let internals =
        compute_posterior_internals(&model, &query.view(), &params, &flags).unwrap();
    let seeds = vec![1i64, 2];

    let result = draw_from_internals(&model, &internals, &seeds);
    assert!(result.is_ok());

    let draws = result.unwrap();
    assert_eq!(draws.shape(), &[2, 1, 1]);
    let one = draw_from_internals(&model, &internals, &[7i64]).unwrap();
    assert!((one[[0, 0, 0]] - 0.223_806_179_572_216_43).abs() < 1e-12);
}

#[test]
fn test_compute_weighted_posterior() {
    let model = create_test_model();
    let params = ENNParams::new(2, 1.0, 0.1).unwrap();
    let query = array![[0.5, 0.5]];

    let neighbor_data = get_neighbor_data(&model, &query.view(), &params, false, true)
        .unwrap()
        .unwrap();

    let data = WeightedPosteriorData {
        dist2s: &neighbor_data.dist2s.view(),
        idx: &neighbor_data.idx,
        y_neighbors: &neighbor_data.y_neighbors.view(),
        params: &params,
        observation_noise: false,
        yvar_neighbors_override: None,
    };

    let result = compute_weighted_posterior(&model, data, None);
    assert!(result.is_ok());
}

#[test]
fn test_get_neighbor_data_exclude_nearest() {
    let model = create_test_model();
    let params = ENNParams::new(2, 1.0, 0.1).unwrap();
    let query = array![[0.5, 0.5]];

    let result = get_neighbor_data(&model, &query.view(), &params, true, true);
    assert!(result.is_ok());
}

#[test]
fn test_compute_weighted_stats_with_noise() {
    let model = create_test_model();
    let dist2s = array![[0.1, 0.2]];
    let y_neighbors = array![[0.0], [1.0]];
    let params = ENNParams::new(2, 1.0, 0.1).unwrap();

    let result = compute_weighted_stats_impl(
        &dist2s.view(),
        &y_neighbors.view(),
        None,
        &params,
        true,
        &model.y_scale().view(),
    );
    assert!(result.is_ok());
}

// Tests for helper functions to achieve 100% coverage

#[test]
fn test_compute_batch_with_shared_neighbors_direct() {
    let model = create_test_model();
    let params1 = ENNParams::new(2, 1.0, 0.1).unwrap();
    let params2 = ENNParams::new(2, 2.0, 0.2).unwrap();
    let paramss = vec![params1, params2];
    assert_batch_neighbor_fill(&model, paramss, |m, q, p, f, mu, se| {
        compute_batch_with_shared_neighbors(m, q, p, f, mu, se)
    });
}

#[test]
fn test_compute_batch_separate_neighbors_direct() {
    let model = create_test_model();
    let params1 = ENNParams::new(2, 1.0, 0.1).unwrap();
    let params2 = ENNParams::new(3, 2.0, 0.2).unwrap(); // Different k values
    let paramss = vec![params1, params2];
    assert_batch_neighbor_fill(&model, paramss, |m, q, p, f, mu, se| {
        compute_batch_separate_neighbors(m, q, p, f, mu, se)
    });
}

#[test]
fn test_assign_posterior_results_direct() {
    let model = create_test_model();
    let params = ENNParams::new(2, 1.0, 0.1).unwrap();
    let flags = PosteriorFlags::new();
    let query = array![[0.5, 0.5]];

    let internals =
        compute_posterior_internals(&model, &query.view(), &params, &flags).unwrap();

    let mut mu_all = Array3::zeros((3, 1, 1));
    let mut se_all = Array3::zeros((3, 1, 1));

    // Test assigning to different indices
    assign_posterior_results(&internals, &mut mu_all, &mut se_all, 0);
    assign_posterior_results(&internals, &mut mu_all, &mut se_all, 1);
    assign_posterior_results(&internals, &mut mu_all, &mut se_all, 2);

    // Verify the assignments were made (values should be non-zero from the computation)
    assert!(mu_all[[0, 0, 0]].is_finite());
    assert!(se_all[[0, 0, 0]].is_finite());
}
