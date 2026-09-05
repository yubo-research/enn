//! Dual warped/natural posterior API under non-identity y_bounds.

use super::*;
use crate::backend::EnnStorage;
use crate::index::IndexDriver;
use crate::model::EpistemicNearestNeighbors;
use ndarray::array;

fn logit_unit_model() -> EpistemicNearestNeighbors {
    let train_x = array![[0.0], [1.0], [0.5]];
    let train_y = array![[0.1], [0.9], [0.5]];
    let bounds = array![[0.0, 1.0]];
    EpistemicNearestNeighbors::new_with_storage(
        train_x,
        train_y,
        None,
        false,
        IndexDriver::Exact,
        EnnStorage::InMemory,
        None,
        Some(bounds),
    )
    .unwrap()
}

#[test]
fn batch_posterior_naturalizes_under_y_bounds() {
    use crate::traits::PosteriorComputation;

    let model = logit_unit_model();
    let params = ENNParams::new(2, 1.0, 0.0).unwrap();
    let flags = PosteriorFlags::new();
    let query = array![[0.25]];

    let natural = model
        .batch_posterior(&query.view(), &[params], &flags)
        .unwrap();
    let warped =
        PosteriorComputation::batch_posterior(&model, &query.view(), &[params], &flags).unwrap();

    assert_eq!(natural.mu.shape(), &[1, 1, 1]);
    assert!(natural.mu[[0, 0, 0]] > 0.0 && natural.mu[[0, 0, 0]] < 1.0);
    assert!((natural.mu[[0, 0, 0]] - warped.mu[[0, 0, 0]]).abs() > 1e-6);
}

#[test]
fn posterior_warped_differs_from_naturalized_posterior() {
    let model = logit_unit_model();
    let params = ENNParams::new(2, 1.0, 0.0).unwrap();
    let flags = PosteriorFlags::new();
    let query = array![[0.25]];

    let warped = model
        .posterior_warped(&query.view(), &params, &flags)
        .unwrap();
    let natural = model.posterior(&query.view(), &params, &flags).unwrap();

    assert!(natural.mu[[0, 0]] > 0.0 && natural.mu[[0, 0]] < 1.0);
    assert!((natural.mu[[0, 0]] - warped.mu[[0, 0]]).abs() > 1e-6);

    let mut copy = warped.clone();
    model.naturalize_enn_normal(&mut copy).unwrap();
    assert!((copy.mu[[0, 0]] - natural.mu[[0, 0]]).abs() < 1e-12);
    assert!((copy.se[[0, 0]] - natural.se[[0, 0]]).abs() < 1e-12);
}

#[test]
fn posterior_function_draw_warped_naturalize_draws_3d() {
    let model = logit_unit_model();
    let params = ENNParams::new(2, 1.0, 0.0).unwrap();
    let flags = PosteriorFlags::new();
    let query = array![[0.5]];
    let seeds = vec![7i64, 8];

    let (warped_draws, _) = model
        .posterior_function_draw_warped(&query.view(), &params, &seeds, &flags)
        .unwrap();
    let (natural_draws, _) = model
        .posterior_function_draw(&query.view(), &params, &seeds, &flags)
        .unwrap();

    assert_eq!(warped_draws.shape(), natural_draws.shape());
    assert!(natural_draws.iter().all(|&v| v > 0.0 && v < 1.0));

    let max_diff = warped_draws
        .iter()
        .zip(natural_draws.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_diff > 1e-6);

    let mut manual = warped_draws.clone();
    model.naturalize_draws_3d(&mut manual);
    for (a, b) in manual.iter().zip(natural_draws.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn conditional_posterior_warped_vs_natural() {
    let model = logit_unit_model();
    let params = ENNParams::new(2, 1.0, 0.0).unwrap();
    let flags = PosteriorFlags::new();
    let query = array![[0.4]];
    let x_whatif = array![[0.5]];
    let y_whatif = array![[0.6]];

    let warped = model
        .conditional_posterior_warped(
            &x_whatif.view(),
            &y_whatif.view(),
            &query.view(),
            &params,
            &flags,
        )
        .unwrap();
    let natural = model
        .conditional_posterior(
            &x_whatif.view(),
            &y_whatif.view(),
            &query.view(),
            &params,
            &flags,
        )
        .unwrap();

    assert!(natural.mu[[0, 0]] > 0.0 && natural.mu[[0, 0]] < 1.0);
    assert!((natural.mu[[0, 0]] - warped.mu[[0, 0]]).abs() > 1e-6);
}

#[test]
fn y_scale_row_matches_warped_fit_scale_under_y_bounds() {
    let model = logit_unit_model();
    let public = model.y_scale_row()[[0, 0]];
    let internal = model.y_scale()[0];
    assert!(
        (public - internal).abs() < 1e-12,
        "y_scale_row={public} must match fit y_scale={internal}"
    );

    let all: Vec<usize> = (0..model.len()).collect();
    let (_, y_nat, _) = model.train_rows_at(&all).unwrap();
    let mean = y_nat.mean().unwrap();
    let nat_var = y_nat.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / (y_nat.len() as f64);
    let nat_std = nat_var.sqrt().max(1e-12);
    assert!(
        (public - nat_std).abs() > 0.1,
        "public scale must not be natural-y std under bounds: public={public} nat_std={nat_std}"
    );
}

#[test]
fn y_bounds_edge_lower_only_log_warp_roundtrip_and_oob() {
    let train_x = array![[0.0], [1.0], [0.5]];
    let train_y = array![[0.5], [2.0], [1.0]];
    let bounds = array![[0.0, f64::INFINITY]];
    let model = EpistemicNearestNeighbors::new_with_storage(
        train_x.clone(),
        train_y.clone(),
        None,
        false,
        IndexDriver::Exact,
        EnnStorage::InMemory,
        None,
        Some(bounds.clone()),
    )
    .unwrap();

    let all: Vec<usize> = (0..3).collect();
    let (_, y_nat, _) = model.train_rows_at(&all).unwrap();
    for (a, b) in y_nat.iter().zip(train_y.iter()) {
        assert!((a - b).abs() < 1e-12);
    }
    let (_, y_z, _) = model.rows().train_rows_at(&all).unwrap();
    assert!((y_z[[0, 0]] - 0.5_f64.ln()).abs() < 1e-12);

    let oob = EpistemicNearestNeighbors::new_with_storage(
        train_x,
        array![[-0.1], [2.0], [1.0]],
        None,
        false,
        IndexDriver::Exact,
        EnnStorage::InMemory,
        None,
        Some(bounds),
    );
    assert!(oob.is_err(), "y <= lower must be rejected for (0, inf)");
}

#[test]
fn y_bounds_edge_upper_only_neglog_warp_and_samples_in_bounds() {
    let train_x = array![[0.0], [1.0], [0.5]];
    let train_y = array![[1.0], [4.0], [2.5]];
    let bounds = array![[f64::NEG_INFINITY, 5.0]];
    let model = EpistemicNearestNeighbors::new_with_storage(
        train_x,
        train_y,
        None,
        false,
        IndexDriver::Exact,
        EnnStorage::InMemory,
        None,
        Some(bounds),
    )
    .unwrap();
    let params = ENNParams::new(2, 1.0, 0.0).unwrap();
    let flags = PosteriorFlags::new();
    let query = array![[0.25]];
    let (draws, _) = model
        .posterior_function_draw(&query.view(), &params, &[1i64, 2, 3], &flags)
        .unwrap();
    assert!(draws.iter().all(|&v| v < 5.0 && v.is_finite()));
    let natural = model.posterior(&query.view(), &params, &flags).unwrap();
    assert!(natural.mu[[0, 0]] < 5.0);
}

#[test]
fn y_bounds_edge_near_open_endpoints_logit() {
    let train_x = array![[0.0], [1.0], [0.5], [0.25]];
    let train_y = array![[1e-6], [1.0 - 1e-6], [0.5], [0.25]];
    let bounds = array![[0.0, 1.0]];
    let model = EpistemicNearestNeighbors::new_with_storage(
        train_x,
        train_y.clone(),
        None,
        false,
        IndexDriver::Exact,
        EnnStorage::InMemory,
        None,
        Some(bounds),
    )
    .unwrap();
    let all: Vec<usize> = (0..4).collect();
    let (_, y_nat, _) = model.train_rows_at(&all).unwrap();
    for (a, b) in y_nat.iter().zip(train_y.iter()) {
        assert!((a - b).abs() < 1e-9, "{a} vs {b}");
    }
    let (_, y_z, _) = model.rows().train_rows_at(&all).unwrap();
    assert!(y_z[[0, 0]].is_finite() && y_z[[0, 0]] < -10.0);
    assert!(y_z[[1, 0]].is_finite() && y_z[[1, 0]] > 10.0);
}

#[test]
fn y_bounds_edge_empty_model_y_scale_row_is_ones() {
    let model = EpistemicNearestNeighbors::new_empty_with_y_bounds(
        1,
        1,
        IndexDriver::Exact,
        EnnStorage::InMemory,
        None,
        None,
        Some(array![[0.0, 1.0]]),
    )
    .unwrap();
    assert_eq!(model.len(), 0);
    let scale = model.y_scale_row();
    assert_eq!(scale.shape(), &[1, 1]);
    assert!((scale[[0, 0]] - 1.0).abs() < 1e-15);
}

#[test]
fn y_bounds_edge_identity_equals_storage() {
    let train_x = array![[0.0], [1.0]];
    let train_y = array![[1.5], [-0.5]];
    let model = EpistemicNearestNeighbors::new_with_storage(
        train_x,
        train_y.clone(),
        None,
        false,
        IndexDriver::Exact,
        EnnStorage::InMemory,
        None,
        None,
    )
    .unwrap();
    let all: Vec<usize> = (0..2).collect();
    let (_, y_pub, _) = model.train_rows_at(&all).unwrap();
    let (_, y_z, _) = model.rows().train_rows_at(&all).unwrap();
    assert_eq!(y_pub, y_z);
    assert_eq!(y_pub, train_y);
    assert!((model.y_scale_row()[[0, 0]] - model.y_scale()[0]).abs() < 1e-15);
}
