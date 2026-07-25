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
fn posterior_warped_differs_from_naturalized_posterior() {
    let model = logit_unit_model();
    let params = ENNParams::new(2, 1.0, 0.0).unwrap();
    let flags = PosteriorFlags::new();
    let query = array![[0.25]];

    let warped = model
        .posterior_warped(&query.view(), &params, &flags)
        .unwrap();
    let natural = model.posterior(&query.view(), &params, &flags).unwrap();

    // Natural mu must lie in (0,1); warped logit mu need not.
    assert!(natural.mu[[0, 0]] > 0.0 && natural.mu[[0, 0]] < 1.0);
    assert!((natural.mu[[0, 0]] - warped.mu[[0, 0]]).abs() > 1e-6);

    // naturalize_enn_normal: applying inv to warped mu recovers natural mu.
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
    // naturalize_draws_3d must change values under logit bounds
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
