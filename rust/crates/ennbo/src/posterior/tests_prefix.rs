use super::*;
use crate::model::EpistemicNearestNeighbors;
use ndarray::{Array2, Axis};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::sync::Mutex;

static TOGGLE: Mutex<()> = Mutex::new(());

const DIM: usize = 6;
const SHARED: usize = 4;

fn matrix(rng: &mut ChaCha8Rng, n: usize, lattice: bool) -> Array2<f64> {
    Array2::from_shape_fn((n, DIM), |_| {
        if lattice {
            f64::from(rng.gen_range(0..3))
        } else {
            rng.gen_range(-1.0..1.0)
        }
    })
}

fn prefix_batch(rng: &mut ChaCha8Rng, rows: usize, run: usize, lattice: bool) -> Array2<f64> {
    let mut x = matrix(rng, rows, lattice);
    for r in 0..rows {
        for c in 0..SHARED {
            x[[r, c]] = x[[(r / run) * run, c]];
        }
    }
    x
}

fn both_paths(
    model: &EpistemicNearestNeighbors,
    x: &Array2<f64>,
    params: &ENNParams,
    flags: &PosteriorFlags,
) -> (DrawInternals, DrawInternals) {
    let _guard = TOGGLE.lock().unwrap_or_else(|e| e.into_inner());
    crate::set_prefix_reuse(false);
    let base = compute_posterior_internals(model, &x.view(), params, flags).unwrap();
    crate::set_prefix_reuse(true);
    let fast = compute_posterior_internals(model, &x.view(), params, flags).unwrap();
    (base, fast)
}

fn assert_identical(model: &EpistemicNearestNeighbors, x: &Array2<f64>, k: i32, noise: bool) {
    let params = ENNParams::new(k, 1.0, 0.1).unwrap();
    let flags = PosteriorFlags::new().with_observation_noise(noise);
    let (base, fast) = both_paths(model, x, &params, &flags);
    assert_eq!(base, fast);
}

#[test]
fn shared_prefix_batches_give_bit_identical_posteriors() {
    let mut rng = ChaCha8Rng::seed_from_u64(1);
    let train_x = matrix(&mut rng, 400, false);
    let train_y = Array2::from_shape_fn((400, 2), |_| rng.gen_range(-1.0..1.0));
    let model = crate::test_helpers::test_model(train_x, train_y, None, false);
    let batch = prefix_batch(&mut rng, 40, 8, false);
    assert_identical(&model, &batch, 10, false);
    assert_identical(&model, &batch, 10, true);
    assert_identical(&model, &batch, 1, true);
    let no_shared = matrix(&mut rng, 40, false);
    assert_identical(&model, &no_shared, 10, true);
}

#[test]
fn observation_variances_and_weighting_are_identical() {
    let mut rng = ChaCha8Rng::seed_from_u64(2);
    let train_x = matrix(&mut rng, 300, false);
    let train_y = Array2::from_shape_fn((300, 1), |_| rng.gen_range(-1.0..1.0));
    let train_yvar = Array2::from_shape_fn((300, 1), |_| rng.gen_range(0.01..0.5));
    let model = crate::test_helpers::test_model(train_x, train_y, Some(train_yvar), true);
    let batch = prefix_batch(&mut rng, 32, 8, false);
    assert_identical(&model, &batch, 7, true);
}

#[test]
fn y_bounds_warping_is_identical() {
    let mut rng = ChaCha8Rng::seed_from_u64(3);
    let train_x = matrix(&mut rng, 200, false);
    let train_y = Array2::from_shape_fn((200, 1), |_| rng.gen_range(0.05..0.95));
    let bounds = ndarray::array![[0.0, 1.0]];
    let model = crate::test_helpers::test_model_with(false, |layout| {
        EpistemicNearestNeighbors::new_with_storage(train_x, train_y, None, layout, Some(bounds))
            .unwrap()
    });
    let batch = prefix_batch(&mut rng, 24, 6, false);
    assert_identical(&model, &batch, 5, true);
}

#[test]
fn duplicate_rows_and_lattice_ties_are_identical() {
    let mut rng = ChaCha8Rng::seed_from_u64(4);
    let train_x = matrix(&mut rng, 250, true);
    let train_y = Array2::from_shape_fn((250, 1), |_| rng.gen_range(-1.0..1.0));
    let model = crate::test_helpers::test_model(train_x, train_y, None, false);
    let batch = prefix_batch(&mut rng, 36, 6, true);
    assert_identical(&model, &batch, 10, true);
}

#[test]
fn k_at_or_above_n_and_empty_model_are_identical() {
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let train_x = matrix(&mut rng, 8, false);
    let train_y = Array2::from_shape_fn((8, 1), |_| rng.gen_range(-1.0..1.0));
    let model = crate::test_helpers::test_model(train_x, train_y, None, false);
    let batch = prefix_batch(&mut rng, 12, 4, false);
    assert_identical(&model, &batch, 8, true);
    assert_identical(&model, &batch, 20, true);

    let empty = crate::test_helpers::test_model(
        Array2::zeros((0, DIM)),
        Array2::zeros((0, 1)),
        None,
        false,
    );
    assert_identical(&empty, &batch, 5, true);
    assert_eq!(batch.len_of(Axis(1)), DIM);
}
