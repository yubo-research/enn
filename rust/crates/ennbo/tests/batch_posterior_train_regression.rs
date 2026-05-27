use ennbo::index::IndexDriver;
use ennbo::model::EpistemicNearestNeighbors;
use ennbo::params::{ENNParams, PosteriorFlags};
use ennbo::traits::PosteriorComputation;
use ndarray::array;

#[test]
fn batch_posterior_on_train_matches_posterior_when_no_neighbors() {
    let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let train_y = array![[0.0], [1.0], [1.0], [2.0]];
    let model = EpistemicNearestNeighbors::new(
        train_x.clone(),
        train_y,
        None,
        false,
        IndexDriver::Exact,
    )
    .unwrap();
    let bad = ENNParams {
        k_num_neighbors: 0,
        epistemic_variance_scale: 1.0,
        aleatoric_variance_scale: 0.1,
    };
    let flags = PosteriorFlags::new();
    let batch = model.batch_posterior(&train_x.view(), &[bad], &flags).unwrap();
    let ind = model.posterior(&train_x.view(), &bad, &flags).unwrap();
    for i in 0..train_x.nrows() {
        assert_eq!(batch.mu[[0, i, 0]], ind.mu[[i, 0]]);
        assert_eq!(batch.se[[0, i, 0]], ind.se[[i, 0]]);
    }
}

#[test]
fn batch_posterior_on_train_row_matches_single_query_posterior() {
    let train_x = ndarray::Array2::from_shape_fn((20, 1), |(i, _)| {
        (i as f64 - 9.5) / 3.0 + 0.01 * (i as f64)
    });
    let train_y = ndarray::Array2::from_shape_fn((20, 1), |(i, _)| {
        let z = (i as f64 + 1.0) * 0.37 - 2.1;
        z * 100.0
    });
    let model = EpistemicNearestNeighbors::new(
        train_x.clone(),
        train_y,
        Some(ndarray::Array2::zeros((20, 1))),
        false,
        IndexDriver::Exact,
    )
    .unwrap();
    let params = ENNParams {
        k_num_neighbors: 10,
        epistemic_variance_scale: 80.0,
        aleatoric_variance_scale: 0.0,
    };
    let flags = PosteriorFlags {
        exclude_nearest: false,
        observation_noise: false,
    };
    let batch = model
        .batch_posterior(&train_x.view(), &[params], &flags)
        .unwrap();
    for i in 0..train_x.nrows() {
        let row = train_x.slice(ndarray::s![i..i + 1, ..]);
        let one = model.posterior(&row, &params, &flags).unwrap();
        let mu_diff = (batch.mu[[0, i, 0]] - one.mu[[0, 0]]).abs();
        let se_diff = (batch.se[[0, i, 0]] - one.se[[0, 0]]).abs();
        assert!(
            mu_diff < 1e-12 && se_diff < 1e-12,
            "row {i}: mu_diff={mu_diff:e} se_diff={se_diff:e}"
        );
    }
}
