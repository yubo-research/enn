use ennbo::index::IndexDriver;
use ennbo::model::EpistemicNearestNeighbors;
use ennbo::params::{ENNParams, PosteriorFlags};
use ndarray::array;

#[test]
fn batch_posterior_on_train_matches_posterior_when_no_neighbors() {
    let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let train_y = array![[0.0], [1.0], [1.0], [2.0]];
    let model =
        EpistemicNearestNeighbors::new(train_x.clone(), train_y, None, false, IndexDriver::Flat)
            .unwrap();
    let bad = ENNParams {
        k_num_neighbors: 0,
        epistemic_variance_scale: 1.0,
        aleatoric_variance_scale: 0.1,
    };
    let flags = PosteriorFlags::new();
    let batch = model
        .batch_posterior(&train_x.view(), &[bad], &flags)
        .unwrap();
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
        IndexDriver::Flat,
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

fn assert_mixed_batch_matches(
    model: &EpistemicNearestNeighbors,
    query: &ndarray::Array2<f64>,
    exclude: bool,
) {
    let mut params: Vec<_> = [30, 1, 1000, 3, 10]
        .into_iter()
        .map(|k| ENNParams::new(k, 0.3, 0.1).unwrap())
        .collect();
    params.push(ENNParams {
        k_num_neighbors: 0,
        ..params[0]
    });
    let flags = PosteriorFlags {
        exclude_nearest: exclude,
        observation_noise: false,
    };
    let batch = model
        .batch_posterior(&query.view(), &params, &flags)
        .unwrap();
    assert!(batch.idx.is_none());
    for (i, p) in params.iter().enumerate() {
        let single = model.posterior(&query.view(), p, &flags).unwrap();
        for (mixed, scalar) in [
            (&batch.mu, &single.mu),
            (&batch.se, &single.se),
            (&batch.se_epi, &single.se_epi),
            (&batch.se_ale, &single.se_ale),
        ] {
            let difference = mixed.index_axis(ndarray::Axis(0), i).to_owned().into_dyn() - scalar;
            assert!(
                difference.iter().all(|d| d.abs() < 1e-12),
                "k={}: {difference:?}",
                p.k_num_neighbors
            );
        }
    }
}

#[test]
fn mixed_batch_light_handles_ties_self_queries_and_clipped_counts() {
    for tied in [false, true] {
        let x = ndarray::Array2::from_shape_fn((100, 2), |(i, j)| {
            if tied {
                1.0
            } else {
                ((i * 7 + j * 3) as f64 * 0.03).sin()
            }
        });
        let y = ndarray::Array2::from_shape_fn((100, 2), |(i, j)| (i + j) as f64);
        let model =
            EpistemicNearestNeighbors::new(x.clone(), y, None, false, IndexDriver::Flat).unwrap();
        for exclude in [false, true] {
            assert_mixed_batch_matches(&model, &x, exclude);
            assert_mixed_batch_matches(&model, &array![[0.3, 0.7]], exclude);
        }
    }
}

#[test]
fn mixed_batch_light_handles_empty_models_and_queries() {
    for n in [0, 1, 3] {
        let x = ndarray::Array2::zeros((n, 2));
        let y = ndarray::Array2::ones((n, 2));
        let model = EpistemicNearestNeighbors::new(x, y, None, false, IndexDriver::Flat).unwrap();
        assert_mixed_batch_matches(&model, &array![[0.3, 0.7]], false);
        assert_mixed_batch_matches(&model, &ndarray::Array2::zeros((0, 2)), false);
        let params = [
            ENNParams::new(3, 1.0, 0.0).unwrap(),
            ENNParams::new(10, 1.0, 0.0).unwrap(),
        ];
        assert!(model
            .batch_posterior(
                &ndarray::Array2::zeros((1, 3)).view(),
                &params,
                &PosteriorFlags::new()
            )
            .is_err());
        if n == 1 {
            let flags = PosteriorFlags {
                exclude_nearest: true,
                observation_noise: false,
            };
            assert!(model
                .batch_posterior(&array![[0.3, 0.7]].view(), &params, &flags)
                .is_err());
        }
    }
}

#[test]
fn mixed_batch_preserves_each_k_search_before_f64_refinement() {
    let mut x = ndarray::Array2::from_shape_fn((100, 1), |(i, _)| 1.0 + i as f64 * 1e-10);
    x[[0, 0]] = 0.0;
    let y = ndarray::Array2::from_shape_fn((100, 2), |(i, j)| (i + j) as f64);
    let model = EpistemicNearestNeighbors::new(x, y, None, false, IndexDriver::Flat).unwrap();
    assert_mixed_batch_matches(&model, &array![[2.0]], false);
}
