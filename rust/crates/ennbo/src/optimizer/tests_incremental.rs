use ndarray::array;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::fitter::{fit_probability, num_random_fit_candidates, ENNFitter};
use crate::optimizer::ObservationDelta;
use crate::index::IndexDriver;
use crate::model::EpistemicNearestNeighbors;
use crate::surrogate::{ENNSurrogate, ENNSurrogateConfig, Surrogate};

#[test]
fn scale_x_false_index_not_stale_after_add() {
    let train_x = array![[0.0, 0.0], [1.0, 0.0]];
    let train_y = array![[0.0], [1.0]];
    let mut model =
        EpistemicNearestNeighbors::new(train_x, train_y, None, false, IndexDriver::Exact).unwrap();
    model.sync_index().unwrap();
    assert!(!model.is_index_stale());
    let x_add = array![[0.5, 0.5]];
    let y_add = array![[0.5]];
    model.add(&x_add.view(), &y_add.view(), None).unwrap();
    assert!(!model.is_index_stale());
    model.sync_index().unwrap();
    assert_eq!(model.num_obs(), 3);
}

#[test]
fn scale_x_true_index_stale_after_add() {
    let train_x = array![[0.0, 0.0], [1.0, 0.0]];
    let train_y = array![[0.0], [1.0]];
    let mut model =
        EpistemicNearestNeighbors::new(train_x, train_y, None, true, IndexDriver::Exact).unwrap();
    model.sync_index().unwrap();
    let x_add = array![[0.5, 0.5]];
    let y_add = array![[0.5]];
    model.add(&x_add.view(), &y_add.view(), None).unwrap();
    assert!(model.is_index_stale());
}

#[test]
fn fit_policy_probability_and_candidate_count() {
    assert!((fit_probability(50) - 1.0).abs() < 1e-12);
    assert!((fit_probability(200) - 0.5).abs() < 1e-12);
    assert_eq!(num_random_fit_candidates(200), 1);
    assert_eq!(num_random_fit_candidates(50), 2);
}

#[test]
fn enn_fitter_first_fit_always_runs() {
    let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let train_y = array![[0.0], [1.0], [1.0], [2.0]];
    let model =
        EpistemicNearestNeighbors::new(train_x, train_y, None, false, IndexDriver::Exact).unwrap();
    let mut fitter = ENNFitter::new(2, 3, true, 1);
    fitter.reset_y_stats(&model.train_y());
    let mut rng = StdRng::seed_from_u64(99);
    let p = fitter.maybe_fit(&model, &mut rng, None).unwrap();
    assert!(p.is_some());
}

#[test]
fn observation_delta_views() {
    let delta = ObservationDelta {
        old_n: 2,
        new_n: 4,
        x_new: array![[0.1, 0.2], [0.3, 0.4]],
        y_new: array![[1.0], [2.0]],
    };
    assert_eq!(delta.x_new_view().nrows(), 2);
    assert_eq!(delta.y_new_view().nrows(), 2);
}

#[test]
fn enn_surrogate_fit_append_grows_model() {
    let config = ENNSurrogateConfig {
        k: 2,
        num_fit_candidates: 4,
        num_fit_samples: 2,
        ..Default::default()
    };
    let mut sur = ENNSurrogate::new(config);
    let mut rng = StdRng::seed_from_u64(42);
    let x0 = array![[0.0, 0.0], [1.0, 0.0]];
    let y0 = array![[0.0], [1.0]];
    sur.fit_append(&x0.view(), &y0.view(), None, &mut rng).unwrap();
    let x1 = array![[0.5, 0.5]];
    let y1 = array![[1.5]];
    sur.fit_append(&x1.view(), &y1.view(), None, &mut rng).unwrap();
    assert_eq!(sur.model().unwrap().num_obs(), 3);
}
