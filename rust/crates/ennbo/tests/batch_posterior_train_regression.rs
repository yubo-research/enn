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
