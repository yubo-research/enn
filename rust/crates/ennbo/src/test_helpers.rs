use crate::index::IndexDriver;
use crate::model::EpistemicNearestNeighbors;
use ndarray::array;

pub(crate) fn test_epistemic_model_exact_unit_square() -> EpistemicNearestNeighbors {
    let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let train_y = array![[0.0], [1.0], [1.0], [2.0]];

    EpistemicNearestNeighbors::new(train_x, train_y, None, false, IndexDriver::Flat).unwrap()
}
