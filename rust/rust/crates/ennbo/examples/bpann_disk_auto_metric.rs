use ndarray::Array2;
use ennbo::{EpistemicNearestNeighbors, IndexDriver};

fn main() {
    let train_x = Array2::from_shape_fn((8, 2), |(i, j)| (i + j) as f64);
    let train_y = Array2::from_shape_fn((8, 1), |(i, _)| i as f64);
    let mut model = EpistemicNearestNeighbors::new(train_x, train_y, None, false, IndexDriver::Flat)
        .unwrap();
    model.enable_auto_metric(vec![], vec![]).unwrap();
    println!("auto={}", model.metric_learning_auto());
}
