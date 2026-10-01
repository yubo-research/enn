use ndarray::array;
use ennbo::{EpistemicNearestNeighbors, IndexDriver};

fn main() {
    let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let train_y = array![[0.0], [1.0], [1.0], [0.0]];
    let mut model = EpistemicNearestNeighbors::new(
        train_x.clone(),
        train_y.clone(),
        None,
        false,
        IndexDriver::Flat,
    )
    .unwrap();
    let err = model
        .enable_auto_metric(vec![], &train_x.view(), &train_y.view())
        .unwrap_err();
    println!("{err}");
}
