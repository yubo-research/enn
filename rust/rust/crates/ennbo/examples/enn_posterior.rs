use ndarray::array;
use ennbo::{EpistemicNearestNeighbors, ENNParams, IndexDriver, PosteriorFlags};

fn main() {
    let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let train_y = array![[0.0], [1.0], [1.0], [0.0]];
    let model = EpistemicNearestNeighbors::new(train_x, train_y, None, false, IndexDriver::Flat)
        .unwrap();
    let params = ENNParams::new(2, 1.0, 0.1).unwrap();
    let query = array![[0.5, 0.5]];
    let out = model.posterior(&query.view(), &params, &PosteriorFlags::default()).unwrap();
    println!("mu={:?}", out.mu);
}
