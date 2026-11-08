use ennbo::{ENNParams, EpistemicNearestNeighbors, PosteriorFlags};
use ndarray::array;

fn main() {
    let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let train_y = array![[0.0], [1.0], [1.0], [0.0]];
    let dir = tempfile::tempdir().unwrap();
    let model = EpistemicNearestNeighbors::new(
        train_x,
        train_y,
        None,
        ennbo::EnnLayout::new(dir.path(), false).unwrap(),
    )
    .unwrap();
    let params = ENNParams::new(2, 1.0, 0.1).unwrap();
    let query = array![[0.5, 0.5]];
    let out = model
        .posterior(&query.view(), &params, &PosteriorFlags::default())
        .unwrap();
    println!("mu={:?}", out.mu);
}
