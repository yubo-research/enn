use ennbo::EpistemicNearestNeighbors;
use ndarray::array;

fn main() {
    let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let train_y = array![[0.0], [1.0], [1.0], [0.0]];
    let dir = tempfile::tempdir().unwrap();
    let mut model = EpistemicNearestNeighbors::new(
        train_x.clone(),
        train_y.clone(),
        None,
        ennbo::EnnLayout::new(dir.path(), false).unwrap(),
    )
    .unwrap();
    let err = model
        .enable_auto_metric(vec![], &train_x.view(), &train_y.view())
        .unwrap_err();
    println!("{err}");
}
