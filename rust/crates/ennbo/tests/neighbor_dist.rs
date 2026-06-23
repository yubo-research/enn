use ennbo::posterior::neighbor_dist;
use ndarray::array;

#[test]
fn posterior_row_sq_l2() {
    let x = array![2.0, 0.0];
    let y = array![0.0, 0.0];
    let scale = array![2.0, 1.0];
    let d = neighbor_dist::posterior_row_sq_l2(x.view(), y.view(), true, scale.view());
    assert!((d - 1.0).abs() < 1e-12);
}
