use ndarray::array;
use ennbo::AffineCalibrator;

fn main() {
    let rows = array![[1.0, 0.5], [2.0, 1.5], [0.1, 0.2]];
    let cal = AffineCalibrator::from_rows(rows.view()).unwrap();
    println!("a={:?}", cal.a);
}
