use ndarray::{Array1, Array2};

pub(crate) fn centered(x: &Array2<f64>, bounds: &Array2<f64>) -> Array2<f64> {
    Array2::from_shape_fn(x.dim(), |(row, column)| {
        (x[[row, column]] - bounds[[column, 0]]) / (bounds[[column, 1]] - bounds[[column, 0]]) - 0.5
    })
}

pub(crate) fn distance_scales(coefficients: &Array1<f64>, dimensions: usize) -> Array1<f64> {
    let mut scale = Array1::ones(dimensions + coefficients.len());
    for (index, &coefficient) in coefficients.iter().enumerate() {
        scale[dimensions + index] = 1.0 / coefficient.max(1e-12);
    }
    scale
}

pub(crate) fn bake(features: &mut Array2<f64>, coefficients: &Array1<f64>) {
    let offset = features.ncols() - coefficients.len();
    for mut row in features.rows_mut() {
        for (index, &coefficient) in coefficients.iter().enumerate() {
            row[offset + index] *= coefficient;
        }
    }
}
