//! Index-free mean-only selection for the bounded semantic adaptation wrapper.

use numpy::{IntoPyArray, PyArray3, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyfunction]
#[doc = "kiss-coverage-off"]
pub fn small_flat_selection_means<'py>(
    py: Python<'py>,
    train_x: PyReadonlyArray2<f64>,
    train_y: PyReadonlyArray2<f64>,
    query: PyReadonlyArray2<f64>,
    neighbors: Vec<usize>,
) -> Option<Bound<'py, PyArray3<f64>>> {
    ennbo::posterior::small_flat_selection_means(
        &train_x.as_array(),
        &train_y.as_array(),
        &query.as_array(),
        &neighbors,
    )
    .map(|means| means.into_pyarray_bound(py))
}
