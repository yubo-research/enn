use ndarray::{ArrayView2, Ix2};
use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

pub(crate) fn matrix_view<'a>(
    name: &str,
    arr: &'a PyReadonlyArrayDyn<'_, f64>,
) -> PyResult<ArrayView2<'a, f64>> {
    let view = arr.as_array();
    let shape = view.shape().to_vec();
    view.into_dimensionality::<Ix2>().map_err(|_| {
        PyValueError::new_err(format!(
            "{name} must be 2-dimensional, got shape {shape:?}"
        ))
    })
}

pub(crate) fn text_list(values: &Bound<'_, PyAny>, message: &str) -> PyResult<Vec<String>> {
    values
        .iter()?
        .map(|value| value?.extract::<String>().map_err(|_| value_error(message)))
        .collect()
}
