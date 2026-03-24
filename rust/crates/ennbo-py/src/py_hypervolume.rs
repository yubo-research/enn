//! Hypervolume calculation Python bindings.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Python wrapper for hypervolume_2d_max
#[pyfunction(name = "hypervolume_2d_max")]
pub fn hypervolume_2d_max_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray2<f64>,
    ref_point: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let y_arr = y.as_array();
    let ref_arr = ref_point.as_array();

    // Release GIL for computation
    let result = py.allow_threads(|| ennbo::hypervolume_2d_max(&y_arr, &ref_arr));

    result.map_err(|e| PyValueError::new_err(e.to_string()))
}
