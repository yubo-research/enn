use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction(name = "hypervolume_2d_max")]
pub fn hypervolume_2d_max_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArrayDyn<f64>,
    ref_point: PyReadonlyArrayDyn<f64>,
) -> PyResult<f64> {
    let y_arr = y.as_array();
    let ref_arr = ref_point.as_array();

    let result =
        py.allow_threads(|| ennbo::hypervolume::hypervolume_2d_max_any_rank(&y_arr, &ref_arr));

    result.map_err(|e| PyValueError::new_err(e.to_string()))
}
