//! Hash-based RNG Python bindings.

use ndarray::IxDyn;
use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Python wrapper for normal_hash_batch_multi_seed_fast
#[pyfunction(name = "normal_hash_batch_multi_seed_fast")]
pub fn normal_hash_batch_multi_seed_fast_py<'py>(
    py: Python<'py>,
    function_seeds: PyReadonlyArray1<i64>,
    data_indices: PyReadonlyArrayDyn<i64>,
    num_metrics: i64,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let seeds_arr = function_seeds.as_array();
    let indices_arr = data_indices.as_array();

    let seeds: Vec<i64> = seeds_arr.iter().copied().collect();
    let indices: Vec<i64> = indices_arr.iter().copied().collect();
    let input_shape = indices_arr.shape().to_vec();
    let mut output_shape = Vec::with_capacity(2 + input_shape.len());
    output_shape.push(seeds.len());
    output_shape.extend(input_shape.iter().copied());
    output_shape.push(num_metrics.max(0) as usize);

    // Release GIL for computation
    let result = py.allow_threads(|| {
        ennbo::normal_hash_batch_multi_seed_fast(&seeds, &indices, num_metrics)
    });

    match result {
        Ok(arr) => {
            // Reshape from (num_seeds, flattened_indices, num_metrics)
            // to (num_seeds, *data_indices.shape, num_metrics) for API parity.
            let reshaped = arr
                .into_shape_with_order(IxDyn(&output_shape))
                .map_err(|e| PyValueError::new_err(format!("Shape error: {}", e)))?;
            Ok(reshaped.into_pyarray_bound(py))
        }
        Err(e) => Err(PyValueError::new_err(e.to_string())),
    }
}
