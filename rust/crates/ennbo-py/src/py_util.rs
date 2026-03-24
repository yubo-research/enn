//! Utility functions Python bindings.

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Python wrapper for standardize_y
#[pyfunction(name = "standardize_y")]
pub fn standardize_y_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
) -> PyResult<(f64, f64)> {
    let y_arr = y.as_array();

    // Release GIL for computation
    let (center, scale) = py.allow_threads(|| ennbo::standardize_y(&y_arr));

    Ok((center, scale))
}

/// Python wrapper for pareto_front_2d_maximize
#[pyfunction(name = "pareto_front_2d_maximize")]
pub fn pareto_front_2d_maximize_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    b: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<usize>>> {
    let a_arr = a.as_array();
    let b_arr = b.as_array();

    // Release GIL for computation
    let result = py.allow_threads(|| ennbo::pareto_front_2d_maximize(&a_arr, &b_arr, None));

    let front = ndarray::Array1::from_vec(result);
    Ok(front.into_pyarray_bound(py))
}

/// Python wrapper for calculate_sobol_indices
#[pyfunction(name = "calculate_sobol_indices")]
pub fn calculate_sobol_indices_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x_arr = x.as_array();
    let y_arr = y.as_array();

    let sobol = py.allow_threads(|| ennbo::calculate_sobol_indices(&x_arr, &y_arr));
    Ok(Array1::from_vec(sobol.to_vec()).into_pyarray_bound(py))
}

/// Python helper for deterministic Sobol sequence generation.
#[pyfunction(name = "sobol_sequence")]
#[pyo3(signature = (dimension, num_points, seed=0))]
pub fn sobol_sequence_py<'py>(
    py: Python<'py>,
    dimension: usize,
    num_points: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    use pyo3::exceptions::PyValueError;

    let mut engine = ennbo::candidates::SobolEngine::new(dimension)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = ndarray::Array2::zeros((num_points, dimension));
    for i in 0..num_points {
        let row = engine
            .sample(&mut rng)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        for j in 0..dimension {
            out[[i, j]] = row[j];
        }
    }
    Ok(out.into_dyn().into_pyarray_bound(py))
}
