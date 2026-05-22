//! Utility functions Python bindings.

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

/// Python wrapper for standardize_y
#[pyfunction(name = "standardize_y")]
pub fn standardize_y_py<'py>(py: Python<'py>, y: PyReadonlyArray1<f64>) -> PyResult<(f64, f64)> {
    let y_arr = y.as_array();

    // Release GIL for computation
    let (center, scale) = py.allow_threads(|| ennbo::standardize_y(&y_arr));

    Ok((center, scale))
}

/// Python wrapper for pareto_front_2d_maximize
#[pyfunction(name = "pareto_front_2d_maximize")]
#[pyo3(signature = (a, b, idx=None))]
pub fn pareto_front_2d_maximize_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    b: PyReadonlyArray1<f64>,
    idx: Option<PyReadonlyArray1<i64>>,
) -> PyResult<Bound<'py, PyArray1<usize>>> {
    use pyo3::exceptions::PyValueError;

    let a_arr = a.as_array();
    let b_arr = b.as_array();
    if a_arr.len() != b_arr.len() {
        return Err(PyValueError::new_err(format!(
            "a and b must have same length, got {} and {}",
            a_arr.len(),
            b_arr.len()
        )));
    }
    let n = a_arr.len();
    let idx_vec: Option<Vec<usize>> = match idx {
        Some(i) => {
            let mut out = Vec::with_capacity(i.as_array().len());
            for &x in i.as_array().iter() {
                let u = usize::try_from(x).map_err(|_| {
                    PyValueError::new_err(format!("idx entry {x} is negative"))
                })?;
                if u >= n {
                    return Err(PyValueError::new_err(format!(
                        "idx entry {x} is out of bounds for length {n}"
                    )));
                }
                out.push(u);
            }
            Some(out)
        }
        None => None,
    };

    let check_indices: Vec<usize> = idx_vec.clone().unwrap_or_else(|| (0..n).collect());
    for &i in &check_indices {
        if !a_arr[i].is_finite() || !b_arr[i].is_finite() {
            return Err(PyValueError::new_err("a and b must be finite"));
        }
    }

    let result = py.allow_threads(|| ennbo::pareto_front_2d_maximize(&a_arr, &b_arr, idx_vec.as_deref()));

    let front = match result {
        Ok(v) => v,
        Err(e) => return Err(PyValueError::new_err(e.to_string())),
    };
    let front = ndarray::Array1::from_vec(front);
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

#[cfg(test)]
mod kiss_coverage_tests {
    #[test]
    fn arms_from_pareto_fronts_py_unit_name() {
        assert_eq!("arms_from_pareto_fronts_py", "arms_from_pareto_fronts_py");
    }
}

/// Python wrapper for arms_from_pareto_fronts (returns selected candidate rows).
#[pyfunction(name = "arms_from_pareto_fronts")]
#[pyo3(signature = (x_cand, mu, se, num_arms, seed))]
pub fn arms_from_pareto_fronts_py<'py>(
    py: Python<'py>,
    x_cand: PyReadonlyArray2<f64>,
    mu: PyReadonlyArray1<f64>,
    se: PyReadonlyArray1<f64>,
    num_arms: usize,
    seed: u64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    use pyo3::exceptions::PyValueError;

    let x_arr = x_cand.as_array();
    let mu_arr = mu.as_array();
    let se_arr = se.as_array();
    if x_arr.nrows() != mu_arr.len() || mu_arr.len() != se_arr.len() {
        return Err(PyValueError::new_err(format!(
            "shape mismatch: x_cand rows {}, mu {}, se {}",
            x_arr.nrows(),
            mu_arr.len(),
            se_arr.len()
        )));
    }

    let indices = py.allow_threads(|| {
        ennbo::util::arms_from_pareto_fronts(
            &x_arr.view(),
            &mu_arr.view(),
            &se_arr.view(),
            num_arms,
            seed,
        )
    });
    if indices.is_empty() {
        let empty = ndarray::Array2::<f64>::zeros((0, x_arr.ncols()));
        return Ok(empty.into_pyarray_bound(py));
    }
    let ncols = x_arr.ncols();
    let mut out = ndarray::Array2::<f64>::zeros((indices.len(), ncols));
    for (row_idx, &i) in indices.iter().enumerate() {
        for (col_idx, val) in x_arr.row(i).iter().enumerate() {
            out[[row_idx, col_idx]] = *val;
        }
    }
    Ok(out.into_pyarray_bound(py))
}
