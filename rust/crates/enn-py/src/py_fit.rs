//! Parameter fitting Python bindings.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::py_model::PyEpistemicNearestNeighbors;

/// Python wrapper for subsample_loglik
#[allow(clippy::too_many_arguments)]
#[pyfunction(name = "subsample_loglik")]
#[pyo3(signature = (model, x, y, k_values, epistemic_scales, aleatoric_scales, p, seed, y_std=None))]
pub fn subsample_loglik_py(
    model: &PyEpistemicNearestNeighbors,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray2<f64>,
    k_values: Vec<i32>,
    epistemic_scales: Vec<f64>,
    aleatoric_scales: Vec<f64>,
    p: usize,
    seed: u64,
    y_std: Option<PyReadonlyArray1<f64>>,
) -> PyResult<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);

    // Build params list
    let n_params = k_values.len();
    if epistemic_scales.len() != n_params || aleatoric_scales.len() != n_params {
        return Err(PyValueError::new_err(
            "k_values, epistemic_scales, and aleatoric_scales must have same length",
        ));
    }

    let mut paramss = Vec::with_capacity(n_params);
    for i in 0..n_params {
        let params = ennbo::ENNParams::new(k_values[i], epistemic_scales[i], aleatoric_scales[i])
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        paramss.push(params);
    }

    let y_std_arr = y_std.as_ref().map(|v| v.as_array());

    let result = ennbo::subsample_loglik(
        &model.inner,
        &x.as_array(),
        &y.as_array(),
        &paramss,
        p,
        &mut rng,
        y_std_arr.as_ref(),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result)
}
