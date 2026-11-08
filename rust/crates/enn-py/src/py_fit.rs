use numpy::{PyReadonlyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::py_args::matrix_view;
use crate::py_model::{params_list, PyEpistemicNearestNeighbors};

#[allow(clippy::too_many_arguments)]
#[pyfunction(name = "subsample_loglik")]
#[pyo3(signature = (model, x, y, k_values, epistemic_scales, aleatoric_scales, p=None, seed=0, y_std=None))]
pub fn subsample_loglik_py(
    model: &PyEpistemicNearestNeighbors,
    x: PyReadonlyArrayDyn<f64>,
    y: PyReadonlyArrayDyn<f64>,
    k_values: Vec<i32>,
    epistemic_scales: Vec<f64>,
    aleatoric_scales: Vec<f64>,
    p: Option<usize>,
    seed: u64,
    y_std: Option<PyReadonlyArray1<f64>>,
) -> PyResult<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);

    let paramss = params_list(k_values, epistemic_scales, aleatoric_scales)?;

    let y_std_arr = y_std.as_ref().map(|v| v.as_array());

    let p = p.unwrap_or(ennbo::fit::DEFAULT_SUBSAMPLE_P);
    let result = ennbo::subsample_loglik(
        &model.inner,
        &matrix_view("x", &x)?,
        &matrix_view("y", &y)?,
        &paramss,
        p,
        &mut rng,
        y_std_arr.as_ref(),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result)
}
