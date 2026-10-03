//! Parameter fitting Python bindings.

use numpy::{PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::py_model::{PyENNParams, PyEpistemicNearestNeighbors};

/// One-shot proof that `EpistemicNearestNeighbors.add` appended rows to one model.
#[pyclass(name = "ENNAddToken", frozen)]
#[derive(Clone, Copy)]
pub struct PyAddToken {
    pub(crate) inner: ennbo::AddToken,
}

/// Tell the rows of `token`'s `add` to the model's incremental fitter, then fit.
#[allow(clippy::too_many_arguments)]
#[pyfunction(name = "enn_fit_incremental")]
#[pyo3(signature = (model, token, k, seed, num_fit_candidates=None, num_fit_samples=None, params_warm_start=None))]
#[doc = "kiss-coverage-off"]
pub fn enn_fit_incremental_py(
    mut model: PyRefMut<'_, PyEpistemicNearestNeighbors>,
    token: &PyAddToken,
    k: i32,
    seed: u64,
    num_fit_candidates: Option<usize>,
    num_fit_samples: Option<usize>,
    params_warm_start: Option<PyENNParams>,
) -> PyResult<PyENNParams> {
    let opts = ennbo::IncrementalAsk {
        k,
        seed,
        num_fit_candidates,
        num_fit_samples,
        params_warm_start: params_warm_start.map(|p| p.inner),
    };
    let model = &mut *model;
    model
        .incremental
        .ask(&model.inner, token.inner, &opts)
        .map(|inner| PyENNParams { inner })
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Python wrapper for subsample_loglik
#[allow(clippy::too_many_arguments)]
#[pyfunction(name = "subsample_loglik")]
#[pyo3(signature = (model, x, y, k_values, epistemic_scales, aleatoric_scales, p=None, seed=0, y_std=None))]
#[doc = "kiss-coverage-off"]
pub fn subsample_loglik_py(
    model: &PyEpistemicNearestNeighbors,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray2<f64>,
    k_values: Vec<i32>,
    epistemic_scales: Vec<f64>,
    aleatoric_scales: Vec<f64>,
    p: Option<usize>,
    seed: u64,
    y_std: Option<PyReadonlyArray1<f64>>,
) -> PyResult<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(seed);


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

    let p = p.unwrap_or(ennbo::fit::DEFAULT_SUBSAMPLE_P);
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
