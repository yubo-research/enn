//! Bindings for calibrator, normal intervals, and benchmarks.

use ndarray::{Array2, ArrayD};
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn as_2d(arr: &ArrayD<f64>) -> Array2<f64> {
    let rows = if arr.ndim() == 1 { arr.len() } else { arr.shape()[0] };
    let cols = if arr.ndim() == 1 { 1 } else { arr.shape()[1] };
    Array2::from_shape_vec((rows, cols), arr.iter().copied().collect()).expect("2d")
}

#[pyfunction]
#[pyo3(signature = (mu, y, se=None))]
#[allow(clippy::type_complexity)]
#[doc = "kiss-coverage-off"]
pub fn fit_affine<'py>(
    py: Python<'py>,
    mu: PyReadonlyArray2<f64>,
    y: PyReadonlyArray2<f64>,
    se: Option<PyReadonlyArray2<f64>>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let se_owned = se.as_ref().map(|v| v.as_array().to_owned());
    let cal = ennbo::calibration::AffineCalibrator::fit(
        mu.as_array().view(),
        y.as_array().view(),
        se_owned.as_ref().map(|v| v.view()),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        cal.a.into_pyarray_bound(py),
        cal.b.into_pyarray_bound(py),
        cal.c.into_pyarray_bound(py),
    ))
}

#[pyfunction]
#[pyo3(signature = (mu, se, num_samples, seed, y_bounds=None, clip=None))]
#[doc = "kiss-coverage-off"]
pub fn sample_normal<'py>(
    py: Python<'py>,
    mu: PyReadonlyArray2<f64>,
    se: PyReadonlyArray2<f64>,
    num_samples: usize,
    seed: u64,
    y_bounds: Option<PyReadonlyArray2<f64>>,
    clip: Option<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let mu_d = mu.as_array().to_owned().into_dyn();
    let se_d = se.as_array().to_owned().into_dyn();
    let bounds = y_bounds.as_ref().map(|v| v.as_array().to_owned());
    let out = ennbo::normal_sample::sample_normal(&mu_d, &se_d, bounds.as_ref(), num_samples, seed, clip)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let rows = mu.as_array().nrows();
    let cols = mu.as_array().ncols();
    let flat: Vec<f64> = out.iter().copied().collect();
    let arr = Array2::from_shape_vec((rows, cols * num_samples), flat)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray_bound(py))
}

#[pyfunction]
#[doc = "kiss-coverage-off"]
pub fn z_crit(level: f64) -> PyResult<f64> {
    ennbo::normal_sample::z_crit(level).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (mu, se, level, y_bounds=None))]
#[allow(clippy::type_complexity)]
#[doc = "kiss-coverage-off"]
pub fn confidence_interval<'py>(
    py: Python<'py>,
    mu: PyReadonlyArray2<f64>,
    se: PyReadonlyArray2<f64>,
    level: f64,
    y_bounds: Option<PyReadonlyArray2<f64>>,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let mu_d = mu.as_array().to_owned().into_dyn();
    let se_d = se.as_array().to_owned().into_dyn();
    let bounds = y_bounds.as_ref().map(|v| v.as_array().to_owned());
    let (lo, hi) = ennbo::normal_sample::confidence_interval(&mu_d, &se_d, bounds.as_ref(), level)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((as_2d(&lo).into_pyarray_bound(py), as_2d(&hi).into_pyarray_bound(py)))
}

#[pyfunction]
#[doc = "kiss-coverage-off"]
pub fn ackley_core<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    a: f64,
    b: f64,
    c: f64,
) -> Bound<'py, PyArray1<f64>> {
    let y = ennbo::benchmarks::ackley_core(x.as_array().view(), a, b, c);
    y.into_pyarray_bound(py)
}

fn owned_bounds(bounds: &Option<PyReadonlyArray2<f64>>) -> Option<Array2<f64>> {
    bounds.as_ref().map(|v| v.as_array().to_owned())
}

fn vec_of(a: PyReadonlyArray1<f64>) -> Vec<f64> {
    a.as_array().iter().copied().collect()
}

#[pyfunction]
#[pyo3(signature = (a, b, mu, y_bounds=None, se=None))]
#[doc = "kiss-coverage-off"]
pub fn affine_map_mu<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    b: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray2<f64>,
    y_bounds: Option<PyReadonlyArray2<f64>>,
    se: Option<PyReadonlyArray2<f64>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let bounds = owned_bounds(&y_bounds);
    let se_owned = owned_bounds(&se);
    let out = ennbo::calibration::map_mu_with(
        &vec_of(a),
        &vec_of(b),
        mu.as_array().view(),
        bounds.as_ref(),
        se_owned.as_ref(),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (a, b, c, draws, mu, y_bounds=None, se=None))]
#[allow(clippy::too_many_arguments)]
#[doc = "kiss-coverage-off"]
pub fn affine_map_draws<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    b: PyReadonlyArray1<f64>,
    c: PyReadonlyArray1<f64>,
    draws: PyReadonlyArray2<f64>,
    mu: PyReadonlyArray2<f64>,
    y_bounds: Option<PyReadonlyArray2<f64>>,
    se: Option<PyReadonlyArray2<f64>>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let bounds = owned_bounds(&y_bounds);
    let se_owned = owned_bounds(&se);
    let out = ennbo::calibration::map_draws_with(
        &vec_of(a),
        &vec_of(b),
        &vec_of(c),
        draws.as_array().view(),
        mu.as_array().view(),
        bounds.as_ref(),
        se_owned.as_ref(),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(out.into_pyarray_bound(py))
}

#[pyfunction]
#[pyo3(signature = (a, b, c, mu, se_epi, se_ale, y_bounds=None))]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
#[doc = "kiss-coverage-off"]
pub fn affine_apply<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    b: PyReadonlyArray1<f64>,
    c: PyReadonlyArray1<f64>,
    mu: PyReadonlyArray2<f64>,
    se_epi: PyReadonlyArray2<f64>,
    se_ale: PyReadonlyArray2<f64>,
    y_bounds: Option<PyReadonlyArray2<f64>>,
) -> PyResult<(
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
)> {
    let bounds = owned_bounds(&y_bounds);
    let (mu_p, se, epi, ale) = ennbo::calibration::apply_normal(
        &vec_of(a),
        &vec_of(b),
        &vec_of(c),
        mu.as_array().view(),
        se_epi.as_array().view(),
        se_ale.as_array().view(),
        bounds.as_ref(),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        mu_p.into_pyarray_bound(py),
        se.into_pyarray_bound(py),
        epi.into_pyarray_bound(py),
        ale.into_pyarray_bound(py),
    ))
}

#[pyfunction]
#[doc = "kiss-coverage-off"]
pub fn choose_indices(n: usize, p: usize, seed: u64) -> PyResult<Vec<usize>> {
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    if p > n {
        return Err(PyValueError::new_err("p > n"));
    }
    let mut rng = StdRng::seed_from_u64(seed);
    let mut idx: Vec<usize> = (0..n).collect();
    for i in 0..p {
        let j = rng.gen_range(i..n);
        idx.swap(i, j);
    }
    idx.truncate(p);
    Ok(idx)
}

#[pyfunction]
#[doc = "kiss-coverage-off"]
pub fn separable_unimodal<'py>(py: Python<'py>, x: PyReadonlyArray2<f64>) -> Bound<'py, PyArray2<f64>> {
    ennbo::benchmarks::separable_unimodal(x.as_array().view()).into_pyarray_bound(py)
}
