//! Python bindings for AUTO metric state on an ENN model.

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::py_model::PyEpistemicNearestNeighbors;

#[pyfunction]
#[doc = "kiss-coverage-off"]
pub fn metric_weights<'py>(
    py: Python<'py>,
    model: PyRef<'_, PyEpistemicNearestNeighbors>,
) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
    Ok(model.inner.metric_weights().map(|w| w.into_pyarray_bound(py)))
}

#[pyfunction]
#[doc = "kiss-coverage-off"]
pub fn metric_built<'py>(
    py: Python<'py>,
    model: PyRef<'_, PyEpistemicNearestNeighbors>,
) -> PyResult<Option<Bound<'py, PyArray1<f64>>>> {
    Ok(model.inner.metric_built().map(|w| w.into_pyarray_bound(py)))
}

#[pyfunction]
#[doc = "kiss-coverage-off"]
pub fn metric_heldout_gain(model: PyRef<'_, PyEpistemicNearestNeighbors>) -> Option<f64> {
    model.inner.metric_heldout_gain()
}

#[pyfunction]
#[doc = "kiss-coverage-off"]
pub fn metric_num_seen(model: PyRef<'_, PyEpistemicNearestNeighbors>) -> usize {
    model.inner.metric_num_seen()
}

#[pyfunction]
#[doc = "kiss-coverage-off"]
pub fn metric_num_refits(model: PyRef<'_, PyEpistemicNearestNeighbors>) -> usize {
    model.inner.metric_num_refits()
}

#[pyfunction]
#[doc = "kiss-coverage-off"]
pub fn metric_num_rescales(model: PyRef<'_, PyEpistemicNearestNeighbors>) -> usize {
    model.inner.metric_num_rescales()
}

#[pyfunction]
#[doc = "kiss-coverage-off"]
pub fn metric_num_rebuilds(model: PyRef<'_, PyEpistemicNearestNeighbors>) -> usize {
    model.inner.metric_num_rebuilds()
}

#[pyfunction]
#[doc = "kiss-coverage-off"]
pub fn metric_uses_learned(model: PyRef<'_, PyEpistemicNearestNeighbors>) -> bool {
    model.inner.metric_uses_learned()
}

#[pyfunction]
#[pyo3(signature = (model, weights, rebuild_drift=None))]
#[doc = "kiss-coverage-off"]
pub fn metric_set_weights(
    mut model: PyRefMut<'_, PyEpistemicNearestNeighbors>,
    weights: PyReadonlyArray1<f64>,
    rebuild_drift: Option<f64>,
) -> PyResult<bool> {
    let w = weights.as_array();
    let flat: Vec<f64> = w.iter().copied().collect();
    model
        .inner
        .metric_set_weights(&flat, rebuild_drift)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

fn flat_xy(x: &Array2<f64>, y: &Array2<f64>) -> (Vec<f64>, Vec<f64>, usize, usize, usize) {
    (
        x.iter().copied().collect(),
        y.iter().copied().collect(),
        x.nrows(),
        x.ncols(),
        y.ncols(),
    )
}

#[pyfunction]
#[pyo3(signature = (x, y, tied=None))]
#[doc = "kiss-coverage-off"]
pub fn dependence_weights<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray2<f64>,
    tied: Option<Vec<Vec<usize>>>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x = x.as_array();
    let y = y.as_array();
    let (xf, yf, n, d, m) = flat_xy(&x.to_owned(), &y.to_owned());
    let w = ennbo::metric_weights::dependence_weights(&xf, n, d, &yf, m, &tied.unwrap_or_default());
    Ok(PyArray1::from_vec_bound(py, w))
}

#[pyfunction]
#[pyo3(signature = (x, y, tied=None))]
#[doc = "kiss-coverage-off"]
pub fn auto_weights<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray2<f64>,
    tied: Option<Vec<Vec<usize>>>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64)> {
    let x = x.as_array().to_owned();
    let y = y.as_array().to_owned();
    let (xf, yf, n, d, m) = flat_xy(&x, &y);
    let (w, gain) = ennbo::metric_auto::auto_weights(&xf, n, d, &yf, m, &tied.unwrap_or_default());
    Ok((PyArray1::from_vec_bound(py, w), gain))
}

#[pyfunction]
#[doc = "kiss-coverage-off"]
pub fn sobol_index<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
) -> Bound<'py, PyArray1<f64>> {
    let x = x.as_array();
    let y = y.as_array();
    let xf: Vec<f64> = x.iter().copied().collect();
    let yf: Vec<f64> = y.iter().copied().collect();
    let s = ennbo::metric_sobol::sobol_index(&xf, x.nrows(), x.ncols(), &yf, None);
    PyArray1::from_vec_bound(py, s)
}

#[pyfunction]
#[pyo3(signature = (n, num_cells=None))]
#[doc = "kiss-coverage-off"]
pub fn null_sd(n: usize, num_cells: Option<usize>) -> f64 {
    ennbo::metric_sobol::null_sd(n, num_cells)
}

#[pyfunction]
#[doc = "kiss-coverage-off"]
pub fn group_sobol_index(x: PyReadonlyArray2<f64>, y: PyReadonlyArray1<f64>) -> f64 {
    let x = x.as_array();
    let y = y.as_array();
    let xf: Vec<f64> = x.iter().copied().collect();
    let yf: Vec<f64> = y.iter().copied().collect();
    let group: Vec<usize> = (0..x.ncols()).collect();
    ennbo::metric_sobol::group_sobol_index(&xf, x.nrows(), x.ncols(), &group, &yf)
}

#[pyfunction]
#[doc = "kiss-coverage-off"]
pub fn loo_loglik(x: PyReadonlyArray2<f64>, y: PyReadonlyArray2<f64>, a: PyReadonlyArray1<f64>, k: usize) -> f64 {
    let x = x.as_array();
    let y = y.as_array();
    let (xf, yf, n, d, m) = flat_xy(&x.to_owned(), &y.to_owned());
    let weights: Vec<f64> = a.as_array().iter().copied().collect();
    ennbo::metric_loo::loo_loglik(&xf, n, d, &yf, m, &weights, k)
}

#[pyclass]
#[doc = "kiss-coverage-off"]
pub struct PyReservoir {
    dim: usize,
    outputs: usize,
    capacity: usize,
    rng: ennbo::numpy_pcg::NumpyPcg64,
    xs: Vec<f64>,
    ys: Vec<f64>,
    len: usize,
    num_seen: usize,
}

#[pymethods]
impl PyReservoir {
    #[new]
    fn new(capacity: usize, num_dim: usize, seed: u64, num_outputs: usize) -> PyResult<Self> {
        if capacity < 1 {
            return Err(PyValueError::new_err(format!("capacity must be >= 1, got {capacity}")));
        }
        Ok(Self {
            dim: num_dim,
            outputs: num_outputs,
            capacity,
            rng: ennbo::numpy_pcg::NumpyPcg64::from_seed(seed),
            xs: vec![0.0; capacity * num_dim],
            ys: vec![0.0; capacity * num_outputs],
            len: 0,
            num_seen: 0,
        })
    }

    fn add(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray2<f64>) -> PyResult<()> {
        let x = x.as_array();
        let y = y.as_array();
        if x.nrows() != y.nrows() {
            return Err(PyValueError::new_err(format!("x has {} rows but y has {}", x.nrows(), y.nrows())));
        }
        if y.ncols() != self.outputs {
            return Err(PyValueError::new_err(format!(
                "y has {} columns, expected {}",
                y.ncols(),
                self.outputs
            )));
        }
        for i in 0..x.nrows() {
            if self.len < self.capacity {
                let slot = self.len;
                for j in 0..self.dim {
                    self.xs[slot * self.dim + j] = x[[i, j]];
                }
                for j in 0..self.outputs {
                    self.ys[slot * self.outputs + j] = y[[i, j]];
                }
                self.len += 1;
            } else {
                let slot = self.rng.integers_high(self.num_seen as u64 + 1) as usize;
                if slot < self.capacity {
                    for j in 0..self.dim {
                        self.xs[slot * self.dim + j] = x[[i, j]];
                    }
                    for j in 0..self.outputs {
                        self.ys[slot * self.outputs + j] = y[[i, j]];
                    }
                }
            }
            self.num_seen += 1;
        }
        Ok(())
    }

    #[getter]
    fn num_seen(&self) -> usize {
        self.num_seen
    }

    #[getter]
    fn capacity(&self) -> usize {
        self.capacity
    }

    fn __len__(&self) -> usize {
        self.len
    }

    fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        Array2::from_shape_vec((self.len, self.dim), self.xs[..self.len * self.dim].to_vec())
            .expect("shape")
            .into_pyarray_bound(py)
    }

    fn y<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        Array2::from_shape_vec(
            (self.len, self.outputs),
            self.ys[..self.len * self.outputs].to_vec(),
        )
        .expect("shape")
        .into_pyarray_bound(py)
    }
}
