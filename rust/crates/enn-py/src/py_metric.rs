//! Python bindings for AUTO metric state on an ENN model.

use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::py_model::PyEpistemicNearestNeighbors;

#[pyclass]
#[doc = "kiss-coverage-off"]
pub struct MetricSnapshot {
    #[pyo3(get)]
    pub num_seen: usize,
    #[pyo3(get)]
    pub num_refits: usize,
    #[pyo3(get)]
    pub num_rescales: usize,
    #[pyo3(get)]
    pub num_rebuilds: usize,
    #[pyo3(get)]
    pub heldout_gain: Option<f64>,
    #[pyo3(get)]
    pub uses_learned_metric: bool,
    #[pyo3(get)]
    pub weights: Vec<f64>,
    #[pyo3(get)]
    pub built: Vec<f64>,
}

#[pyfunction]
#[doc = "kiss-coverage-off"]
pub fn metric_snapshot(model: PyRef<'_, PyEpistemicNearestNeighbors>) -> Option<MetricSnapshot> {
    model.inner.metric_snapshot().map(|snap| MetricSnapshot {
        num_seen: snap.num_seen,
        num_refits: snap.counters.num_refits,
        num_rescales: snap.counters.num_rescales,
        num_rebuilds: snap.counters.num_rebuilds,
        heldout_gain: snap.heldout_gain,
        uses_learned_metric: snap.uses_learned,
        weights: snap.weights,
        built: snap.built,
    })
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
#[pyo3(signature = (x, y, tied=None, floor=None))]
#[doc = "kiss-coverage-off"]
pub fn dependence_weights<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray2<f64>,
    tied: Option<Vec<Vec<usize>>>,
    floor: Option<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x = x.as_array();
    let y = y.as_array();
    let (xf, yf, n, d, m) = flat_xy(&x.to_owned(), &y.to_owned());
    let floor = floor.unwrap_or(ennbo::metric_weights::DEPENDENCE_FLOOR);
    let w = ennbo::metric_weights::dependence_weights(
        &xf,
        n,
        d,
        &yf,
        m,
        &tied.unwrap_or_default(),
        floor,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(PyArray1::from_vec_bound(py, w))
}

#[pyfunction]
#[pyo3(signature = (x, y, k, tied=None))]
#[doc = "kiss-coverage-off"]
pub fn auto_weights<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray2<f64>,
    k: usize,
    tied: Option<Vec<Vec<usize>>>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64)> {
    let x = x.as_array().to_owned();
    let y = y.as_array().to_owned();
    let (xf, yf, n, d, m) = flat_xy(&x, &y);
    let (w, gain) =
        ennbo::metric_auto::auto_weights(&xf, n, d, &yf, m, k, &tied.unwrap_or_default())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((PyArray1::from_vec_bound(py, w), gain))
}

#[pyfunction]
#[pyo3(signature = (x, y, num_bins=None))]
#[doc = "kiss-coverage-off"]
pub fn sobol_index<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
    num_bins: Option<usize>,
) -> Bound<'py, PyArray1<f64>> {
    let x = x.as_array();
    let y = y.as_array();
    let xf: Vec<f64> = x.iter().copied().collect();
    let yf: Vec<f64> = y.iter().copied().collect();
    let s = ennbo::metric_sobol::sobol_index(&xf, x.nrows(), x.ncols(), &yf, num_bins);
    PyArray1::from_vec_bound(py, s)
}

#[pyfunction]
#[doc = "kiss-coverage-off"]
pub fn weight_drift(weights: PyReadonlyArray1<f64>, built: PyReadonlyArray1<f64>) -> PyResult<f64> {
    let w: Vec<f64> = weights.as_array().iter().copied().collect();
    let b: Vec<f64> = built.as_array().iter().copied().collect();
    if w.len() != b.len() {
        return Err(PyValueError::new_err(format!(
            "weights length {} != built length {}",
            w.len(),
            b.len()
        )));
    }
    Ok(ennbo::metric_auto::weight_drift(&w, &b))
}

#[pyfunction]
#[doc = "kiss-coverage-off"]
pub fn auto_uses_learned_metric(heldout_gain: f64) -> bool {
    ennbo::metric_auto::auto_uses_learned_metric(heldout_gain)
}

#[pyfunction]
#[doc = "kiss-coverage-off"]
pub fn metric_tied(model: PyRef<'_, PyEpistemicNearestNeighbors>) -> PyResult<Vec<Vec<usize>>> {
    model
        .inner
        .metric_tied()
        .map(|groups| groups.to_vec())
        .ok_or_else(|| PyValueError::new_err("tied_dims require metric_learning=AUTO"))
}

#[pyfunction]
#[pyo3(signature = (model, refit_growth=None, rebuild_drift=None, seed=None, reservoir_capacity=None, tied_dims=None))]
#[doc = "kiss-coverage-off"]
pub fn metric_configure(
    mut model: PyRefMut<'_, PyEpistemicNearestNeighbors>,
    refit_growth: Option<f64>,
    rebuild_drift: Option<f64>,
    seed: Option<i64>,
    reservoir_capacity: Option<usize>,
    tied_dims: Option<Vec<Vec<usize>>>,
) -> PyResult<()> {
    if let Some(seed) = seed {
        if seed < 0 {
            return Err(PyValueError::new_err(format!("seed must be >= 0, got {seed}")));
        }
    }
    if let Some(groups) = &tied_dims {
        let stored = model.inner.metric_tied().map(|g| g.to_vec()).unwrap_or_default();
        if groups != &stored {
            return Err(PyValueError::new_err(format!(
                "tied_dims {groups:?} do not match the model groups {stored:?}"
            )));
        }
    }
    let seed = seed.unwrap_or(0) as u64;
    let refit_growth = refit_growth.unwrap_or(ennbo::metric_auto::AUTO_REFIT_GROWTH);
    let rebuild_drift = rebuild_drift.unwrap_or(ennbo::metric_auto::DEFAULT_REBUILD_DRIFT);
    let reservoir_capacity =
        reservoir_capacity.unwrap_or(ennbo::metric_auto::AUTO_RESERVOIR_CAPACITY);
    model
        .inner
        .metric_configure(refit_growth, rebuild_drift, seed, reservoir_capacity)
        .map_err(|e| PyValueError::new_err(e.to_string()))
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
pub fn loo_loglik(
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray2<f64>,
    a: PyReadonlyArray1<f64>,
    k: usize,
) -> f64 {
    let x = x.as_array();
    let y = y.as_array();
    let (xf, yf, n, d, m) = flat_xy(&x.to_owned(), &y.to_owned());
    let weights: Vec<f64> = a.as_array().iter().copied().collect();
    ennbo::metric_loo::loo_loglik(&xf, n, d, &yf, m, &weights, k)
}

#[pyfunction]
#[doc = "kiss-coverage-off"]
pub fn validate_tied_dims(tied: Vec<Vec<i64>>, num_dim: usize) -> PyResult<()> {
    ennbo::metric_weights::validate_tied_dims_signed(&tied, num_dim)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyclass]
#[doc = "kiss-coverage-off"]
pub struct PyReservoir {
    inner: ennbo::reservoir::RowReservoir,
}

#[pymethods]
impl PyReservoir {
    #[new]
    fn new(capacity: usize, num_dim: usize, seed: u64, num_outputs: usize) -> PyResult<Self> {
        let inner = ennbo::reservoir::RowReservoir::new(capacity, num_dim, num_outputs, seed)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    fn add(&mut self, x: PyReadonlyArray2<f64>, y: PyReadonlyArray2<f64>) -> PyResult<()> {
        self.inner
            .push_rows(&x.as_array(), &y.as_array())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[getter]
    fn num_seen(&self) -> usize {
        self.inner.num_seen()
    }

    #[getter]
    fn capacity(&self) -> usize {
        self.inner.capacity()
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn x<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let len = self.inner.len();
        let dim = self.inner.num_dim();
        Array2::from_shape_vec((len, dim), self.inner.x().to_vec())
            .expect("shape")
            .into_pyarray_bound(py)
    }

    fn y<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let len = self.inner.len();
        let outputs = self.inner.num_outputs();
        Array2::from_shape_vec((len, outputs), self.inner.y().to_vec())
            .expect("shape")
            .into_pyarray_bound(py)
    }
}
