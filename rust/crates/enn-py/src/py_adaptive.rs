
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray2};
use pyo3::prelude::*;

#[pyfunction]
pub fn small_flat_selection_means<'py>(
    py: Python<'py>,
    train_x: PyReadonlyArray2<f64>,
    train_y: PyReadonlyArray2<f64>,
    query: PyReadonlyArray2<f64>,
    neighbors: Vec<usize>,
) -> Option<Bound<'py, PyArray3<f64>>> {
    ennbo::posterior::small_flat_selection_means(
        &train_x.as_array(),
        &train_y.as_array(),
        &query.as_array(),
        &neighbors,
    )
    .map(|means| means.into_pyarray_bound(py))
}

type MarginalArrays<'py> = (
    Bound<'py, numpy::PyArray2<f64>>,
    Bound<'py, numpy::PyArray2<f64>>,
);

#[pyclass(name = "AdaptiveSemanticENN")]
pub struct PyAdaptiveSemanticENN {
    inner: ennbo::adaptive::AdaptiveSemanticENN,
}
#[pymethods]
impl PyAdaptiveSemanticENN {
    #[new]
    #[pyo3(signature = (bounds, input, output, capacity, seed, linear_mean, adaptation))]
    fn new(
        bounds: Option<PyReadonlyArray2<f64>>,
        input: Option<PyReadonlyArray2<f64>>,
        output: Option<PyReadonlyArray2<f64>>,
        capacity: usize,
        seed: u64,
        linear_mean: bool,
        adaptation: String,
    ) -> PyResult<Self> {
        let metadata = ennbo::adaptive_basis::Metadata {
            bounds: bounds.map(|v| v.as_array().to_owned()),
            input: input.map(|v| v.as_array().to_owned()),
            output: output.map(|v| v.as_array().to_owned()),
        };
        ennbo::adaptive::AdaptiveSemanticENN::new(metadata, capacity, seed, linear_mean, adaptation)
            .map(|inner| Self { inner })
            .map_err(value_error)
    }
    #[pyo3(signature = (x, y, weight=None))]
    fn add(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray2<f64>,
        weight: Option<f64>,
    ) -> PyResult<()> {
        let (x, y) = (x.as_array().to_owned(), y.as_array().to_owned());
        py.allow_threads(|| self.inner.add(x, y, weight))
            .map_err(value_error)
    }
    fn posterior<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<MarginalArrays<'py>> {
        let x = x.as_array().to_owned();
        py.allow_threads(|| self.inner.posterior(&x))
            .map(|(mu, se)| (mu.into_pyarray_bound(py), se.into_pyarray_bound(py)))
            .map_err(value_error)
    }
    #[getter]
    fn choice(&self) -> Option<(String, f64, String, usize, f64)> {
        self.inner.choice().map(|c| {
            (
                c.geometry.clone(),
                c.weight,
                c.output.clone(),
                c.neighbors,
                c.alpha,
            )
        })
    }
    #[getter]
    fn num_seen(&self) -> usize {
        self.inner.num_seen
    }
    #[getter]
    fn num_refits(&self) -> usize {
        self.inner.num_refits
    }
    #[getter]
    fn num_retained(&self) -> usize {
        self.inner.num_retained()
    }
}
fn value_error(error: ennbo::error::ENNError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(error.to_string())
}
