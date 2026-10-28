//! Array marshalling for semantic math implemented in ennbo.
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Resolve metadata and deduplicate descriptions in the native core.
#[pyfunction]
#[pyo3(signature = (names, descriptions=None))]
#[doc = "kiss-coverage-off"]
pub fn semantic_texts(
    names: Vec<String>,
    descriptions: Option<Vec<Option<String>>>,
) -> PyResult<(Vec<String>, Vec<usize>)> {
    ennbo::semantic_text::prepare(&names, descriptions.as_deref())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Release the GIL while loading and running the CPU sentence transformer.
#[pyfunction]
#[doc = "kiss-coverage-off"]
pub fn semantic_encode<'py>(
    py: Python<'py>,
    texts: Vec<String>,
    model: String,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    py.allow_threads(|| ennbo::semantic_encoder::encode(&texts, &model))
        .map(|v| v.into_pyarray_bound(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Construct shared input-output features from deduplicated text embeddings.
#[pyfunction]
#[doc = "kiss-coverage-off"]
pub fn semantic_input_output_projection<'py>(
    py: Python<'py>,
    embeddings: PyReadonlyArray2<f64>,
    rows: Vec<usize>,
    input_count: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    ennbo::semantic_relationships::projection(&embeddings.as_array(), &rows, input_count)
        .map(|v| v.into_pyarray_bound(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Restore original variable ordering before projection construction.
#[pyfunction]
#[doc = "kiss-coverage-off"]
pub fn semantic_text_projection<'py>(
    py: Python<'py>,
    embeddings: PyReadonlyArray2<f64>,
    rows: Vec<usize>,
    cutoff: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    ennbo::semantic_text::project(&embeddings.as_array(), &rows, cutoff)
        .map(|v| v.into_pyarray_bound(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[doc = "kiss-coverage-off"]
pub fn semantic_projection<'py>(
    py: Python<'py>,
    embeddings: PyReadonlyArray2<f64>,
    cutoff: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    ennbo::semantic_projection::projection(&embeddings.as_array(), cutoff)
        .map(|v| v.into_pyarray_bound(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}
#[pyfunction]
#[doc = "kiss-coverage-off"]
pub fn semantic_weighted<'py>(
    py: Python<'py>,
    projection: PyReadonlyArray2<f64>,
    weight: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    ennbo::semantic_projection::weighted(&projection.as_array(), weight)
        .map(|v| v.into_pyarray_bound(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}
#[pyfunction]
#[doc = "kiss-coverage-off"]
pub fn semantic_transform<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    bounds: PyReadonlyArray2<f64>,
    projection: PyReadonlyArray2<f64>,
    only: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    ennbo::semantic_projection::transform(
        &x.as_array(),
        &bounds.as_array(),
        &projection.as_array().to_owned(),
        only,
    )
    .map(|v| v.into_pyarray_bound(py))
    .map_err(|e| PyValueError::new_err(e.to_string()))
}
#[pyfunction]
#[doc = "kiss-coverage-off"]
pub fn semantic_inverse<'py>(
    py: Python<'py>,
    z: PyReadonlyArray2<f64>,
    bounds: PyReadonlyArray2<f64>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    ennbo::semantic_projection::inverse(&z.as_array(), &bounds.as_array())
        .map(|v| v.into_pyarray_bound(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Original input rows for noninvertible feature representations.
#[pyclass(name = "InputHistory")]
pub struct PyInputHistory {
    inner: ennbo::input_history::InputHistory,
}
#[pymethods]
impl PyInputHistory {
    #[new]
    #[doc = "kiss-coverage-off"]
    fn new(
        rows: PyReadonlyArray2<f64>,
        model: PyRef<'_, crate::py_model::PyEpistemicNearestNeighbors>,
    ) -> PyResult<Self> {
        ennbo::input_history::InputHistory::for_model(rows.as_array().to_owned(), &model.inner)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    #[doc = "kiss-coverage-off"]
    fn append(&mut self, rows: PyReadonlyArray2<f64>) -> PyResult<()> {
        self.inner
            .append(&rows.as_array())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    #[doc = "kiss-coverage-off"]
    fn gather<'py>(
        &self,
        py: Python<'py>,
        indices: Vec<usize>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        self.inner
            .gather(&indices)
            .map(|v| v.into_pyarray_bound(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}
