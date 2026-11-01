use crate::py_args::{text_list, value_error};
use ennbo::semantic_definition as definition;
use numpy::{IntoPyArray, PyArray2, PyArrayDyn, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

const VARIABLE_NAMES: &str = "variable names must be nonempty and unique";
const OUTPUT_NAMES: &str = "at least two unique nonempty output names are required";
const RELATIONSHIP_OUTPUTS: &str = "output names must be nonempty and unique";
const INPUT_NAMES: &str = "input names must match the problem definition";

#[pyfunction]
pub fn validate_semantic_definition(
    names: &Bound<'_, PyAny>,
    bounds: PyReadonlyArrayDyn<f64>,
    cutoff: f64,
    relative_weight: f64,
) -> PyResult<()> {
    definition::validate_input_definition(
        &text_list(names, VARIABLE_NAMES)?,
        &bounds.as_array(),
        cutoff,
        relative_weight,
    )
    .map_err(value_error)
}

#[pyfunction]
pub fn validate_semantic_cache(
    projection: PyReadonlyArrayDyn<f64>,
    variables: usize,
) -> PyResult<()> {
    definition::validate_cached_projection(&projection.as_array(), variables).map_err(value_error)
}

#[pyfunction]
pub fn validate_semantic_outputs(names: &Bound<'_, PyAny>, cutoff: f64) -> PyResult<()> {
    definition::validate_output_names(&text_list(names, OUTPUT_NAMES)?).map_err(value_error)?;
    definition::validate_cutoff(cutoff).map_err(value_error)
}

type OutputDefinition<'py> = (
    Vec<String>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
);

#[pyfunction]
pub fn semantic_output_definition<'py>(
    py: Python<'py>,
    names: &Bound<'py, PyAny>,
    embeddings: PyReadonlyArrayDyn<f64>,
    loadings: PyReadonlyArrayDyn<f64>,
) -> PyResult<OutputDefinition<'py>> {
    let names = text_list(names, OUTPUT_NAMES)?;
    definition::validate_output_names(&names).map_err(value_error)?;
    let unit = definition::normalized_embeddings(&embeddings.as_array(), names.len())
        .map_err(value_error)?;
    let loadings =
        definition::checked_loadings(&loadings.as_array(), names.len()).map_err(value_error)?;
    Ok((
        names,
        unit.into_pyarray_bound(py),
        loadings.into_pyarray_bound(py),
    ))
}

#[pyfunction]
pub fn semantic_output_embeddings<'py>(
    py: Python<'py>,
    encoded: PyReadonlyArrayDyn<f64>,
    texts: usize,
    rows: Vec<usize>,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    definition::output_embeddings(&encoded.as_array(), texts, &rows)
        .map(|v| v.into_pyarray_bound(py))
        .map_err(value_error)
}

#[pyfunction]
#[pyo3(signature = (names, output_names, descriptions=None, output_descriptions=None))]
pub fn semantic_relationship_texts(
    names: Vec<String>,
    output_names: &Bound<'_, PyAny>,
    descriptions: Option<Vec<Option<String>>>,
    output_descriptions: Option<Vec<Option<String>>>,
) -> PyResult<(Vec<String>, Vec<usize>)> {
    ennbo::semantic_relationships::texts(
        &names,
        &text_list(output_names, RELATIONSHIP_OUTPUTS)?,
        descriptions.as_deref(),
        output_descriptions.as_deref(),
    )
    .map_err(value_error)
}

#[pyfunction]
#[pyo3(signature = (names, descriptions=None))]
pub fn semantic_texts(
    names: Vec<String>,
    descriptions: Option<Vec<Option<String>>>,
) -> PyResult<(Vec<String>, Vec<usize>)> {
    ennbo::semantic_text::prepare(&names, descriptions.as_deref())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
pub fn semantic_encode<'py>(
    py: Python<'py>,
    texts: Vec<String>,
    model: String,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    py.allow_threads(|| ennbo::semantic_encoder::encode(&texts, &model))
        .map(|v| v.into_pyarray_bound(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
pub fn semantic_input_output_projection<'py>(
    py: Python<'py>,
    embeddings: PyReadonlyArrayDyn<f64>,
    rows: Vec<usize>,
    input_count: usize,
    cutoff: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let encoded = embeddings.as_array();
    let matrix = definition::embedding_matrix(&encoded).map_err(value_error)?;
    ennbo::semantic_relationships::projection(&matrix, &rows, input_count, cutoff)
        .map(|v| v.into_pyarray_bound(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
pub fn semantic_text_projection<'py>(
    py: Python<'py>,
    embeddings: PyReadonlyArrayDyn<f64>,
    rows: Vec<usize>,
    cutoff: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let encoded = embeddings.as_array();
    let matrix = definition::embedding_matrix(&encoded).map_err(value_error)?;
    ennbo::semantic_text::project(&matrix, &rows, cutoff)
        .map(|v| v.into_pyarray_bound(py))
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
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
#[pyo3(signature = (x, names, given_names, bounds, projection, only))]
pub fn semantic_transform<'py>(
    py: Python<'py>,
    x: PyReadonlyArrayDyn<f64>,
    names: Vec<String>,
    given_names: Option<&Bound<'py, PyAny>>,
    bounds: PyReadonlyArray2<f64>,
    projection: PyReadonlyArray2<f64>,
    only: bool,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let given = given_names
        .map(|given| text_list(given, INPUT_NAMES))
        .transpose()?;
    let (rows, single) =
        definition::ordered_rows(&x.as_array(), &names, given.as_deref()).map_err(value_error)?;
    let features = ennbo::semantic_projection::transform(
        &rows.view(),
        &bounds.as_array(),
        &projection.as_array().to_owned(),
        only,
    )
    .map_err(value_error)?;
    let out = if single {
        features.row(0).to_owned().into_dyn()
    } else {
        features.into_dyn()
    };
    Ok(out.into_pyarray_bound(py))
}
#[pyclass(name = "InputHistory")]
pub struct PyInputHistory {
    inner: ennbo::input_history::InputHistory,
}
#[pymethods]
impl PyInputHistory {
    #[new]
    fn new(
        rows: PyReadonlyArray2<f64>,
        model: PyRef<'_, crate::py_model::PyEpistemicNearestNeighbors>,
    ) -> PyResult<Self> {
        ennbo::input_history::InputHistory::for_model(rows.as_array().to_owned(), &model.inner)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    fn append(&mut self, rows: PyReadonlyArray2<f64>) -> PyResult<()> {
        self.inner
            .append(&rows.as_array())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
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

#[pyfunction]
pub fn semantic_output_projection<'py>(
    py: Python<'py>,
    names: &Bound<'py, PyAny>,
    embeddings: PyReadonlyArrayDyn<f64>,
    cutoff: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let names = text_list(names, OUTPUT_NAMES)?;
    definition::validate_output_names(&names).map_err(value_error)?;
    definition::validate_cutoff(cutoff).map_err(value_error)?;
    let unit = definition::normalized_embeddings(&embeddings.as_array(), names.len())
        .map_err(value_error)?;
    ennbo::semantic_projection::output_projection(&unit.view(), cutoff)
        .map(|v| v.into_pyarray_bound(py))
        .map_err(value_error)
}
