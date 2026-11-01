use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

pub(crate) fn text_list(values: &Bound<'_, PyAny>, message: &str) -> PyResult<Vec<String>> {
    values
        .iter()?
        .map(|value| value?.extract::<String>().map_err(|_| value_error(message)))
        .collect()
}
