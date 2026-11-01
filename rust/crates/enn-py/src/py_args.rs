use ndarray::Array2;
use numpy::PyReadonlyArrayDyn;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBool;

pub(crate) fn value_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}

pub(crate) fn text_list(values: &Bound<'_, PyAny>, message: &str) -> PyResult<Vec<String>> {
    values
        .iter()?
        .map(|value| value?.extract::<String>().map_err(|_| value_error(message)))
        .collect()
}

pub(crate) fn matrix_or(
    values: &PyReadonlyArrayDyn<'_, f64>,
    message: &str,
) -> PyResult<Array2<f64>> {
    values
        .as_array()
        .to_owned()
        .into_dimensionality()
        .map_err(|_| value_error(message))
}

fn is_boolean(value: &Bound<'_, PyAny>) -> PyResult<bool> {
    let name = value.get_type().name()?;
    Ok(value.is_instance_of::<PyBool>() || name == "bool" || name == "bool_")
}

pub(crate) fn count_at_least(
    value: &Bound<'_, PyAny>,
    minimum: usize,
    message: &str,
) -> PyResult<usize> {
    if is_boolean(value)? {
        return Err(value_error(message));
    }
    match value.extract::<usize>() {
        Ok(count) if count >= minimum => Ok(count),
        _ => Err(value_error(message)),
    }
}

pub(crate) fn optional_finite_number(
    value: Option<&Bound<'_, PyAny>>,
    message: &str,
) -> PyResult<Option<f64>> {
    let Some(value) = value.filter(|v| !v.is_none()) else {
        return Ok(None);
    };
    if is_boolean(value)? {
        return Err(value_error(message));
    }
    let number = value.extract::<f64>().map_err(|_| value_error(message))?;
    Ok(Some(number))
}
