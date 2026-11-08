use ennbo::categorical_plan::{CatEncoding, CatOptions};
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn encoding(name: &str) -> PyResult<CatEncoding> {
    match name {
        "ONE_HOT" => Ok(CatEncoding::OneHot),
        "RADEMACHER" => Ok(CatEncoding::Rademacher),
        "AUTO" => Ok(CatEncoding::Auto),
        _ => Err(PyValueError::new_err("unknown categorical encoding")),
    }
}

#[pyfunction]
pub fn cat_column_plan(card: usize, name: &str, cap: usize) -> PyResult<(usize, &'static str)> {
    ennbo::categorical_plan::column_plan(card, encoding(name)?, cap)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyclass(name = "CatEncoder")]
pub struct PyCatEncoder {
    #[pyo3(get)]
    num_dim: usize,
    inner: ennbo::categorical_encoder::CatEncoder,
}

#[pymethods]
impl PyCatEncoder {
    #[new]
    #[pyo3(signature = (kinds, cards, name, code_dim, hash_dim, seed, dim_cap, separation))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        kinds: Vec<bool>,
        cards: Option<Vec<Option<usize>>>,
        name: &str,
        code_dim: Option<usize>,
        hash_dim: Option<usize>,
        seed: i128,
        dim_cap: usize,
        separation: f64,
    ) -> PyResult<Self> {
        let seed = u64::try_from(seed)
            .map_err(|_| PyValueError::new_err("seed must be in the u64 range"))?;
        let options = CatOptions {
            encoding: encoding(name)?,
            code_dim,
            hash_dim,
            seed,
            dim_cap,
            separation,
        };
        ennbo::categorical_encoder::CatEncoder::new(kinds, cards, options)
            .map(|inner| Self {
                num_dim: inner.num_dim(),
                inner,
            })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    #[getter]
    fn styles(&self) -> Vec<&'static str> {
        self.inner
            .blocks
            .iter()
            .filter(|b| b.style != "continuous")
            .map(|b| b.style)
            .collect()
    }
    #[getter]
    fn code_dims(&self) -> Vec<usize> {
        self.inner
            .blocks
            .iter()
            .filter(|b| b.style != "continuous")
            .map(|b| b.width)
            .collect()
    }
    fn tied_dims(&self) -> Vec<Vec<usize>> {
        self.inner.tied_dims()
    }
    fn encode<'py>(
        &self,
        py: Python<'py>,
        x: &Bound<'_, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let encoded = if let Ok(values) = x.extract::<PyReadonlyArray2<i64>>() {
            self.inner.encode_i64(&values.as_array())
        } else if let Ok(values) = x.extract::<PyReadonlyArray2<f64>>() {
            self.inner.encode(&values.as_array())
        } else {
            self.inner
                .encode_mixed(&mixed_inputs(py, x, &self.inner.kinds)?.view())
        };
        encoded
            .map(|v| v.into_pyarray_bound(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
    fn snap<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        domains: Vec<Vec<i64>>,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        self.inner
            .snap(&x.as_array(), &domains)
            .map(|v| v.into_pyarray_bound(py))
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pyfunction]
pub fn cat_log_width(card: usize, cap: usize) -> PyResult<usize> {
    ennbo::categorical::log_code_width(card, cap).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn mixed_inputs(
    py: Python<'_>,
    values: &Bound<'_, PyAny>,
    kinds: &[bool],
) -> PyResult<ndarray::Array2<ennbo::categorical_encoder::InputValue>> {
    use ennbo::categorical_encoder::InputValue;
    let shape: (usize, usize) = values
        .getattr("shape")?
        .extract()
        .map_err(|_| PyValueError::new_err("x shape must be a matrix"))?;
    if shape.1 != kinds.len() {
        return Err(PyValueError::new_err("x shape does not match kinds"));
    }
    let rows: Vec<Vec<PyObject>> = values.call_method0("tolist")?.extract()?;
    let mut cells = Vec::with_capacity(shape.0 * shape.1);
    for row in rows {
        for (j, cell) in row.into_iter().enumerate() {
            let value = cell.bind(py);
            let cell = match (kinds[j], value.extract::<i64>()) {
                (true, Ok(label)) => InputValue::Category(label),
                _ => InputValue::Number(value.extract()?),
            };
            cells.push(cell);
        }
    }
    ndarray::Array2::from_shape_vec(shape, cells).map_err(|e| PyValueError::new_err(e.to_string()))
}
