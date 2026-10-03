//! Python wrapper for `ENNParams`, split out of `py_model` for file-size limits.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Wrapper for ENNParams
#[pyclass(name = "ENNParams")]
#[derive(Clone, Copy)]
pub struct PyENNParams {
    pub(crate) inner: ennbo::ENNParams,
}

#[pymethods]
impl PyENNParams {
    #[new]
    #[pyo3(signature = (k_num_neighbors, epistemic_variance_scale, aleatoric_variance_scale))]
    #[doc = "kiss-coverage-off"]
    pub(crate) fn new(
        k_num_neighbors: i32,
        epistemic_variance_scale: f64,
        aleatoric_variance_scale: f64,
    ) -> PyResult<Self> {
        let inner = ennbo::ENNParams::new(
            k_num_neighbors,
            epistemic_variance_scale,
            aleatoric_variance_scale,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    #[doc = "kiss-coverage-off"]
    pub(crate) fn k_num_neighbors(&self) -> i32 {
        self.inner.k_num_neighbors
    }

    #[getter]
    #[doc = "kiss-coverage-off"]
    pub(crate) fn epistemic_variance_scale(&self) -> f64 {
        self.inner.epistemic_variance_scale
    }

    #[getter]
    #[doc = "kiss-coverage-off"]
    pub(crate) fn aleatoric_variance_scale(&self) -> f64 {
        self.inner.aleatoric_variance_scale
    }

    #[doc = "kiss-coverage-off"]
    fn __repr__(&self) -> String {
        format!(
            "ENNParams(k={}, epi={:.4}, ale={:.4})",
            self.inner.k_num_neighbors,
            self.inner.epistemic_variance_scale,
            self.inner.aleatoric_variance_scale
        )
    }
}
