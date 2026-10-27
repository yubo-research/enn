//! Placement parsing delegates policy to Rust's backend-specific layout.
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::path::PathBuf;

#[doc = "kiss-coverage-off"]
pub(crate) fn index_driver_from_wire(name: &str) -> PyResult<ennbo::IndexDriver> {
    ennbo::IndexDriver::from_wire(name)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown index_driver: {name}")))
}

#[pyfunction(name = "validate_enn_placement", signature = (index_driver, work_dir, weighting))]
#[doc = "kiss-coverage-off"]
pub fn validate_enn_placement_py(
    index_driver: &str,
    work_dir: Option<String>,
    weighting: bool,
) -> PyResult<()> {
    ennbo::EnnLayout::new(
        index_driver_from_wire(index_driver)?,
        work_dir.map(PathBuf::from),
        weighting,
    )
    .map(|_| ())
    .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (carried, explicit, semantic_only))]
#[doc = "kiss-coverage-off"]
pub fn resolve_tied_groups(
    carried: Vec<Vec<usize>>,
    explicit: Option<Vec<Vec<usize>>>,
    semantic_only: bool,
) -> PyResult<Vec<Vec<usize>>> {
    ennbo::metric_weights::resolve_tied_groups(&carried, explicit.as_deref(), semantic_only)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}
