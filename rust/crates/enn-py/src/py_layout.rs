use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

#[pyfunction(name = "validate_enn_placement", signature = (work_dir))]
pub fn validate_enn_placement_py(work_dir: String) -> PyResult<()> {
    ennbo::EnnLayout::new(work_dir, false)
        .map(|_| ())
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction]
#[pyo3(signature = (carried, explicit, semantic_only))]
pub fn resolve_tied_groups(
    carried: Vec<Vec<usize>>,
    explicit: Option<Vec<Vec<usize>>>,
    semantic_only: bool,
) -> PyResult<Vec<Vec<usize>>> {
    ennbo::metric_weights::resolve_tied_groups(&carried, explicit.as_deref(), semantic_only)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}
