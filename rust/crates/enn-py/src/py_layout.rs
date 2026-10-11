use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::path::PathBuf;

pub(crate) fn index_driver_from_wire(name: &str) -> PyResult<ennbo::IndexDriver> {
    ennbo::IndexDriver::from_wire(name)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown index_driver: {name}")))
}

pub(crate) fn enn_storage_from_wire(name: &str) -> PyResult<ennbo::EnnStorage> {
    ennbo::EnnStorage::from_wire(name)
        .ok_or_else(|| PyValueError::new_err(format!("Unknown enn_storage: {name}")))
}

pub(crate) fn metric_learning_from_wire(
    name: &str,
) -> PyResult<ennbo::metric_auto::MetricLearning> {
    ennbo::metric_auto::MetricLearning::parse(name).ok_or_else(|| {
        PyValueError::new_err(format!(
            "metric_learning must be 'NONE' or 'AUTO', got {name}"
        ))
    })
}

pub(crate) fn enn_storage_optional(name: Option<&str>) -> PyResult<Option<ennbo::EnnStorage>> {
    match name {
        None => Ok(None),
        Some(name) => Ok(Some(enn_storage_from_wire(name)?)),
    }
}

#[pyfunction(name = "validate_enn_placement")]
#[pyo3(signature = (index_driver, enn_storage, work_dir, scale_x, metric_learning))]
pub fn validate_enn_placement_py(
    index_driver: &str,
    enn_storage: Option<&str>,
    work_dir: Option<String>,
    scale_x: bool,
    metric_learning: &str,
) -> PyResult<()> {
    let driver = index_driver_from_wire(index_driver)?;
    let storage = enn_storage_optional(enn_storage)?;
    let metric = metric_learning_from_wire(metric_learning)?;
    ennbo::EnnLayout::try_from_parts(
        driver,
        storage,
        work_dir.map(PathBuf::from),
        scale_x,
        metric,
    )
    .map(|_| ())
    .map_err(|e| PyValueError::new_err(e.to_string()))
}
