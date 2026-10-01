//! Placement checks. The rule itself lives in `EnnLayout::try_from_parts`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::path::PathBuf;

#[pyfunction(name = "validate_enn_placement")]
#[pyo3(signature = (index_driver, enn_storage, work_dir, scale_x, metric_learning))]
#[doc = "kiss-coverage-off"]
pub fn validate_enn_placement_py(
    index_driver: &str,
    enn_storage: Option<&str>,
    work_dir: Option<String>,
    scale_x: bool,
    metric_learning: &str,
) -> PyResult<()> {
    use ennbo::index::IndexDriver;
    use ennbo::metric_auto::MetricLearning;
    use ennbo::EnnStorage;
    let driver = match index_driver {
        "FLAT" => IndexDriver::Flat,
        "BPANN_DISK" => IndexDriver::BpAnnDisk,
        other => {
            return Err(PyValueError::new_err(format!(
                "Unknown index_driver: {other}"
            )))
        }
    };
    let storage = match enn_storage {
        None => None,
        Some("DISK") => Some(EnnStorage::Disk),
        Some("MEMORY") => Some(EnnStorage::InMemory),
        Some(other) => {
            return Err(PyValueError::new_err(format!(
                "Unknown enn_storage: {other}"
            )))
        }
    };
    let metric = MetricLearning::parse(metric_learning).ok_or_else(|| {
        PyValueError::new_err(format!(
            "metric_learning must be 'none' or 'auto', got {metric_learning}"
        ))
    })?;
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
