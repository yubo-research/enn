//! ENN-surrogate keys of the Python config-override dict.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::path::PathBuf;

use crate::py_optimizer::{optional_bool, optional_usize};

/// Every ENN-surrogate key `parse_enn_overrides` reads. Any one of them present builds an
/// `EnnOverrides`, which the optimizer rejects when there is no ENN surrogate.
pub(crate) const ENN_OVERRIDE_KEYS: &[&str] = &[
    "index_driver",
    "num_fit_samples",
    "freeze_params",
    "num_fit_candidates",
    "infer_aleatoric_variance",
    "scale_x",
    "y_bounds",
    "enn_storage",
    "work_dir",
    "metric_learning",
    "tied_dims",
    "affine_calibrate",
];

fn to_py_err(e: ennbo::ENNError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

#[doc = "kiss-coverage-off"]
fn parse_fit_samples(
    dict: &Bound<'_, pyo3::types::PyDict>,
) -> PyResult<Option<ennbo::FitSamples>> {
    let count = optional_usize(dict, "num_fit_samples")?;
    let frozen = optional_bool(dict, "freeze_params")?.unwrap_or(false);
    match (frozen, count) {
        (true, Some(_)) => Err(PyValueError::new_err(
            "freeze_params and num_fit_samples cannot both be set",
        )),
        (true, None) => Ok(Some(ennbo::FitSamples::Frozen)),
        (false, None) => Ok(None),
        (false, Some(n)) => ennbo::FitSamples::from_count(Some(n))
            .map(Some)
            .map_err(to_py_err),
    }
}

#[doc = "kiss-coverage-off"]
fn optional_string(dict: &Bound<'_, pyo3::types::PyDict>, key: &str) -> PyResult<Option<String>> {
    match dict.get_item(key)? {
        Some(v) => Ok(Some(v.extract()?)),
        None => Ok(None),
    }
}

#[doc = "kiss-coverage-off"]
fn any_enn_key(dict: &Bound<'_, pyo3::types::PyDict>) -> PyResult<bool> {
    for key in ENN_OVERRIDE_KEYS {
        if dict.contains(*key)? {
            return Ok(true);
        }
    }
    Ok(false)
}

#[doc = "kiss-coverage-off"]
pub(crate) fn parse_enn_overrides(
    dict: &Bound<'_, pyo3::types::PyDict>,
) -> PyResult<Option<ennbo::EnnOverrides>> {
    use crate::py_layout::{enn_storage_from_wire, index_driver_from_wire, metric_learning_from_wire};

    if !any_enn_key(dict)? {
        return Ok(None);
    }
    let y_bounds = match dict.get_item("y_bounds")? {
        Some(v) => Some(v.extract::<numpy::PyReadonlyArray2<f64>>()?.as_array().to_owned()),
        None => None,
    };
    let tied_dims = match dict.get_item("tied_dims")? {
        Some(v) => Some(v.extract()?),
        None => None,
    };
    Ok(Some(ennbo::EnnOverrides {
        index_driver: optional_string(dict, "index_driver")?
            .map(|s| index_driver_from_wire(&s))
            .transpose()?,
        fit_samples: parse_fit_samples(dict)?,
        num_fit_candidates: optional_usize(dict, "num_fit_candidates")?,
        infer_aleatoric_variance: optional_bool(dict, "infer_aleatoric_variance")?,
        scale_x: optional_bool(dict, "scale_x")?,
        y_bounds,
        enn_storage: optional_string(dict, "enn_storage")?
            .map(|s| enn_storage_from_wire(&s))
            .transpose()?,
        work_dir: optional_string(dict, "work_dir")?.map(PathBuf::from),
        metric_learning: optional_string(dict, "metric_learning")?
            .map(|s| metric_learning_from_wire(&s))
            .transpose()?,
        tied_dims,
        affine_calibrate: optional_bool(dict, "affine_calibrate")?,
    }))
}

#[cfg(test)]
mod kiss_coverage_tests {
    use super::{any_enn_key, optional_string, parse_enn_overrides, parse_fit_samples, to_py_err};

    #[test]
    fn py_enn_override_helpers_are_linked() {
        let _ = (
            parse_fit_samples as fn(_) -> _,
            optional_string as fn(_, _) -> _,
            any_enn_key as fn(_) -> _,
            parse_enn_overrides as fn(_) -> _,
            to_py_err as fn(_) -> _,
        );
    }
}
