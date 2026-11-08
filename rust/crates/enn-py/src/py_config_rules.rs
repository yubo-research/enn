use crate::py_args::value_error;
use ennbo::config::{
    validate_candidate_counts, validate_fit_settings, validate_multi_objective,
    validate_positive_option, FitSettings,
};
use pyo3::prelude::*;

#[pyfunction(name = "validate_candidate_counts")]
pub fn validate_candidate_counts_py(
    min_candidates: f64,
    max_candidates: f64,
    num_candidates_per_dim: f64,
    num_candidates_per_arm: f64,
) -> PyResult<()> {
    validate_candidate_counts(&[
        ("min_candidates", min_candidates),
        ("max_candidates", max_candidates),
        ("num_candidates_per_dim", num_candidates_per_dim),
        ("num_candidates_per_arm", num_candidates_per_arm),
    ])
    .map_err(value_error)
}

#[pyfunction(name = "validate_fit_settings")]
#[pyo3(signature = (num_fit_samples, num_fit_candidates, infer_aleatoric_variance_scale, affine_calibrate))]
pub fn validate_fit_settings_py(
    num_fit_samples: Option<f64>,
    num_fit_candidates: Option<f64>,
    infer_aleatoric_variance_scale: Option<bool>,
    affine_calibrate: bool,
) -> PyResult<()> {
    validate_fit_settings(&FitSettings {
        num_fit_samples,
        num_fit_candidates,
        infer_aleatoric_variance_scale,
        affine_calibrate,
    })
    .map_err(value_error)
}

#[pyfunction(name = "validate_num_init")]
#[pyo3(signature = (num_init))]
pub fn validate_num_init_py(num_init: Option<f64>) -> PyResult<()> {
    validate_positive_option("num_init", num_init).map_err(value_error)
}

#[pyfunction(name = "validate_multi_objective")]
pub fn validate_multi_objective_py(num_metrics: f64, alpha: f64) -> PyResult<()> {
    validate_multi_objective(num_metrics, alpha).map_err(value_error)
}

#[pyfunction(name = "validate_length_overrides")]
#[pyo3(signature = (length_init, length_min, length_max))]
pub fn validate_length_overrides_py(
    length_init: Option<f64>,
    length_min: Option<f64>,
    length_max: Option<f64>,
) -> PyResult<()> {
    ennbo::trust_region::validate_length_overrides(length_init, length_min, length_max)
        .map_err(value_error)
}
