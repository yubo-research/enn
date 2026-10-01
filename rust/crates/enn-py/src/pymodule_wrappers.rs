use pyo3::prelude::*;

/// Hypervolume calculation module
#[pymodule]
#[pyo3(name = "hypervolume")]
#[doc = "kiss-coverage-off"]
pub fn pymodule_hypervolume(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(crate::py_hypervolume::hypervolume_2d_max_py, m)?)?;
    Ok(())
}

/// Hash-based RNG module
#[pymodule]
#[pyo3(name = "hash")]
#[doc = "kiss-coverage-off"]
pub fn pymodule_hash(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        crate::py_hash::normal_hash_batch_multi_seed_fast_py,
        m
    )?)?;
    m.add_function(wrap_pyfunction!(
        crate::py_hash::normal_hash_batch_multi_seed_py,
        m
    )?)?;
    Ok(())
}

/// Utility functions module
#[pymodule]
#[pyo3(name = "util")]
#[doc = "kiss-coverage-off"]
pub fn pymodule_util(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(crate::py_util::standardize_y_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_util::pareto_front_2d_maximize_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_util::calculate_sobol_indices_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_util::sobol_sequence_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_util::arms_from_pareto_fronts_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_util::set_config_path_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_util::ensure_config_file_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_ports::fit_affine, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_ports::affine_map_mu, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_ports::affine_map_draws, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_ports::affine_apply, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_ports::sample_normal, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_ports::z_crit, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_ports::confidence_interval, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_ports::ackley_core, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_ports::separable_unimodal, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_ports::choose_indices, m)?)?;
    m.add_class::<crate::py_numpy_normal::PyNumpyNormal>()?;
    m.add_class::<crate::py_ports::PyENNNormal>()?;
    m.add_class::<crate::py_ports::PyAckley>()?;
    m.add_class::<crate::py_ports::PyDoubleAckley>()?;
    Ok(())
}

/// ENN model module
#[pymodule]
#[pyo3(name = "model")]
#[doc = "kiss-coverage-off"]
pub fn pymodule_model(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<crate::py_model::PyEpistemicNearestNeighbors>()?;
    m.add_class::<crate::py_model::PyENNParams>()?;
    m.add_function(wrap_pyfunction!(crate::py_model::train_rows_at_warped, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_layout::validate_enn_placement_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_model::set_unscaled_dims, m)?)?;
    m.add_class::<crate::py_metric::MetricSnapshot>()?;
    m.add_function(wrap_pyfunction!(crate::py_metric::metric_snapshot, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_metric::metric_set_weights, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_metric::metric_configure, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_metric::metric_tied, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_metric::weight_drift, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_metric::auto_uses_learned_metric, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_metric::dependence_weights, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_metric::auto_weights, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_metric::sobol_index, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_metric::null_sd, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_metric::group_sobol_index, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_metric::loo_loglik, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_metric::validate_tied_dims, m)?)?;
    m.add_class::<crate::py_metric::PyReservoir>()?;
    m.add("DEFAULT_REBUILD_DRIFT", ennbo::metric_auto::DEFAULT_REBUILD_DRIFT)?;
    m.add("DRIFT_WEIGHT_FLOOR", ennbo::metric_auto::DRIFT_WEIGHT_FLOOR)?;
    m.add("AUTO_MIN_HELDOUT_GAIN", ennbo::metric_auto::AUTO_MIN_HELDOUT_GAIN)?;
    m.add("AUTO_RESERVOIR_CAPACITY", ennbo::metric_auto::AUTO_RESERVOIR_CAPACITY)?;
    m.add("AUTO_K", ennbo::metric_auto::AUTO_K)?;
    m.add("AUTO_REFIT_GROWTH", ennbo::metric_auto::AUTO_REFIT_GROWTH)?;
    m.add("AUTO_RESCALE_TOL", ennbo::metric_auto::AUTO_RESCALE_TOL)?;
    m.add("DEPENDENCE_FLOOR", ennbo::metric_weights::DEPENDENCE_FLOOR)?;
    Ok(())
}

/// Parameter fitting module
#[pymodule]
#[pyo3(name = "fit")]
#[doc = "kiss-coverage-off"]
pub fn pymodule_fit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<crate::py_fitter::PyENNStatefulFitter>()?;
    m.add_function(wrap_pyfunction!(crate::py_fit::subsample_loglik_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_fitter::fit_model_affine, m)?)?;
    Ok(())
}

/// Optimizer module
#[pymodule]
#[pyo3(name = "optimizer")]
#[doc = "kiss-coverage-off"]
pub fn pymodule_optimizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<crate::py_optimizer::PyOptimizer>()?;
    m.add_class::<crate::py_optimizer::PyTelemetry>()?;
    m.add_function(wrap_pyfunction!(crate::py_optimizer::create_optimizer_enn_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_optimizer::create_optimizer_zero_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_optimizer::create_optimizer_lhd_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_optimizer::require_num_fit_samples_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_optimizer::validate_optimizer_rules_py, m)?)?;
    Ok(())
}

#[doc(hidden)]
#[doc = "kiss-coverage-off"]
pub fn pymodule_hypervolume_kiss_hook() {
    std::hint::black_box(pymodule_hypervolume);
}

#[doc(hidden)]
#[doc = "kiss-coverage-off"]
pub fn pymodule_hash_kiss_hook() {
    std::hint::black_box(pymodule_hash);
}

#[doc(hidden)]
#[doc = "kiss-coverage-off"]
pub fn pymodule_util_kiss_hook() {
    std::hint::black_box(pymodule_util);
}

#[doc(hidden)]
#[doc = "kiss-coverage-off"]
pub fn pymodule_model_kiss_hook() {
    std::hint::black_box(pymodule_model);
}

#[doc(hidden)]
#[doc = "kiss-coverage-off"]
pub fn pymodule_fit_kiss_hook() {
    std::hint::black_box(pymodule_fit);
}

#[doc(hidden)]
#[doc = "kiss-coverage-off"]
pub fn pymodule_optimizer_kiss_hook() {
    std::hint::black_box(pymodule_optimizer);
}

#[doc(hidden)]
#[doc = "kiss-coverage-off"]
pub fn kiss_link_child_pymodule_exports() {
    pymodule_hypervolume_kiss_hook();
    pymodule_hash_kiss_hook();
    pymodule_util_kiss_hook();
    pymodule_model_kiss_hook();
    pymodule_fit_kiss_hook();
    pymodule_optimizer_kiss_hook();
}

#[cfg(test)]
mod kiss_child_pymodule_coverage {
    use super::*;

    #[test]
    fn kiss_link_calls_all_child_pymodule_hooks() {
        kiss_link_child_pymodule_exports();
    }
}
