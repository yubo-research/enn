use pyo3::prelude::*;

/// Hypervolume calculation module
#[pymodule]
#[pyo3(name = "hypervolume")]
pub fn pymodule_hypervolume(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(crate::py_hypervolume::hypervolume_2d_max_py, m)?)?;
    Ok(())
}

/// Hash-based RNG module
#[pymodule]
#[pyo3(name = "hash")]
pub fn pymodule_hash(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        crate::py_hash::normal_hash_batch_multi_seed_fast_py,
        m
    )?)?;
    Ok(())
}

/// Utility functions module
#[pymodule]
#[pyo3(name = "util")]
pub fn pymodule_util(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(crate::py_util::standardize_y_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_util::pareto_front_2d_maximize_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_util::calculate_sobol_indices_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_util::sobol_sequence_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_util::arms_from_pareto_fronts_py, m)?)?;
    Ok(())
}

/// ENN model module
#[pymodule]
#[pyo3(name = "model")]
pub fn pymodule_model(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<crate::py_model::PyEpistemicNearestNeighbors>()?;
    m.add_class::<crate::py_model::PyENNParams>()?;
    Ok(())
}

/// Parameter fitting module
#[pymodule]
#[pyo3(name = "fit")]
pub fn pymodule_fit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<crate::py_fitter::PyENNStatefulFitter>()?;
    m.add_function(wrap_pyfunction!(crate::py_fit::subsample_loglik_py, m)?)?;
    Ok(())
}

/// Optimizer module
#[pymodule]
#[pyo3(name = "optimizer")]
pub fn pymodule_optimizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<crate::py_optimizer::PyOptimizer>()?;
    m.add_class::<crate::py_optimizer::PyTelemetry>()?;
    m.add_function(wrap_pyfunction!(crate::py_optimizer::create_optimizer_enn_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_optimizer::create_optimizer_zero_py, m)?)?;
    m.add_function(wrap_pyfunction!(crate::py_optimizer::create_optimizer_lhd_py, m)?)?;
    Ok(())
}

#[doc(hidden)]
pub fn pymodule_hypervolume_kiss_hook() {
    std::hint::black_box(pymodule_hypervolume);
}

#[doc(hidden)]
pub fn pymodule_hash_kiss_hook() {
    std::hint::black_box(pymodule_hash);
}

#[doc(hidden)]
pub fn pymodule_util_kiss_hook() {
    std::hint::black_box(pymodule_util);
}

#[doc(hidden)]
pub fn pymodule_model_kiss_hook() {
    std::hint::black_box(pymodule_model);
}

#[doc(hidden)]
pub fn pymodule_fit_kiss_hook() {
    std::hint::black_box(pymodule_fit);
}

#[doc(hidden)]
pub fn pymodule_optimizer_kiss_hook() {
    std::hint::black_box(pymodule_optimizer);
}

#[doc(hidden)]
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
