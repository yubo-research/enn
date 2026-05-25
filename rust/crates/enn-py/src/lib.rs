//! Python bindings for ENN core algorithms using PyO3.

#![allow(
    clippy::pedantic,
    clippy::nursery,
    clippy::cargo,
    clippy::useless_conversion
)]

use pyo3::prelude::*;
use pyo3::wrap_pymodule;

pub mod enn_py_build {
    include!("enn_py_build_api.inc.rs");
    use super::link_rpath;
    define_enn_py_build_api!(link_rpath);
}
pub mod link_rpath;
pub mod py_fit;
pub mod py_fitter;
pub mod py_hash;
pub mod py_hypervolume;
pub mod py_model;
pub mod py_optimizer;
pub mod py_util;

/// Hypervolume calculation module
#[pymodule]
pub(crate) fn hypervolume(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_hypervolume::hypervolume_2d_max_py, m)?)?;
    Ok(())
}

/// Hash-based RNG module
#[pymodule]
pub(crate) fn hash(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(
        py_hash::normal_hash_batch_multi_seed_fast_py,
        m
    )?)?;
    Ok(())
}

/// Utility functions module
#[pymodule]
pub(crate) fn util(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_util::standardize_y_py, m)?)?;
    m.add_function(wrap_pyfunction!(py_util::pareto_front_2d_maximize_py, m)?)?;
    m.add_function(wrap_pyfunction!(py_util::calculate_sobol_indices_py, m)?)?;
    m.add_function(wrap_pyfunction!(py_util::sobol_sequence_py, m)?)?;
    m.add_function(wrap_pyfunction!(py_util::arms_from_pareto_fronts_py, m)?)?;
    Ok(())
}

/// ENN model module
#[pymodule]
#[pyo3(name = "model")]
pub(crate) fn init_model_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<py_model::PyEpistemicNearestNeighbors>()?;
    m.add_class::<py_model::PyENNParams>()?;
    Ok(())
}

/// Parameter fitting module
#[pymodule]
#[pyo3(name = "fit")]
pub(crate) fn init_fit_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<py_fitter::PyENNStatefulFitter>()?;
    m.add_function(wrap_pyfunction!(py_fit::subsample_loglik_py, m)?)?;
    Ok(())
}

/// Optimizer module
#[pymodule]
pub(crate) fn optimizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<py_optimizer::PyOptimizer>()?;
    m.add_class::<py_optimizer::PyTelemetry>()?;
    m.add_function(wrap_pyfunction!(py_optimizer::create_optimizer_enn_py, m)?)?;
    m.add_function(wrap_pyfunction!(py_optimizer::create_optimizer_zero_py, m)?)?;
    m.add_function(wrap_pyfunction!(py_optimizer::create_optimizer_lhd_py, m)?)?;
    Ok(())
}

/// Main module (`import enn.enn_rust` when built with maturin `module-name = "enn.enn_rust"`).
#[pymodule]
#[pyo3(name = "enn_rust")]
pub(crate) fn enn_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(hypervolume))?;
    m.add_wrapped(wrap_pymodule!(hash))?;
    m.add_wrapped(wrap_pymodule!(util))?;
    m.add_wrapped(wrap_pymodule!(init_model_module))?;
    m.add_wrapped(wrap_pymodule!(init_fit_module))?;
    m.add_wrapped(wrap_pymodule!(optimizer))?;
    Ok(())
}

#[cfg(test)]
mod kiss_pymodule_coverage {
    use super::*;

    #[test]
    fn pymodule_init_fns_are_linked() {
        let _ = (
            hypervolume,
            hash,
            util,
            init_model_module,
            init_fit_module,
            optimizer,
            enn_rust,
        );
    }
}
