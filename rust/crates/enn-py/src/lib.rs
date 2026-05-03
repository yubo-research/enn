//! Python bindings for ENN core algorithms using PyO3.

#![allow(
    clippy::pedantic,
    clippy::nursery,
    clippy::cargo,
    clippy::useless_conversion,
)]

use pyo3::prelude::*;
use pyo3::wrap_pymodule;

pub mod py_hypervolume;
pub mod py_hash;
pub mod py_util;
pub mod py_model;
pub mod py_fit;
pub mod py_optimizer;

/// Hypervolume calculation module
#[pymodule]
fn hypervolume(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_hypervolume::hypervolume_2d_max_py, m)?)?;
    Ok(())
}

/// Hash-based RNG module
#[pymodule]
fn hash(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_hash::normal_hash_batch_multi_seed_fast_py, m)?)?;
    Ok(())
}

/// Utility functions module
#[pymodule]
fn util(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_util::standardize_y_py, m)?)?;
    m.add_function(wrap_pyfunction!(py_util::pareto_front_2d_maximize_py, m)?)?;
    m.add_function(wrap_pyfunction!(py_util::calculate_sobol_indices_py, m)?)?;
    m.add_function(wrap_pyfunction!(py_util::sobol_sequence_py, m)?)?;
    Ok(())
}

/// ENN model module
#[pymodule]
fn model(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<py_model::PyEpistemicNearestNeighbors>()?;
    m.add_class::<py_model::PyENNParams>()?;
    Ok(())
}

/// Parameter fitting module
#[pymodule]
fn fit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(py_fit::enn_fit_py, m)?)?;
    m.add_function(wrap_pyfunction!(py_fit::subsample_loglik_py, m)?)?;
    Ok(())
}

/// Optimizer module
#[pymodule]
fn optimizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<py_optimizer::PyOptimizer>()?;
    m.add_class::<py_optimizer::PyTelemetry>()?;
    m.add_function(wrap_pyfunction!(py_optimizer::create_optimizer_enn_py, m)?)?;
    m.add_function(wrap_pyfunction!(py_optimizer::create_optimizer_zero_py, m)?)?;
    m.add_function(wrap_pyfunction!(py_optimizer::create_optimizer_lhd_py, m)?)?;
    Ok(())
}

/// Main module
#[pymodule]
fn enn_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(hypervolume))?;
    m.add_wrapped(wrap_pymodule!(hash))?;
    m.add_wrapped(wrap_pymodule!(util))?;
    m.add_wrapped(wrap_pymodule!(model))?;
    m.add_wrapped(wrap_pymodule!(fit))?;
    m.add_wrapped(wrap_pymodule!(optimizer))?;
    Ok(())
}
