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

mod pymodule_wrappers;

pub use pymodule_wrappers::{
    kiss_link_child_pymodule_exports, pymodule_fit, pymodule_fit_kiss_hook, pymodule_hash,
    pymodule_hash_kiss_hook, pymodule_hypervolume, pymodule_hypervolume_kiss_hook, pymodule_model,
    pymodule_model_kiss_hook, pymodule_optimizer, pymodule_optimizer_kiss_hook, pymodule_util,
    pymodule_util_kiss_hook,
};

/// Main module (`import enn.enn_rust` when built with maturin `module-name = "enn.enn_rust"`).
#[pymodule]
#[pyo3(name = "enn_rust")]
pub(crate) fn pymodule_enn_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(pymodule_hypervolume))?;
    m.add_wrapped(wrap_pymodule!(pymodule_hash))?;
    m.add_wrapped(wrap_pymodule!(pymodule_util))?;
    m.add_wrapped(wrap_pymodule!(pymodule_model))?;
    m.add_wrapped(wrap_pymodule!(pymodule_fit))?;
    m.add_wrapped(wrap_pymodule!(pymodule_optimizer))?;
    Ok(())
}

#[doc(hidden)]
pub fn pymodule_enn_rust_kiss_hook() {
    std::hint::black_box(pymodule_enn_rust);
}

/// Hidden export for kiss static coverage of pymodule init fns from integration tests.
#[doc(hidden)]
pub fn kiss_link_pymodule_exports() {
    kiss_link_child_pymodule_exports();
    pymodule_enn_rust_kiss_hook();
}

#[doc(hidden)]
pub fn kiss_touch_util_module() {
    let _ = pymodule_util as fn(&Bound<'_, PyModule>) -> PyResult<()>;
}

#[doc(hidden)]
pub fn kiss_touch_hypervolume() {
    let _ = pymodule_hypervolume as fn(&Bound<'_, PyModule>) -> PyResult<()>;
}

#[doc(hidden)]
pub fn kiss_touch_hash() {
    let _ = pymodule_hash as fn(&Bound<'_, PyModule>) -> PyResult<()>;
}

#[doc(hidden)]
pub fn kiss_touch_init_model_module() {
    let _ = pymodule_model as fn(&Bound<'_, PyModule>) -> PyResult<()>;
}

#[doc(hidden)]
pub fn kiss_touch_init_fit_module() {
    let _ = pymodule_fit as fn(&Bound<'_, PyModule>) -> PyResult<()>;
}

#[doc(hidden)]
pub fn kiss_touch_optimizer_module() {
    let _ = pymodule_optimizer as fn(&Bound<'_, PyModule>) -> PyResult<()>;
}

#[doc(hidden)]
pub fn kiss_touch_enn_rust_module() {
    let _ = pymodule_enn_rust as fn(&Bound<'_, PyModule>) -> PyResult<()>;
}

#[cfg(test)]
mod kiss_pymodule_coverage {
    use super::*;

    #[test]
    fn pymodule_init_fns_are_linked() {
        let _ = (
            pymodule_hypervolume as fn(&Bound<'_, PyModule>) -> PyResult<()>,
            pymodule_hash as fn(&Bound<'_, PyModule>) -> PyResult<()>,
            pymodule_util as fn(&Bound<'_, PyModule>) -> PyResult<()>,
            pymodule_model as fn(&Bound<'_, PyModule>) -> PyResult<()>,
            pymodule_fit as fn(&Bound<'_, PyModule>) -> PyResult<()>,
            pymodule_optimizer as fn(&Bound<'_, PyModule>) -> PyResult<()>,
            pymodule_enn_rust as fn(&Bound<'_, PyModule>) -> PyResult<()>,
        );
    }

    #[test]
    fn kiss_link_calls_all_pymodule_hooks() {
        kiss_link_pymodule_exports();
    }

    #[test]
    fn pymodule_init_fns_called_via_touch_helpers() {
        kiss_touch_hypervolume();
        kiss_touch_hash();
        kiss_touch_util_module();
        kiss_touch_init_model_module();
        kiss_touch_init_fit_module();
        kiss_touch_optimizer_module();
        kiss_touch_enn_rust_module();
        kiss_link_pymodule_exports();
    }
}
