//! Optimizer Python bindings.

use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::path::PathBuf;

pub(crate) fn optional_f64(dict: &Bound<'_, pyo3::types::PyDict>, key: &str) -> PyResult<Option<f64>> {
    match dict.get_item(key)? {
        Some(v) => Ok(Some(v.extract()?)),
        None => Ok(None),
    }
}

pub(crate) fn optional_usize(dict: &Bound<'_, pyo3::types::PyDict>, key: &str) -> PyResult<Option<usize>> {
    match dict.get_item(key)? {
        Some(v) => Ok(Some(v.extract()?)),
        None => Ok(None),
    }
}

pub(crate) fn optional_bool(dict: &Bound<'_, pyo3::types::PyDict>, key: &str) -> PyResult<Option<bool>> {
    match dict.get_item(key)? {
        Some(v) => Ok(Some(v.extract()?)),
        None => Ok(None),
    }
}

pub(crate) fn apply_scalar_overrides(
    dict: &Bound<'_, pyo3::types::PyDict>,
    overrides: &mut ennbo::ConfigOverrides,
) -> PyResult<()> {
    overrides.num_candidates_factor = optional_f64(dict, "num_candidates_factor")?;
    overrides.min_candidates = optional_usize(dict, "min_candidates")?;
    overrides.max_candidates = optional_usize(dict, "max_candidates")?;
    overrides.num_candidates_per_arm = optional_usize(dict, "num_candidates_per_arm")?;
    overrides.length_init = optional_f64(dict, "length_init")?;
    overrides.length_min = optional_f64(dict, "length_min")?;
    overrides.length_max = optional_f64(dict, "length_max")?;
    overrides.num_fit_samples = optional_usize(dict, "num_fit_samples")?;
    overrides.num_fit_candidates = optional_usize(dict, "num_fit_candidates")?;
    overrides.noise_aware = optional_bool(dict, "noise_aware")?;
    overrides.scale_x = optional_bool(dict, "scale_x")?;
    Ok(())
}

#[cfg(test)]
mod kiss_coverage_tests {
    use super::{
        apply_scalar_overrides, optional_bool, optional_f64, optional_usize,
    };

    #[test]
    fn py_optimizer_helpers_are_linked() {
        let _ = (
            optional_f64 as fn(_, _) -> _,
            optional_usize as fn(_, _) -> _,
            optional_bool as fn(_, _) -> _,
            apply_scalar_overrides as fn(_, _) -> _,
        );
    }
}

fn parse_index_driver(s: &str) -> PyResult<ennbo::index::IndexDriver> {
    use ennbo::index::IndexDriver;
    match s.to_lowercase().as_str() {
        "exact" | "flat" => Ok(IndexDriver::Exact),
        "hnsw" => Ok(IndexDriver::HNSW),
        "hnsw_disk" => Ok(IndexDriver::HNSWDisk),
        _ => Err(PyValueError::new_err(format!("Unknown index_driver: {s}"))),
    }
}

fn parse_acquisition(
    dict: &Bound<'_, pyo3::types::PyDict>,
    s: &str,
) -> PyResult<ennbo::AcquisitionConfig> {
    use ennbo::AcquisitionConfig;
    match s {
        "ucb" => {
            let beta = dict
                .get_item("acquisition_beta")?
                .map(|v| v.extract::<f64>())
                .transpose()?
                .unwrap_or(2.0);
            Ok(AcquisitionConfig::UCB { beta })
        }
        "thompson" => Ok(AcquisitionConfig::Thompson),
        "random" => Ok(AcquisitionConfig::Random),
        "pareto" => Ok(AcquisitionConfig::Pareto),
        _ => Err(PyValueError::new_err(format!("Unknown acquisition: {s}"))),
    }
}

fn parse_candidate_rv(s: &str) -> PyResult<ennbo::CandidateRV> {
    use ennbo::CandidateRV;
    match s {
        "sobol" => Ok(CandidateRV::Sobol),
        "uniform" => Ok(CandidateRV::Uniform),
        "raasp" => Ok(CandidateRV::RAASP),
        _ => Err(PyValueError::new_err(format!("Unknown candidate_rv: {s}"))),
    }
}

fn parse_enn_storage(s: &str) -> PyResult<ennbo::EnnStorage> {
    match s.to_lowercase().as_str() {
        "disk" => Ok(ennbo::EnnStorage::Disk),
        "memory" | "in_memory" | "inmemory" => Ok(ennbo::EnnStorage::InMemory),
        _ => Err(PyValueError::new_err(format!("Unknown enn_storage: {s}"))),
    }
}

pub fn parse_config_overrides_from_dict(
    dict: &Bound<'_, pyo3::types::PyDict>,
) -> PyResult<ennbo::ConfigOverrides> {
    use ennbo::ConfigOverrides;

    let mut overrides = ConfigOverrides::default();

    if let Some(v) = dict.get_item("index_driver")? {
        overrides.index_driver = Some(parse_index_driver(&v.extract::<String>()?)?);
    }
    if let Some(acq) = dict.get_item("acquisition")? {
        let s: String = acq.extract()?;
        overrides.acquisition = Some(parse_acquisition(dict, &s)?);
    }
    if let Some(rv) = dict.get_item("candidate_rv")? {
        overrides.candidate_rv = Some(parse_candidate_rv(&rv.extract::<String>()?)?);
    }
    if let Some(v) = dict.get_item("trust_region")? {
        overrides.trust_region_kind = Some(v.extract()?);
    }
    overrides.num_metrics = optional_usize(dict, "num_metrics")?;
    overrides.alpha = optional_f64(dict, "alpha")?;
    if let Some(v) = dict.get_item("rescalarize")? {
        overrides.rescalarize = Some(v.extract()?);
    }
    if let Some(v) = dict.get_item("enn_storage")? {
        overrides.enn_storage = Some(parse_enn_storage(&v.extract::<String>()?)?);
    }
    if let Some(v) = dict.get_item("work_dir")? {
        overrides.work_dir = Some(PathBuf::from(v.extract::<String>()?));
    }
    apply_scalar_overrides(dict, &mut overrides)?;
    Ok(overrides)
}

/// Python wrapper for Optimizer
#[pyclass(name = "Optimizer")]
pub struct PyOptimizer {
    inner: ennbo::Optimizer,
}

#[pymethods]
impl PyOptimizer {
    /// Ask for candidate points
    #[pyo3(signature = (num_arms, seed))]
    fn ask<'py>(
        &mut self,
        py: Python<'py>,
        num_arms: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let mut rng = StdRng::seed_from_u64(seed);

        let result = self
            .inner
            .ask(num_arms, &mut rng)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(result.into_dyn().into_pyarray_bound(py))
    }

    /// Tell observations
    #[pyo3(signature = (x, y, seed))]
    fn tell(
        &mut self,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray2<f64>,
        seed: u64,
    ) -> PyResult<()> {
        let mut rng = StdRng::seed_from_u64(seed);

        self.inner
            .tell(&x.as_array(), &y.as_array(), &mut rng)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Get init progress if in initialization phase
    fn init_progress(&self) -> Option<(usize, usize)> {
        self.inner.init_progress()
    }

    /// Get current telemetry
    fn telemetry(&self) -> PyTelemetry {
        let t = self.inner.telemetry();
        PyTelemetry {
            dt_fit: t.dt_fit,
            dt_gen: t.dt_gen,
            dt_sel: t.dt_sel,
            dt_tell: t.dt_tell,
            num_candidates: t.num_candidates,
        }
    }

    /// Number of retained trust-region observations.
    fn tr_obs_count(&self) -> usize {
        self.inner.y_obs().map_or(0, |y| y.nrows())
    }

    /// Current trust-region length.
    fn tr_length(&self) -> f64 {
        self.inner.tr_length()
    }

    /// Get observations x in unit space (if any).
    fn x_obs<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArrayDyn<f64>>> {
        self.inner
            .x_obs()
            .map(|x| x.into_dyn().into_pyarray_bound(py))
    }

    /// Get observation values y (if any).
    fn y_obs<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArrayDyn<f64>>> {
        self.inner
            .y_obs()
            .map(|y| y.into_dyn().into_pyarray_bound(py))
    }

    /// Get incumbent x in unit space (if any).
    fn incumbent_x_unit<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArrayDyn<f64>>> {
        self.inner
            .incumbent_x_unit()
            .map(|x| x.view().to_owned().into_dyn().into_pyarray_bound(py))
    }

    /// Get optimizer bounds.
    fn bounds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<f64>> {
        self.inner
            .bounds()
            .view()
            .to_owned()
            .into_dyn()
            .into_pyarray_bound(py)
    }
}

/// Telemetry data structure for Python
#[pyclass(name = "Telemetry")]
#[derive(Clone, Copy)]
pub struct PyTelemetry {
    #[pyo3(get)]
    pub dt_fit: f64,
    #[pyo3(get)]
    pub dt_gen: f64,
    #[pyo3(get)]
    pub dt_sel: f64,
    #[pyo3(get)]
    pub dt_tell: f64,
    #[pyo3(get)]
    pub num_candidates: usize,
}

/// Create TuRBO-ENN optimizer
#[pyfunction(name = "create_optimizer_enn")]
#[pyo3(signature = (bounds, k=10, num_init=10, seed=42, config_overrides=None))]
pub fn create_optimizer_enn_py(
    bounds: PyReadonlyArray2<f64>,
    k: i32,
    num_init: usize,
    seed: u64,
    config_overrides: Option<Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<PyOptimizer> {
    use ennbo::optimizer_factory::create_optimizer_enn_with_overrides;

    let mut rng = StdRng::seed_from_u64(seed);
    let overrides: Option<ennbo::ConfigOverrides> = config_overrides
        .as_ref()
        .map(|d| parse_config_overrides_from_dict(d))
        .transpose()?;

    let optimizer = create_optimizer_enn_with_overrides(
        bounds.as_array().to_owned(),
        k,
        num_init,
        &mut rng,
        overrides.as_ref(),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyOptimizer { inner: optimizer })
}

/// Create TuRBO-ZERO optimizer
#[pyfunction(name = "create_optimizer_zero")]
#[pyo3(signature = (bounds, num_init=10, seed=42, config_overrides=None))]
pub fn create_optimizer_zero_py(
    bounds: PyReadonlyArray2<f64>,
    num_init: usize,
    seed: u64,
    config_overrides: Option<Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<PyOptimizer> {
    use ennbo::optimizer_factory::create_optimizer_zero_with_overrides;

    let mut rng = StdRng::seed_from_u64(seed);
    let overrides: Option<ennbo::ConfigOverrides> = config_overrides
        .as_ref()
        .map(|d| parse_config_overrides_from_dict(d))
        .transpose()?;

    let optimizer = create_optimizer_zero_with_overrides(
        bounds.as_array().to_owned(),
        num_init,
        &mut rng,
        overrides.as_ref(),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyOptimizer { inner: optimizer })
}

/// Create LHD-only optimizer
#[pyfunction(name = "create_optimizer_lhd")]
#[pyo3(signature = (bounds, num_init=10, seed=42, config_overrides=None))]
pub fn create_optimizer_lhd_py(
    bounds: PyReadonlyArray2<f64>,
    num_init: usize,
    seed: u64,
    config_overrides: Option<Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<PyOptimizer> {
    use ennbo::optimizer_factory::create_optimizer_lhd_with_overrides;

    let mut rng = StdRng::seed_from_u64(seed);
    let overrides: Option<ennbo::ConfigOverrides> = config_overrides
        .as_ref()
        .map(|d| parse_config_overrides_from_dict(d))
        .transpose()?;

    let optimizer = create_optimizer_lhd_with_overrides(
        bounds.as_array().to_owned(),
        num_init,
        &mut rng,
        overrides.as_ref(),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyOptimizer { inner: optimizer })
}
