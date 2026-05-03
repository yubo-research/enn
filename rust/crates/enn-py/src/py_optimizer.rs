//! Optimizer Python bindings.

use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

pub fn parse_config_overrides_from_dict(
    dict: &Bound<'_, pyo3::types::PyDict>,
) -> PyResult<ennbo::ConfigOverrides> {
    use ennbo::{AcquisitionConfig, CandidateRV, ConfigOverrides};
    use ennbo::index::IndexDriver;

    let mut overrides = ConfigOverrides::default();

    if let Some(v) = dict.get_item("index_driver")? {
        let s: String = v.extract()?;
        overrides.index_driver = Some(match s.to_lowercase().as_str() {
            "exact" | "flat" => IndexDriver::Exact,
            "hnsw" => IndexDriver::HNSW,
            _ => return Err(PyValueError::new_err(format!("Unknown index_driver: {}", s))),
        });
    }
    if let Some(acq) = dict.get_item("acquisition")? {
        let s: String = acq.extract()?;
        overrides.acquisition = Some(match s.as_str() {
            "ucb" => {
                let beta = dict
                    .get_item("acquisition_beta")?
                    .map(|v| v.extract::<f64>())
                    .transpose()?
                    .unwrap_or(2.0);
                AcquisitionConfig::UCB { beta }
            }
            "thompson" => AcquisitionConfig::Thompson,
            "random" => AcquisitionConfig::Random,
            "pareto" => AcquisitionConfig::Pareto,
            _ => return Err(PyValueError::new_err(format!("Unknown acquisition: {}", s))),
        });
    }
    if let Some(rv) = dict.get_item("candidate_rv")? {
        let s: String = rv.extract()?;
        overrides.candidate_rv = Some(match s.as_str() {
            "sobol" => CandidateRV::Sobol,
            "uniform" => CandidateRV::Uniform,
            "raasp" => CandidateRV::RAASP,
            _ => return Err(PyValueError::new_err(format!("Unknown candidate_rv: {}", s))),
        });
    }
    if let Some(v) = dict.get_item("num_candidates_factor")? {
        overrides.num_candidates_factor = Some(v.extract::<f64>()?);
    }
    if let Some(v) = dict.get_item("min_candidates")? {
        overrides.min_candidates = Some(v.extract::<usize>()?);
    }
    if let Some(v) = dict.get_item("max_candidates")? {
        overrides.max_candidates = Some(v.extract::<usize>()?);
    }
    if let Some(v) = dict.get_item("length_init")? {
        overrides.length_init = Some(v.extract::<f64>()?);
    }
    if let Some(v) = dict.get_item("length_min")? {
        overrides.length_min = Some(v.extract::<f64>()?);
    }
    if let Some(v) = dict.get_item("length_max")? {
        overrides.length_max = Some(v.extract::<f64>()?);
    }
    if let Some(v) = dict.get_item("trailing_obs")? {
        overrides.trailing_obs = Some(v.extract::<usize>()?);
    }
    if let Some(v) = dict.get_item("num_fit_samples")? {
        overrides.num_fit_samples = Some(v.extract::<usize>()?);
    }
    if let Some(v) = dict.get_item("num_fit_candidates")? {
        overrides.num_fit_candidates = Some(v.extract::<usize>()?);
    }
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
        }
    }

    /// Number of retained trust-region observations.
    fn tr_obs_count(&self) -> usize {
        self.inner.y_obs().map_or(0, |y| y.nrows())
    }

    /// Current trust-region length.
    fn tr_length(&self) -> f64 {
        self.inner.trust_region().length()
    }

    /// Get observations x in unit space (if any).
    fn x_obs<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArrayDyn<f64>>> {
        self.inner.x_obs().map(|x| x.into_dyn().into_pyarray_bound(py))
    }

    /// Get observation values y (if any).
    fn y_obs<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArrayDyn<f64>>> {
        self.inner.y_obs().map(|y| y.into_dyn().into_pyarray_bound(py))
    }

    /// Get incumbent x in unit space (if any).
    fn incumbent_x_unit<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArrayDyn<f64>>> {
        self.inner.incumbent_x_unit().map(|x| x.view().to_owned().into_dyn().into_pyarray_bound(py))
    }

    /// Get optimizer bounds.
    fn bounds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<f64>> {
        self.inner.bounds().view().to_owned().into_dyn().into_pyarray_bound(py)
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
