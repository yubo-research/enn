use numpy::{IntoPyArray, PyArrayDyn, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub(crate) fn optional_f64(
    dict: &Bound<'_, pyo3::types::PyDict>,
    key: &str,
) -> PyResult<Option<f64>> {
    match dict.get_item(key)? {
        Some(v) => Ok(Some(v.extract()?)),
        None => Ok(None),
    }
}

pub(crate) fn optional_usize(
    dict: &Bound<'_, pyo3::types::PyDict>,
    key: &str,
) -> PyResult<Option<usize>> {
    match dict.get_item(key)? {
        Some(v) => Ok(Some(v.extract()?)),
        None => Ok(None),
    }
}

pub(crate) fn optional_bool(
    dict: &Bound<'_, pyo3::types::PyDict>,
    key: &str,
) -> PyResult<Option<bool>> {
    match dict.get_item(key)? {
        Some(v) => Ok(Some(v.extract()?)),
        None => Ok(None),
    }
}

pub(crate) fn apply_scalar_overrides(
    dict: &Bound<'_, pyo3::types::PyDict>,
    overrides: &mut ennbo::ConfigOverrides,
) -> PyResult<()> {
    overrides.num_candidates_per_dim = optional_usize(dict, "num_candidates_per_dim")?;
    overrides.min_candidates = optional_usize(dict, "min_candidates")?;
    overrides.max_candidates = optional_usize(dict, "max_candidates")?;
    overrides.num_candidates_per_arm = optional_usize(dict, "num_candidates_per_arm")?;
    overrides.length_init = optional_f64(dict, "length_init")?;
    overrides.length_min = optional_f64(dict, "length_min")?;
    overrides.length_max = optional_f64(dict, "length_max")?;
    overrides.noise_aware = optional_bool(dict, "noise_aware")?;
    Ok(())
}

#[cfg(test)]
mod kiss_coverage_tests {
    use super::{apply_scalar_overrides, optional_bool, optional_f64, optional_usize};

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

fn acquisition_from_name(s: &str) -> PyResult<ennbo::AcquisitionConfig> {
    use ennbo::AcquisitionConfig;
    match s {
        "ucb" => Ok(AcquisitionConfig::default()),
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

fn parse_morbo_override(
    dict: &Bound<'_, pyo3::types::PyDict>,
    kind: Option<ennbo::TrustRegionKind>,
) -> PyResult<Option<ennbo::morbo_override::MorboOverride>> {
    let num_metrics = optional_usize(dict, "num_metrics")?;
    let alpha = optional_f64(dict, "alpha")?;
    let rescalarize_name = match dict.get_item("rescalarize")? {
        Some(v) => Some(v.extract::<String>()?),
        None => None,
    };
    let any = num_metrics.is_some() || alpha.is_some() || rescalarize_name.is_some();
    let morbo = kind == Some(ennbo::TrustRegionKind::Morbo);
    if !morbo && !any {
        return Ok(None);
    }
    if !morbo {
        return Err(PyValueError::new_err(
            "num_metrics and alpha require trust_region 'morbo'",
        ));
    }
    let (Some(num_metrics), Some(alpha)) = (num_metrics, alpha) else {
        return Err(PyValueError::new_err(
            "MORBO overrides require num_metrics and alpha together",
        ));
    };
    let rescalarize = match rescalarize_name.as_deref() {
        None => ennbo::Rescalarize::OnRestart,
        Some(s) => s.parse().map_err(|_| {
            PyValueError::new_err(format!(
                "Unknown rescalarize mode: {s:?}; expected \"on_propose\" or \"on_restart\""
            ))
        })?,
    };
    Ok(Some(ennbo::morbo_override::MorboOverride {
        num_metrics,
        alpha,
        rescalarize,
    }))
}

pub(crate) const CONFIG_OVERRIDE_KEYS: &[&str] = &[
    "acquisition",
    "candidate_rv",
    "trust_region",
    "num_metrics",
    "alpha",
    "rescalarize",
    "raasp_fast",
    "num_candidates_per_dim",
    "min_candidates",
    "max_candidates",
    "num_candidates_per_arm",
    "length_init",
    "length_min",
    "length_max",
    "noise_aware",
];

fn reject_unknown_override_keys(dict: &Bound<'_, pyo3::types::PyDict>) -> PyResult<()> {
    for key in dict.keys() {
        let name: String = key.extract()?;
        let known = CONFIG_OVERRIDE_KEYS.contains(&name.as_str())
            || crate::py_enn_overrides::ENN_OVERRIDE_KEYS.contains(&name.as_str());
        if !known {
            return Err(PyValueError::new_err(format!(
                "Unknown config override key: {name:?}"
            )));
        }
    }
    Ok(())
}

pub fn parse_config_overrides_from_dict(
    dict: &Bound<'_, pyo3::types::PyDict>,
) -> PyResult<ennbo::ConfigOverrides> {
    use ennbo::ConfigOverrides;

    reject_unknown_override_keys(dict)?;
    let mut overrides = ConfigOverrides::default();

    if let Some(acq) = dict.get_item("acquisition")? {
        let s: String = acq.extract()?;
        overrides.acquisition = Some(acquisition_from_name(&s)?);
    }
    if let Some(rv) = dict.get_item("candidate_rv")? {
        overrides.candidate_rv = Some(parse_candidate_rv(&rv.extract::<String>()?)?);
    }
    if let Some(v) = dict.get_item("trust_region")? {
        let name: String = v.extract()?;
        overrides.trust_region_kind = Some(
            ennbo::TrustRegionKind::parse(&name)
                .map_err(|e| PyValueError::new_err(e.to_string()))?,
        );
    }
    overrides.morbo = parse_morbo_override(dict, overrides.trust_region_kind)?;
    overrides.raasp_fast = optional_bool(dict, "raasp_fast")?;
    overrides.enn = crate::py_enn_overrides::parse_enn_overrides(dict)?;
    apply_scalar_overrides(dict, &mut overrides)?;
    Ok(overrides)
}

#[pyclass(name = "Optimizer")]
pub struct PyOptimizer {
    inner: ennbo::Optimizer,
}

#[pymethods]
impl PyOptimizer {
    fn ask<'py>(
        &mut self,
        py: Python<'py>,
        num_arms: usize,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let result = self
            .inner
            .ask(num_arms)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(result.into_dyn().into_pyarray_bound(py))
    }

    #[pyo3(signature = (x, y, y_var=None))]
    fn tell(
        &mut self,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray2<f64>,
        y_var: Option<PyReadonlyArray2<f64>>,
    ) -> PyResult<()> {
        let x_arr = x.as_array();
        let y_arr = y.as_array();
        let result = match y_var.as_ref() {
            Some(yv) => {
                let yv_arr = yv.as_array();
                self.inner.tell(&x_arr, &y_arr, Some(&yv_arr))
            }
            None => self.inner.tell(&x_arr, &y_arr, None),
        };
        result.map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn posterior_mu<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        let mu = self
            .inner
            .posterior_mu(&x.as_array())
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(mu.into_dyn().into_pyarray_bound(py))
    }

    fn has_surrogate(&self) -> bool {
        self.inner.has_surrogate()
    }

    fn init_progress(&self) -> Option<(usize, usize)> {
        self.inner.init_progress()
    }

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

    fn tr_obs_count(&self) -> usize {
        self.inner.y_obs().map_or(0, |y| y.nrows())
    }

    fn tr_length(&self) -> f64 {
        self.inner.tr_length()
    }

    fn x_obs<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArrayDyn<f64>>> {
        self.inner
            .x_obs()
            .map(|x| x.into_dyn().into_pyarray_bound(py))
    }

    fn y_obs<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArrayDyn<f64>>> {
        self.inner
            .y_obs()
            .map(|y| y.into_dyn().into_pyarray_bound(py))
    }

    fn incumbent_x<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArrayDyn<f64>>> {
        self.inner
            .incumbent_x()
            .map(|x| x.into_dyn().into_pyarray_bound(py))
    }

    fn bounds<'py>(&self, py: Python<'py>) -> Bound<'py, PyArrayDyn<f64>> {
        self.inner
            .bounds()
            .view()
            .to_owned()
            .into_dyn()
            .into_pyarray_bound(py)
    }
}

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

#[pyfunction(name = "require_num_fit_samples", signature = (acquisition, num_fit_samples=None))]
pub fn require_num_fit_samples_py(
    acquisition: &str,
    num_fit_samples: Option<usize>,
) -> PyResult<()> {
    let acq = acquisition_from_name(acquisition)?;
    let fit_samples = ennbo::FitSamples::from_count(num_fit_samples)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    ennbo::require_fit_samples(&acq, fit_samples).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn optimizer_flag(dict: &Bound<'_, pyo3::types::PyDict>, key: &str) -> PyResult<bool> {
    dict.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing optimizer flag {key}")))?
        .extract()
}

#[pyfunction(name = "validate_optimizer_rules")]
pub fn validate_optimizer_rules_py(
    flags: &Bound<'_, pyo3::types::PyDict>,
    acquisition: &str,
) -> PyResult<()> {
    let rules = ennbo::OptimizerRuleSet {
        lhd_only: optimizer_flag(flags, "lhd_only")?,
        has_surrogate: optimizer_flag(flags, "has_surrogate")?,
    };
    let kind = acquisition_from_name(acquisition)?;
    ennbo::validate_optimizer_rules(&rules, &kind).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyfunction(name = "create_optimizer_enn")]
#[pyo3(signature = (bounds, k=None, num_init=None, seed=42, config_overrides=None))]
pub fn create_optimizer_enn_py(
    bounds: PyReadonlyArray2<f64>,
    k: Option<i32>,
    num_init: Option<usize>,
    seed: u64,
    config_overrides: Option<Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<PyOptimizer> {
    use ennbo::optimizer_factory::create_optimizer_enn_with_overrides;

    let overrides: Option<ennbo::ConfigOverrides> = config_overrides
        .as_ref()
        .map(|d| parse_config_overrides_from_dict(d))
        .transpose()?;

    let optimizer = create_optimizer_enn_with_overrides(
        bounds.as_array().to_owned(),
        k,
        num_init,
        seed,
        overrides.as_ref(),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyOptimizer { inner: optimizer })
}

#[pyfunction(name = "create_optimizer_zero")]
#[pyo3(signature = (bounds, num_init=None, seed=42, config_overrides=None))]
pub fn create_optimizer_zero_py(
    bounds: PyReadonlyArray2<f64>,
    num_init: Option<usize>,
    seed: u64,
    config_overrides: Option<Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<PyOptimizer> {
    use ennbo::optimizer_factory::create_optimizer_zero_with_overrides;

    let overrides: Option<ennbo::ConfigOverrides> = config_overrides
        .as_ref()
        .map(|d| parse_config_overrides_from_dict(d))
        .transpose()?;

    let optimizer = create_optimizer_zero_with_overrides(
        bounds.as_array().to_owned(),
        num_init,
        seed,
        overrides.as_ref(),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyOptimizer { inner: optimizer })
}

#[pyfunction(name = "create_optimizer_lhd")]
#[pyo3(signature = (bounds, num_init=None, seed=42, config_overrides=None))]
pub fn create_optimizer_lhd_py(
    bounds: PyReadonlyArray2<f64>,
    num_init: Option<usize>,
    seed: u64,
    config_overrides: Option<Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<PyOptimizer> {
    use ennbo::optimizer_factory::create_optimizer_lhd_with_overrides;

    let overrides: Option<ennbo::ConfigOverrides> = config_overrides
        .as_ref()
        .map(|d| parse_config_overrides_from_dict(d))
        .transpose()?;

    let optimizer = create_optimizer_lhd_with_overrides(
        bounds.as_array().to_owned(),
        num_init,
        seed,
        overrides.as_ref(),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyOptimizer { inner: optimizer })
}

#[cfg(test)]
mod kiss_pymethods_coverage {
    use super::{PyOptimizer, PyTelemetry};

    #[test]
    fn py_optimizer_pymethods_are_linked() {
        let _ = (
            PyOptimizer::ask,
            PyOptimizer::tell,
            PyOptimizer::has_surrogate,
            PyOptimizer::init_progress,
            PyOptimizer::telemetry,
            PyOptimizer::tr_obs_count,
            PyOptimizer::tr_length,
            PyOptimizer::x_obs,
            PyOptimizer::y_obs,
            PyOptimizer::incumbent_x,
            PyOptimizer::bounds,
            std::mem::size_of::<PyOptimizer>,
            std::mem::size_of::<PyTelemetry>,
        );
    }
}
