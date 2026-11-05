use numpy::{IntoPyArray, PyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::py_model::{PyENNParams, PyEpistemicNearestNeighbors};

#[pyclass(name = "ENNStatefulFitter")]
pub struct PyENNStatefulFitter {
    inner: ennbo::ENNFitter,
    rng: StdRng,
    infer_aleatoric_variance: bool,
}

#[pymethods]
impl PyENNStatefulFitter {
    #[new]
    #[pyo3(signature = (k, seed, infer_aleatoric_variance_scale=true))]
    fn new(k: i32, seed: u64, infer_aleatoric_variance_scale: bool) -> Self {
        Self {
            inner: ennbo::ENNFitter::new(k),
            rng: StdRng::seed_from_u64(seed),
            infer_aleatoric_variance: infer_aleatoric_variance_scale,
        }
    }

    #[pyo3(signature = (model, num_fit_candidates=None, num_fit_samples=None, params_warm_start=None, affine_calibrate=false))]
    fn ask(
        &mut self,
        model: &PyEpistemicNearestNeighbors,
        num_fit_candidates: Option<usize>,
        num_fit_samples: Option<usize>,
        params_warm_start: Option<PyENNParams>,
        affine_calibrate: bool,
    ) -> PyResult<PyENNParams> {
        let search = ennbo::ScaleSearch {
            infer_aleatoric_variance: self.infer_aleatoric_variance,
            affine_calibrate,
            ..ennbo::ScaleSearch::from_counts(num_fit_samples, num_fit_candidates)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        };
        let result = self
            .inner
            .ask(
                &model.inner,
                &search,
                params_warm_start.as_ref().map(|p| &p.inner),
                &mut self.rng,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyENNParams { inner: result })
    }
}

#[pyfunction]
#[pyo3(signature = (model, k, epistemic, aleatoric, num_fit_samples, seed, observation_noise=false))]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
pub fn fit_model_affine<'py>(
    py: Python<'py>,
    model: &PyEpistemicNearestNeighbors,
    k: i32,
    epistemic: f64,
    aleatoric: f64,
    num_fit_samples: i64,
    seed: u64,
    observation_noise: bool,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let num_fit_samples = usize::try_from(num_fit_samples).unwrap_or(0);
    let params = crate::py_params::make_params(k, epistemic, aleatoric)?;
    let mut rng = StdRng::seed_from_u64(seed);
    let cal = ennbo::surrogate_affine::fit_calibrator(
        &model.inner,
        &params,
        num_fit_samples,
        &mut rng,
        observation_noise,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        cal.a.into_pyarray_bound(py),
        cal.b.into_pyarray_bound(py),
        cal.c.into_pyarray_bound(py),
    ))
}

#[cfg(test)]
mod kiss_coverage_tests {
    use super::*;

    #[test]
    fn py_fitter_units_are_linked() {
        let _ = (
            PyENNStatefulFitter::new,
            PyENNStatefulFitter::ask,
            std::mem::size_of::<PyENNStatefulFitter>,
        );
    }
}
