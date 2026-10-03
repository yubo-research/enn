//! Stateful ENN fitter Python bindings.

use numpy::{IntoPyArray, PyArray1, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::py_model::{PyENNParams, PyEpistemicNearestNeighbors, PosteriorPyOut};

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
    #[doc = "kiss-coverage-off"]
    fn new(k: i32, seed: u64, infer_aleatoric_variance_scale: bool) -> Self {
        Self {
            inner: ennbo::ENNFitter::new(k),
            rng: StdRng::seed_from_u64(seed),
            infer_aleatoric_variance: infer_aleatoric_variance_scale,
        }
    }

    #[pyo3(signature = (x, y, yvar=None, y_bounds=None))]
    #[doc = "kiss-coverage-off"]
    fn tell(
        &mut self,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray2<f64>,
        yvar: Option<PyReadonlyArray2<f64>>,
        y_bounds: Option<PyReadonlyArray2<f64>>,
    ) -> PyResult<()> {
        let yvar_arr = yvar.as_ref().map(|v| v.as_array());
        let y_bounds_owned = y_bounds.as_ref().map(|v| v.as_array().to_owned());
        self.inner
            .tell(
                &x.as_array(),
                &y.as_array(),
                yvar_arr.as_ref(),
                y_bounds_owned.as_ref(),
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[doc = "kiss-coverage-off"]
    fn y_std<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.inner.y_std().to_pyarray_bound(py))
    }

    #[pyo3(signature = (model, num_fit_candidates=None, num_fit_samples=None, params_warm_start=None, affine_calibrate=false))]
    #[doc = "kiss-coverage-off"]
    fn ask(
        &mut self,
        model: &PyEpistemicNearestNeighbors,
        num_fit_candidates: Option<usize>,
        num_fit_samples: Option<usize>,
        params_warm_start: Option<PyENNParams>,
        affine_calibrate: bool,
    ) -> PyResult<PyENNParams> {
        let warm = params_warm_start.as_ref().map(|p| p.inner);
        let search = ennbo::ScaleSearch {
            infer_aleatoric_variance: self.infer_aleatoric_variance,
            affine_calibrate,
            ..ennbo::ScaleSearch::from_counts(num_fit_samples, num_fit_candidates)
                .map_err(|e| PyValueError::new_err(e.to_string()))?
        };
        let result = self
            .inner
            .ask(&model.inner, &search, warm.as_ref(), &mut self.rng)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyENNParams { inner: result })
    }

    #[pyo3(signature = (a, b, c))]
    #[doc = "kiss-coverage-off"]
    fn set_calibrator_abc(
        &mut self,
        a: PyReadonlyArray1<f64>,
        b: PyReadonlyArray1<f64>,
        c: PyReadonlyArray1<f64>,
    ) -> PyResult<()> {
        self.inner
            .set_calibrator_abc(a.as_array(), b.as_array(), c.as_array())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[allow(clippy::type_complexity)]
    #[doc = "kiss-coverage-off"]
    fn affine_coeffs<'py>(
        &self,
        py: Python<'py>,
    ) -> Option<(
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
        Bound<'py, PyArray1<f64>>,
    )> {
        self.inner.calibrator().map(|cal| {
            (
                cal.a.to_pyarray_bound(py),
                cal.b.to_pyarray_bound(py),
                cal.c.to_pyarray_bound(py),
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (model, x, scales, exclude_nearest=false, observation_noise=false))]
    #[doc = "kiss-coverage-off"]
    fn posterior_calibrated<'py>(
        &self,
        py: Python<'py>,
        model: &PyEpistemicNearestNeighbors,
        x: PyReadonlyArray2<f64>,
        scales: PyReadonlyArray1<f64>,
        exclude_nearest: bool,
        observation_noise: bool,
    ) -> PyResult<PosteriorPyOut<'py>> {
        let params = enn_params(&scales)?;
        let flags = posterior_flags(exclude_nearest, observation_noise);
        let cal = self.inner.calibrator().cloned();
        let out = ennbo::surrogate_affine::calibrated_posterior(
            &model.inner,
            &x.as_array(),
            &params,
            &flags,
            cal.as_ref(),
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((
            out.mu.into_pyarray_bound(py),
            out.se.into_pyarray_bound(py),
            out.se_epi.into_pyarray_bound(py),
            out.se_ale.into_pyarray_bound(py),
            out.idx.map(|idx| idx.into_dyn().into_pyarray_bound(py)),
        ))
    }

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    #[pyo3(signature = (model, x, scales, function_seeds, exclude_nearest=false, observation_noise=false))]
    #[doc = "kiss-coverage-off"]
    fn function_draw_calibrated<'py>(
        &self,
        py: Python<'py>,
        model: &PyEpistemicNearestNeighbors,
        x: PyReadonlyArray2<f64>,
        scales: PyReadonlyArray1<f64>,
        function_seeds: Vec<i64>,
        exclude_nearest: bool,
        observation_noise: bool,
    ) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, Vec<Vec<usize>>)> {
        let params = enn_params(&scales)?;
        let flags = posterior_flags(exclude_nearest, observation_noise);
        let cal = self.inner.calibrator().cloned();
        let (draws, idx) = ennbo::surrogate_affine::calibrated_function_draw(
            &model.inner,
            &x.as_array(),
            &params,
            &function_seeds,
            &flags,
            cal.as_ref(),
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((draws.into_dyn().into_pyarray_bound(py), idx))
    }
}

fn enn_params(scales: &PyReadonlyArray1<f64>) -> PyResult<ennbo::ENNParams> {
    let s = scales.as_slice()?;
    if s.len() != 3 {
        return Err(PyValueError::new_err("scales must be [k, epistemic, aleatoric]"));
    }
    ennbo::ENNParams::new(s[0] as i32, s[1], s[2]).map_err(|e| PyValueError::new_err(e.to_string()))
}

fn posterior_flags(exclude_nearest: bool, observation_noise: bool) -> ennbo::PosteriorFlags {
    ennbo::PosteriorFlags::new()
        .with_exclude_nearest(exclude_nearest)
        .with_observation_noise(observation_noise)
}

#[pyfunction]
#[pyo3(signature = (model, k, epistemic, aleatoric, num_fit_samples, seed, observation_noise=false))]
#[allow(clippy::type_complexity, clippy::too_many_arguments)]
#[doc = "kiss-coverage-off"]
pub fn fit_model_affine<'py>(
    py: Python<'py>,
    model: &PyEpistemicNearestNeighbors,
    k: i32,
    epistemic: f64,
    aleatoric: f64,
    num_fit_samples: usize,
    seed: u64,
    observation_noise: bool,
) -> PyResult<(
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
)> {
    let params = ennbo::ENNParams::new(k, epistemic, aleatoric)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
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
            PyENNStatefulFitter::tell,
            PyENNStatefulFitter::y_std,
            PyENNStatefulFitter::ask,
            PyENNStatefulFitter::affine_coeffs,
            PyENNStatefulFitter::set_calibrator_abc,
            PyENNStatefulFitter::posterior_calibrated,
            PyENNStatefulFitter::function_draw_calibrated,
            std::mem::size_of::<PyENNStatefulFitter>,
        );
    }
}
