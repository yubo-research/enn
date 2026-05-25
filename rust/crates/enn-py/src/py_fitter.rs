//! Stateful ENN fitter Python bindings.

use numpy::{PyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::py_model::{PyENNParams, PyEpistemicNearestNeighbors};

#[pyclass(name = "ENNStatefulFitter")]
pub struct PyENNStatefulFitter {
    inner: ennbo::ENNFitter,
    rng: StdRng,
}

#[pymethods]
impl PyENNStatefulFitter {
    #[new]
    #[pyo3(signature = (k, seed, infer_aleatoric_variance_scale=true))]
    fn new(k: i32, seed: u64, infer_aleatoric_variance_scale: bool) -> Self {
        Self {
            inner: ennbo::ENNFitter::new(k, infer_aleatoric_variance_scale),
            rng: StdRng::seed_from_u64(seed),
        }
    }

    #[pyo3(signature = (x, y, yvar=None))]
    fn tell(
        &mut self,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray2<f64>,
        yvar: Option<PyReadonlyArray2<f64>>,
    ) -> PyResult<()> {
        let yvar_arr = yvar.as_ref().map(|v| v.as_array());
        self.inner
            .tell(&x.as_array(), &y.as_array(), yvar_arr.as_ref())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn y_std<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<f64>>> {
        Ok(self.inner.y_std().to_pyarray_bound(py))
    }

    #[pyo3(signature = (model, num_fit_candidates, num_fit_samples, params_warm_start=None))]
    fn ask(
        &mut self,
        model: &PyEpistemicNearestNeighbors,
        num_fit_candidates: usize,
        num_fit_samples: usize,
        params_warm_start: Option<PyENNParams>,
    ) -> PyResult<PyENNParams> {
        let warm = params_warm_start.as_ref().map(|p| p.inner);
        let result = self
            .inner
            .ask(
                &model.inner,
                num_fit_candidates,
                num_fit_samples,
                warm.as_ref(),
                &mut self.rng,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyENNParams { inner: result })
    }
}
