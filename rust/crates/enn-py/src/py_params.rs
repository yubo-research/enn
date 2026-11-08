use numpy::{PyArray1, PyReadonlyArray1, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

pub(crate) type PyCalibratorCoefficients<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray1<f64>>,
);

type PyCalibratorArgument<'py> = (
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, f64>,
);

fn calibrator_from_argument(
    calibrator: Option<PyCalibratorArgument<'_>>,
) -> PyResult<Option<Arc<ennbo::calibration::AffineCalibrator>>> {
    calibrator
        .map(|(a, b, c)| {
            ennbo::calibration::AffineCalibrator::from_coefficients(
                a.as_array(),
                b.as_array(),
                c.as_array(),
            )
            .map(Arc::new)
            .map_err(|e| PyValueError::new_err(e.to_string()))
        })
        .transpose()
}

pub(crate) fn make_params(
    k_num_neighbors: i32,
    epistemic_variance_scale: f64,
    aleatoric_variance_scale: f64,
) -> PyResult<ennbo::ENNParams> {
    ennbo::ENNParams::new(
        k_num_neighbors,
        epistemic_variance_scale,
        aleatoric_variance_scale,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pyclass(name = "ENNParams")]
#[derive(Clone)]
pub struct PyENNParams {
    pub(crate) inner: ennbo::ENNParams,
}

#[pymethods]
impl PyENNParams {
    #[new]
    #[pyo3(signature = (k_num_neighbors, epistemic_variance_scale, aleatoric_variance_scale, calibrator=None))]
    pub(crate) fn new(
        k_num_neighbors: i32,
        epistemic_variance_scale: f64,
        aleatoric_variance_scale: f64,
        calibrator: Option<PyCalibratorArgument<'_>>,
    ) -> PyResult<Self> {
        let inner = make_params(
            k_num_neighbors,
            epistemic_variance_scale,
            aleatoric_variance_scale,
        )?
        .with_calibrator(calibrator_from_argument(calibrator)?);
        Ok(Self { inner })
    }

    #[getter]
    pub(crate) fn k_num_neighbors(&self) -> i32 {
        self.inner.k_num_neighbors
    }

    #[getter]
    pub(crate) fn epistemic_variance_scale(&self) -> f64 {
        self.inner.epistemic_variance_scale
    }

    #[getter]
    pub(crate) fn aleatoric_variance_scale(&self) -> f64 {
        self.inner.aleatoric_variance_scale
    }

    #[getter]
    pub(crate) fn calibrator<'py>(&self, py: Python<'py>) -> Option<PyCalibratorCoefficients<'py>> {
        self.inner.calibrator.as_ref().map(|cal| {
            (
                cal.a.to_pyarray_bound(py),
                cal.b.to_pyarray_bound(py),
                cal.c.to_pyarray_bound(py),
            )
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "ENNParams(k={}, epi={:.4}, ale={:.4})",
            self.inner.k_num_neighbors,
            self.inner.epistemic_variance_scale,
            self.inner.aleatoric_variance_scale
        )
    }
}
