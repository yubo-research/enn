use numpy::{PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, ToPyArray};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::sync::Arc;

#[pyclass(name = "ENNOutputBlend", frozen)]
#[derive(Clone)]
pub struct PyOutputBlend {
    pub(crate) inner: Arc<ennbo::OutputBlend>,
}

#[pymethods]
impl PyOutputBlend {
    #[new]
    #[pyo3(signature = (alpha, projector, center, scale))]
    pub(crate) fn new(
        alpha: PyReadonlyArray1<f64>,
        projector: PyReadonlyArray2<f64>,
        center: PyReadonlyArray1<f64>,
        scale: PyReadonlyArray1<f64>,
    ) -> PyResult<Self> {
        let blend = ennbo::OutputBlend::new(
            alpha.as_array().to_owned(),
            projector.as_array().to_owned(),
            center.as_array().to_owned(),
            scale.as_array().to_owned(),
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(blend),
        })
    }

    #[getter]
    pub(crate) fn alpha<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.alpha.to_pyarray_bound(py)
    }

    #[getter]
    pub(crate) fn projector<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        self.inner.projector.to_pyarray_bound(py)
    }

    #[getter]
    pub(crate) fn center<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.center.to_pyarray_bound(py)
    }

    #[getter]
    pub(crate) fn scale<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        self.inner.scale.to_pyarray_bound(py)
    }

    pub(crate) fn __eq__(&self, other: &Self) -> bool {
        self.inner == other.inner
    }

    fn __repr__(&self) -> String {
        format!("ENNOutputBlend(alpha={})", self.inner.alpha)
    }
}

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

pub(crate) fn inner_blend(blend: Option<PyOutputBlend>) -> Option<Arc<ennbo::OutputBlend>> {
    blend.map(|b| b.inner)
}

pub(crate) fn params_with_blend(
    k_num_neighbors: i32,
    epistemic_variance_scale: f64,
    aleatoric_variance_scale: f64,
    output_blend: Option<PyOutputBlend>,
) -> PyResult<ennbo::ENNParams> {
    ennbo::ENNParams::new(
        k_num_neighbors,
        epistemic_variance_scale,
        aleatoric_variance_scale,
    )
    .map(|p| p.with_output_blend(inner_blend(output_blend)))
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
    #[pyo3(signature = (k_num_neighbors, epistemic_variance_scale, aleatoric_variance_scale, output_blend=None, calibrator=None))]
    pub(crate) fn new(
        k_num_neighbors: i32,
        epistemic_variance_scale: f64,
        aleatoric_variance_scale: f64,
        output_blend: Option<PyOutputBlend>,
        calibrator: Option<PyCalibratorArgument<'_>>,
    ) -> PyResult<Self> {
        let inner = params_with_blend(
            k_num_neighbors,
            epistemic_variance_scale,
            aleatoric_variance_scale,
            output_blend,
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
    pub(crate) fn output_blend(&self) -> Option<PyOutputBlend> {
        self.inner
            .output_blend
            .clone()
            .map(|inner| PyOutputBlend { inner })
    }

    #[getter]
    pub(crate) fn calibrator<'py>(
        &self,
        py: Python<'py>,
    ) -> Option<PyCalibratorCoefficients<'py>> {
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
