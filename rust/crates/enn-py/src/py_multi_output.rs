use crate::py_args::{count_at_least, matrix_or, value_error};
use ennbo::semantic_multi_output::{OutOfFold, SemanticMultiOutputENN};
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::exceptions::{PyAttributeError, PyRuntimeError};
use pyo3::prelude::*;

const TRAINING_MESSAGE: &str =
    "training data must be finite matching matrices with at least four rows";
const QUERY_MESSAGE: &str = "query must be a finite matrix with the fitted input dimension";

type MarginalArrays<'py> = (Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>);

fn marginals<'py>(
    py: Python<'py>,
    result: Result<(Array2<f64>, Array2<f64>), ennbo::error::ENNError>,
) -> PyResult<MarginalArrays<'py>> {
    let (mu, se) = result.map_err(value_error)?;
    Ok((mu.into_pyarray_bound(py), se.into_pyarray_bound(py)))
}

#[pyclass(name = "SemanticMultiOutputENN")]
pub struct PySemanticMultiOutputENN {
    inner: SemanticMultiOutputENN,
}

impl PySemanticMultiOutputENN {
    fn out_of_fold(&self) -> PyResult<&OutOfFold> {
        self.inner
            .out_of_fold()
            .ok_or_else(|| PyAttributeError::new_err("fit must be called first"))
    }

    fn query(&self, x: &PyReadonlyArrayDyn<'_, f64>) -> PyResult<Array2<f64>> {
        if !self.inner.is_fitted() {
            return Err(PyRuntimeError::new_err(
                "fit must be called before posterior",
            ));
        }
        matrix_or(x, QUERY_MESSAGE)
    }
}

#[pymethods]
impl PySemanticMultiOutputENN {
    #[new]
    #[pyo3(signature = (loadings, alpha, k_num_neighbors, epistemic_variance_scale, aleatoric_variance_scale, folds, seed))]
    fn new(
        loadings: PyReadonlyArray2<f64>,
        alpha: f64,
        k_num_neighbors: i32,
        epistemic_variance_scale: f64,
        aleatoric_variance_scale: f64,
        folds: &Bound<'_, PyAny>,
        seed: u64,
    ) -> PyResult<Self> {
        let params = ennbo::ENNParams::new(
            k_num_neighbors,
            epistemic_variance_scale,
            aleatoric_variance_scale,
        )
        .map_err(value_error)?;
        let folds = count_at_least(folds, 2, "folds must be an integer at least two")?;
        SemanticMultiOutputENN::new(loadings.as_array().to_owned(), alpha, params, folds, seed)
            .map(|inner| Self { inner })
            .map_err(value_error)
    }

    fn fit(
        &mut self,
        py: Python<'_>,
        x: PyReadonlyArrayDyn<f64>,
        y: PyReadonlyArrayDyn<f64>,
    ) -> PyResult<()> {
        let x = matrix_or(&x, TRAINING_MESSAGE)?;
        let y = matrix_or(&y, TRAINING_MESSAGE)?;
        py.allow_threads(|| self.inner.fit(x, y))
            .map_err(value_error)
    }

    fn posterior<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArrayDyn<f64>,
    ) -> PyResult<MarginalArrays<'py>> {
        let x = self.query(&x)?;
        let result = py.allow_threads(|| self.inner.posterior(&x));
        marginals(py, result)
    }

    fn uncalibrated_posterior<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArrayDyn<f64>,
    ) -> PyResult<MarginalArrays<'py>> {
        let x = self.query(&x)?;
        let result = py.allow_threads(|| self.inner.uncalibrated_posterior(&x));
        marginals(py, result)
    }

    #[getter]
    fn oof_mean<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(self.out_of_fold()?.mean.clone().into_pyarray_bound(py))
    }

    #[getter]
    fn oof_variance<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(self.out_of_fold()?.variance.clone().into_pyarray_bound(py))
    }

    #[getter]
    fn fold_ids<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray1<i64>>> {
        let ids: Vec<i64> = self
            .out_of_fold()?
            .fold_ids
            .iter()
            .map(|v| *v as i64)
            .collect();
        Ok(PyArray1::from_vec_bound(py, ids))
    }
}
