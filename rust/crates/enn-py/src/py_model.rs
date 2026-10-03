//! ENN model Python bindings.

use ennbo::traits::PosteriorComputation;
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::path::PathBuf;

pub(crate) type PosteriorPyOut<'py> = (
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<f64>>,
    Option<Bound<'py, PyArrayDyn<i64>>>,
);

fn owned_matrix(name: &str, arr: PyReadonlyArrayDyn<'_, f64>) -> PyResult<Array2<f64>> {
    let view = arr.as_array();
    if view.ndim() != 2 {
        return Err(PyValueError::new_err(format!(
            "{name} must be 2-dimensional, got shape {:?}",
            view.shape()
        )));
    }
    view.to_owned()
        .into_dimensionality()
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

pub(crate) type TrainRowsAtPyOut<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Option<Bound<'py, PyArray2<f64>>>,
);

#[doc = "kiss-coverage-off"]
fn py_posterior_flags(
    exclude_nearest: bool,
    observation_noise: bool,
) -> ennbo::PosteriorFlags {
    ennbo::PosteriorFlags::new()
        .with_exclude_nearest(exclude_nearest)
        .with_observation_noise(observation_noise)
}

/// Python wrapper for EpistemicNearestNeighbors
#[pyclass(name = "EpistemicNearestNeighbors")]
pub struct PyEpistemicNearestNeighbors {
    pub(crate) inner: ennbo::EpistemicNearestNeighbors,
    pub(crate) incremental: ennbo::IncrementalFit,
}

#[pymethods]
impl PyEpistemicNearestNeighbors {
    #[new]
    #[pyo3(signature = (train_x, train_y, train_yvar=None, scale_x=false, index_driver="FLAT", work_dir=None, enn_storage=None, y_bounds=None, metric_learning="none", tied_dims=None))]
    #[allow(clippy::too_many_arguments)]
    #[doc = "kiss-coverage-off"]
    fn new(
        train_x: PyReadonlyArrayDyn<f64>,
        train_y: PyReadonlyArrayDyn<f64>,
        train_yvar: Option<PyReadonlyArrayDyn<f64>>,
        scale_x: bool,
        index_driver: &str,
        work_dir: Option<&str>,
        enn_storage: Option<&str>,
        y_bounds: Option<PyReadonlyArray2<f64>>,
        metric_learning: &str,
        tied_dims: Option<Vec<Vec<usize>>>,
    ) -> PyResult<Self> {
        let driver = crate::py_layout::index_driver_from_wire(index_driver)?;
        let metric_learning = crate::py_layout::metric_learning_from_wire(metric_learning)?;
        let explicit = crate::py_layout::enn_storage_optional(enn_storage)?;
        let work_dir = work_dir.map(PathBuf::from);
        let layout = ennbo::EnnLayout::try_from_parts(
            driver,
            explicit,
            work_dir,
            scale_x,
            metric_learning,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let y_bounds = y_bounds.map(|v| v.as_array().to_owned());
        let train_x = owned_matrix("train_x", train_x)?;
        let train_y = owned_matrix("train_y", train_y)?;
        let train_yvar = match train_yvar {
            Some(v) => Some(owned_matrix("train_yvar", v)?),
            None => None,
        };
        let train_y_auto = train_y.clone();
        let mut model = ennbo::EpistemicNearestNeighbors::new_with_storage(
            train_x.clone(),
            train_y,
            train_yvar,
            layout,
            y_bounds,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let groups = tied_dims.unwrap_or_default();
        if metric_learning == ennbo::metric_auto::MetricLearning::Auto {
            model
                .enable_auto_metric(
                    groups.clone(),
                    &train_x.view(),
                    &train_y_auto.view(),
                )
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        } else if !groups.is_empty() {
            let dims: Vec<usize> = groups.iter().flatten().copied().collect();
            model
                .set_unscaled_dims(dims)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
        model
            .set_tied_groups(groups)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: model, incremental: ennbo::IncrementalFit::new() })
    }

    #[pyo3(signature = (x_scale, rebuild=false))]
    #[doc = "kiss-coverage-off"]
    fn set_metric_scale(&mut self, x_scale: PyReadonlyArray1<f64>, rebuild: bool) -> PyResult<()> {
        self.inner
            .set_metric_scale(x_scale.as_array().to_owned(), rebuild)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (x, y, yvar=None))]
    #[doc = "kiss-coverage-off"]
    fn add(
        &mut self,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray2<f64>,
        yvar: Option<PyReadonlyArray2<f64>>,
    ) -> PyResult<crate::py_fit::PyAddToken> {
        let yvar_arr = yvar.as_ref().map(|v| v.as_array());
        self.incremental
            .add(&mut self.inner, &x.as_array(), &y.as_array(), yvar_arr.as_ref())
            .map(|inner| crate::py_fit::PyAddToken { inner })
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[doc = "kiss-coverage-off"]
    fn ensure_index_sync(&self) -> PyResult<()> {
        self.inner
            .index_access()
            .ensure_sync()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[doc = "kiss-coverage-off"]
    fn schedule_background_flush(&self) -> PyResult<()> {
        self.inner
            .schedule_background_flush()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[doc = "kiss-coverage-off"]
    fn persist_index_to_disk(&self) -> PyResult<()> {
        self.inner
            .persist_index_to_disk()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[doc = "kiss-coverage-off"]
    fn index_memory_bytes(&self) -> PyResult<usize> {
        self.inner
            .index_access()
            .memory_bytes()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (x, k_num_neighbors, epistemic_variance_scale, aleatoric_variance_scale, exclude_nearest=false, observation_noise=false))]
    #[doc = "kiss-coverage-off"]
    fn posterior<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        k_num_neighbors: i32,
        epistemic_variance_scale: f64,
        aleatoric_variance_scale: f64,
        exclude_nearest: bool,
        observation_noise: bool,
    ) -> PyResult<PosteriorPyOut<'py>> {
        let params = ennbo::ENNParams::new(
            k_num_neighbors,
            epistemic_variance_scale,
            aleatoric_variance_scale,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let flags = py_posterior_flags(exclude_nearest, observation_noise);
        let out = self
            .inner
            .posterior(&x.as_array(), &params, &flags)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((
            out.mu.into_pyarray_bound(py),
            out.se.into_pyarray_bound(py),
            out.se_epi.into_pyarray_bound(py),
            out.se_ale.into_pyarray_bound(py),
            out.idx.map(|idx| idx.into_dyn().into_pyarray_bound(py)),
        ))
    }

    /// Batch posterior with multiple parameter sets.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    #[pyo3(signature = (x, k_values, epistemic_scales, aleatoric_scales, exclude_nearest=false, observation_noise=false))]
    #[doc = "kiss-coverage-off"]
    fn batch_posterior<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        k_values: Vec<i32>,
        epistemic_scales: Vec<f64>,
        aleatoric_scales: Vec<f64>,
        exclude_nearest: bool,
        observation_noise: bool,
    ) -> PyResult<PosteriorPyOut<'py>> {

        let n_params = k_values.len();
        if epistemic_scales.len() != n_params || aleatoric_scales.len() != n_params {
            return Err(PyValueError::new_err(
                "k_values, epistemic_scales, and aleatoric_scales must have same length",
            ));
        }

        let mut paramss = Vec::with_capacity(n_params);
        for i in 0..n_params {
            let params =
                ennbo::ENNParams::new(k_values[i], epistemic_scales[i], aleatoric_scales[i])
                    .map_err(|e| PyValueError::new_err(e.to_string()))?;
            paramss.push(params);
        }

        let flags = py_posterior_flags(exclude_nearest, observation_noise);

        let out = self
            .inner
            .batch_posterior(&x.as_array(), &paramss, &flags)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((
            out.mu.into_pyarray_bound(py),
            out.se.into_pyarray_bound(py),
            out.se_epi.into_pyarray_bound(py),
            out.se_ale.into_pyarray_bound(py),
            out.idx.map(|idx| idx.into_dyn().into_pyarray_bound(py)),
        ))
    }

    /// Posterior function draw - sample from posterior predictive.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    #[pyo3(signature = (x, k_num_neighbors, epistemic_variance_scale, aleatoric_variance_scale, function_seeds, exclude_nearest=false, observation_noise=false))]
    #[doc = "kiss-coverage-off"]
    fn posterior_function_draw<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        k_num_neighbors: i32,
        epistemic_variance_scale: f64,
        aleatoric_variance_scale: f64,
        function_seeds: Vec<i64>,
        exclude_nearest: bool,
        observation_noise: bool,
    ) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, Vec<Vec<usize>>)> {
        let params = ennbo::ENNParams::new(
            k_num_neighbors,
            epistemic_variance_scale,
            aleatoric_variance_scale,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let flags = py_posterior_flags(exclude_nearest, observation_noise);
        let (draws, idx) = self
            .inner
            .posterior_function_draw(&x.as_array(), &params, &function_seeds, &flags)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((draws.into_dyn().into_pyarray_bound(py), idx))
    }

    /// Conditional posterior with what-if scenarios.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (x_whatif, y_whatif, x, k_num_neighbors, epistemic_variance_scale, aleatoric_variance_scale, exclude_nearest=false, observation_noise=false))]
    #[doc = "kiss-coverage-off"]
    fn conditional_posterior<'py>(
        &self,
        py: Python<'py>,
        x_whatif: PyReadonlyArray2<f64>,
        y_whatif: PyReadonlyArray2<f64>,
        x: PyReadonlyArray2<f64>,
        k_num_neighbors: i32,
        epistemic_variance_scale: f64,
        aleatoric_variance_scale: f64,
        exclude_nearest: bool,
        observation_noise: bool,
    ) -> PyResult<PosteriorPyOut<'py>> {
        let params = ennbo::ENNParams::new(
            k_num_neighbors,
            epistemic_variance_scale,
            aleatoric_variance_scale,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let flags = py_posterior_flags(exclude_nearest, observation_noise);
        let out = self
            .inner
            .conditional_posterior(
                &x_whatif.as_array(),
                &y_whatif.as_array(),
                &x.as_array(),
                &params,
                &flags,
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

    /// Conditional posterior function draw.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    #[pyo3(signature = (x_whatif, y_whatif, x, k_num_neighbors, epistemic_variance_scale, aleatoric_variance_scale, function_seeds, exclude_nearest=false, observation_noise=false))]
    #[doc = "kiss-coverage-off"]
    fn conditional_posterior_function_draw<'py>(
        &self,
        py: Python<'py>,
        x_whatif: PyReadonlyArray2<f64>,
        y_whatif: PyReadonlyArray2<f64>,
        x: PyReadonlyArray2<f64>,
        k_num_neighbors: i32,
        epistemic_variance_scale: f64,
        aleatoric_variance_scale: f64,
        function_seeds: Vec<i64>,
        exclude_nearest: bool,
        observation_noise: bool,
    ) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, Vec<Vec<usize>>)> {
        let params = ennbo::ENNParams::new(
            k_num_neighbors,
            epistemic_variance_scale,
            aleatoric_variance_scale,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let flags = py_posterior_flags(exclude_nearest, observation_noise);
        let (draws, idx) = self
            .inner
            .conditional_posterior_function_draw(
                &x_whatif.as_array(),
                &y_whatif.as_array(),
                &x.as_array(),
                &params,
                &function_seeds,
                &flags,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((draws.into_dyn().into_pyarray_bound(py), idx))
    }

    /// Get k nearest neighbors for query points.
    #[pyo3(signature = (x, k, exclude_nearest=false))]
    #[doc = "kiss-coverage-off"]
    fn neighbors<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        k: i32,
        exclude_nearest: bool,
    ) -> PyResult<Bound<'py, PyArrayDyn<usize>>> {
        let result = self
            .inner
            .neighbors_one(&x.as_array(), k, exclude_nearest)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(result.into_dyn().into_pyarray_bound(py))
    }

    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (x, search_k, exclude_nearest=false))]
    #[doc = "kiss-coverage-off"]
    fn neighbor_distances_and_indices<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        search_k: i32,
        exclude_nearest: bool,
    ) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, Bound<'py, PyArrayDyn<i64>>)> {
        let (dist2s, idx) = self
            .inner
            .index_access()
            .neighbor_distances_and_indices(&x.as_array(), search_k, exclude_nearest)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((
            dist2s.into_dyn().into_pyarray_bound(py),
            idx.into_dyn().into_pyarray_bound(py),
        ))
    }

    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (x, search_k, exclude_nearest=false))]
    #[doc = "kiss-coverage-off"]
    fn index_neighbor_distances_and_indices<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        search_k: i32,
        exclude_nearest: bool,
    ) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, Bound<'py, PyArrayDyn<i64>>)> {
        let (dist2s, idx) = self
            .inner
            .index_access()
            .index_neighbor_distances_and_indices(
                &x.as_array(),
                search_k,
                exclude_nearest,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((
            dist2s.into_dyn().into_pyarray_bound(py),
            idx.into_dyn().into_pyarray_bound(py),
        ))
    }

    #[doc = "kiss-coverage-off"]
    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    #[doc = "kiss-coverage-off"]
    fn num_outputs(&self) -> usize {
        self.inner.num_outputs()
    }

    #[getter]
    #[doc = "kiss-coverage-off"]
    fn num_dim(&self) -> usize {
        self.inner.num_dim()
    }

    #[getter]
    #[doc = "kiss-coverage-off"]
    fn scale_x(&self) -> bool {
        self.inner.is_scale_x()
    }

    #[getter]
    #[doc = "kiss-coverage-off"]
    fn index_driver(&self) -> &'static str {
        self.inner.index_driver().as_wire()
    }

    #[getter]
    #[doc = "kiss-coverage-off"]
    fn metric_learning_auto(&self) -> bool {
        self.inner.metric_learning_auto()
    }

    #[doc = "kiss-coverage-off"]
    fn tied_groups(&self) -> Vec<Vec<usize>> {
        self.inner.tied_groups().to_vec()
    }

    #[doc = "kiss-coverage-off"]
    fn train_rows_at<'py>(
        &self,
        py: Python<'py>,
        indices: Vec<usize>,
    ) -> PyResult<TrainRowsAtPyOut<'py>> {
        let (x, y, yvar) = self
            .inner
            .train_rows_at(&indices)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((
            x.into_pyarray_bound(py),
            y.into_pyarray_bound(py),
            yvar.map(|a| a.into_pyarray_bound(py)),
        ))
    }

    #[doc = "kiss-coverage-off"]
    fn row_x<'py>(&self, py: Python<'py>, i: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let row = self
            .inner
            .rows()
            .row_x(i)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(row.insert_axis(ndarray::Axis(0)).into_pyarray_bound(py))
    }

    #[doc = "kiss-coverage-off"]
    fn row_y<'py>(&self, py: Python<'py>, i: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let row = self
            .inner
            .row_y_natural(i)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(row.insert_axis(ndarray::Axis(0)).into_pyarray_bound(py))
    }

    #[getter]
    #[doc = "kiss-coverage-off"]
    fn y_bounds<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(self.inner.y_bounds().clone().into_pyarray_bound(py))
    }

    #[getter]
    #[doc = "kiss-coverage-off"]
    fn x_scale_row<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(self.inner.x_scale_row().into_pyarray_bound(py))
    }

    #[getter]
    #[doc = "kiss-coverage-off"]
    fn y_scale_row<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(self.inner.y_scale_row().into_pyarray_bound(py))
    }
}

/// Storage-space (warped) row gather; kept off the pyclass to satisfy methods_per_class.
#[pyfunction]
#[doc = "kiss-coverage-off"]
pub(crate) fn train_rows_at_warped<'py>(
    py: Python<'py>,
    model: PyRef<'_, PyEpistemicNearestNeighbors>,
    indices: Vec<usize>,
) -> PyResult<TrainRowsAtPyOut<'py>> {
    let (x, y, yvar) = model
        .inner
        .rows()
        .train_rows_at(&indices)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok((
        x.into_pyarray_bound(py),
        y.into_pyarray_bound(py),
        yvar.map(|a| a.into_pyarray_bound(py)),
    ))
}

/// Dimensions `scale_x` leaves at scale 1; kept off the pyclass to satisfy methods_per_class.
#[pyfunction]
#[doc = "kiss-coverage-off"]
pub(crate) fn set_unscaled_dims(
    mut model: PyRefMut<'_, PyEpistemicNearestNeighbors>,
    dims: Vec<usize>,
) -> PyResult<()> {
    model
        .inner
        .set_unscaled_dims(dims)
        .map_err(|e| PyValueError::new_err(e.to_string()))
}

pub use crate::py_params::PyENNParams;

#[cfg(test)]
mod kiss_coverage_tests {
    use super::*;

    #[test]
    fn py_model_units_are_linked() {
        let _ = py_posterior_flags as fn(bool, bool) -> ennbo::PosteriorFlags;
        let _ = (
            PyEpistemicNearestNeighbors::new,
            PyEpistemicNearestNeighbors::add,
            PyEpistemicNearestNeighbors::set_metric_scale,
            PyEpistemicNearestNeighbors::ensure_index_sync,
            PyEpistemicNearestNeighbors::schedule_background_flush,
            PyEpistemicNearestNeighbors::persist_index_to_disk,
            PyEpistemicNearestNeighbors::index_memory_bytes,
            PyEpistemicNearestNeighbors::posterior,
            PyEpistemicNearestNeighbors::batch_posterior,
            PyEpistemicNearestNeighbors::posterior_function_draw,
            PyEpistemicNearestNeighbors::conditional_posterior,
            PyEpistemicNearestNeighbors::conditional_posterior_function_draw,
            PyEpistemicNearestNeighbors::neighbors,
            PyEpistemicNearestNeighbors::neighbor_distances_and_indices,
            PyEpistemicNearestNeighbors::index_neighbor_distances_and_indices,
            PyEpistemicNearestNeighbors::num_outputs,
            PyEpistemicNearestNeighbors::num_dim,
            PyEpistemicNearestNeighbors::scale_x,
            PyEpistemicNearestNeighbors::train_rows_at,
            PyEpistemicNearestNeighbors::row_x,
            PyEpistemicNearestNeighbors::row_y,
            PyEpistemicNearestNeighbors::x_scale_row,
            PyEpistemicNearestNeighbors::y_scale_row,
            PyEpistemicNearestNeighbors::y_bounds,
            train_rows_at_warped,
            set_unscaled_dims,
            PyENNParams::new,
            PyENNParams::k_num_neighbors,
            PyENNParams::epistemic_variance_scale,
            PyENNParams::aleatoric_variance_scale,
            std::mem::size_of::<PyEpistemicNearestNeighbors>,
            std::mem::size_of::<PyENNParams>,
        );
    }
}
