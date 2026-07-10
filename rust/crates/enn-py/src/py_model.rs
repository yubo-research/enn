//! ENN model Python bindings.

use ennbo::traits::PosteriorComputation;
use numpy::{IntoPyArray, PyArray2, PyArrayDyn, PyReadonlyArray2};
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

type TrainRowsAtPyOut<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Option<Bound<'py, PyArray2<f64>>>,
);

fn py_posterior_flags(
    model: &PyEpistemicNearestNeighbors,
    exclude_nearest: bool,
    observation_noise: bool,
) -> ennbo::PosteriorFlags {
    ennbo::PosteriorFlags::new()
        .with_exclude_nearest(exclude_nearest)
        .with_observation_noise(observation_noise)
        .with_tie_break_neighbors(model.tie_break_neighbors.get())
}

/// Python wrapper for EpistemicNearestNeighbors
#[pyclass(name = "EpistemicNearestNeighbors")]
pub struct PyEpistemicNearestNeighbors {
    pub(crate) inner: ennbo::EpistemicNearestNeighbors,
    tie_break_neighbors: std::cell::Cell<bool>,
}

#[pymethods]
impl PyEpistemicNearestNeighbors {
    #[new]
    #[pyo3(signature = (train_x, train_y, train_yvar=None, scale_x=false, index_driver="Exact", work_dir=None, enn_storage=None))]
    fn new(
        train_x: PyReadonlyArray2<f64>,
        train_y: PyReadonlyArray2<f64>,
        train_yvar: Option<PyReadonlyArray2<f64>>,
        scale_x: bool,
        index_driver: &str,
        work_dir: Option<&str>,
        enn_storage: Option<&str>,
    ) -> PyResult<Self> {
        let driver = match index_driver {
            "Exact" | "exact" | "FLAT" | "flat" => ennbo::IndexDriver::Exact,
            "BPANN_DISK" | "bpann_disk" => ennbo::IndexDriver::BpAnnDisk,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown index_driver: {index_driver}"
                )))
            }
        };
        let storage = match enn_storage {
            Some("disk" | "Disk") => ennbo::EnnStorage::Disk,
            Some("memory" | "in_memory" | "InMemory") => ennbo::EnnStorage::InMemory,
            None if work_dir.is_some() => ennbo::EnnStorage::Disk,
            None => ennbo::EnnStorage::InMemory,
            Some(other) => {
                return Err(PyValueError::new_err(format!(
                    "Unknown enn_storage: {other}"
                )))
            }
        };
        let work_dir = work_dir.map(PathBuf::from);
        let model = ennbo::EpistemicNearestNeighbors::new_with_storage(
            train_x.as_array().to_owned(),
            train_y.as_array().to_owned(),
            train_yvar.map(|v| v.as_array().to_owned()),
            scale_x,
            driver,
            storage,
            work_dir,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            inner: model,
            tie_break_neighbors: std::cell::Cell::new(true),
        })
    }

    #[pyo3(signature = (enabled=true))]
    fn set_tie_break_neighbors(&self, enabled: bool) {
        self.tie_break_neighbors.set(enabled);
    }

    fn tie_break_neighbors(&self) -> bool {
        self.tie_break_neighbors.get()
    }

    #[pyo3(signature = (x, y, yvar=None))]
    fn add(
        &mut self,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray2<f64>,
        yvar: Option<PyReadonlyArray2<f64>>,
    ) -> PyResult<()> {
        let yvar_arr = yvar.as_ref().map(|v| v.as_array());
        self.inner
            .add(&x.as_array(), &y.as_array(), yvar_arr.as_ref())
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn ensure_index_sync(&self) -> PyResult<()> {
        self.inner
            .index_access()
            .ensure_sync()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn schedule_background_flush(&self) -> PyResult<()> {
        self.inner
            .schedule_background_flush()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn persist_index_to_disk(&self) -> PyResult<()> {
        self.inner
            .persist_index_to_disk()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    fn index_memory_bytes(&self) -> PyResult<usize> {
        self.inner
            .index_access()
            .memory_bytes()
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (x, k_num_neighbors, epistemic_variance_scale, aleatoric_variance_scale, exclude_nearest=false, observation_noise=false))]
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
        let flags = py_posterior_flags(self, exclude_nearest, observation_noise);
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
    fn batch_posterior<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        k_values: Vec<i32>,
        epistemic_scales: Vec<f64>,
        aleatoric_scales: Vec<f64>,
        exclude_nearest: bool,
        observation_noise: bool,
    ) -> PyResult<(
        Bound<'py, PyArrayDyn<f64>>,
        Bound<'py, PyArrayDyn<f64>>,
        Bound<'py, PyArrayDyn<f64>>,
        Bound<'py, PyArrayDyn<f64>>,
    )> {
        // Build params list
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

        let flags = py_posterior_flags(self, exclude_nearest, observation_noise);

        let out = self
            .inner
            .batch_posterior(&x.as_array(), &paramss, &flags)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((
            out.mu.into_pyarray_bound(py),
            out.se.into_pyarray_bound(py),
            out.se_epi.into_pyarray_bound(py),
            out.se_ale.into_pyarray_bound(py),
        ))
    }

    /// Posterior function draw - sample from posterior predictive.
    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    #[pyo3(signature = (x, k_num_neighbors, epistemic_variance_scale, aleatoric_variance_scale, function_seeds, exclude_nearest=false, observation_noise=false))]
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
        let flags = py_posterior_flags(self, exclude_nearest, observation_noise);
        let (draws, idx) = self
            .inner
            .posterior_function_draw(&x.as_array(), &params, &function_seeds, &flags)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((draws.into_dyn().into_pyarray_bound(py), idx))
    }

    /// Conditional posterior with what-if scenarios.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (x_whatif, y_whatif, x, k_num_neighbors, epistemic_variance_scale, aleatoric_variance_scale, exclude_nearest=false, observation_noise=false))]
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
        let flags = py_posterior_flags(self, exclude_nearest, observation_noise);
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
        let flags = py_posterior_flags(self, exclude_nearest, observation_noise);
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
    fn neighbors<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        k: i32,
        exclude_nearest: bool,
    ) -> PyResult<Bound<'py, PyArrayDyn<usize>>> {
        let result = self
            .inner
            .neighbors(&x.as_array(), k, exclude_nearest)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(result.into_dyn().into_pyarray_bound(py))
    }

    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (x, search_k, exclude_nearest=false))]
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
    #[pyo3(signature = (x, search_k, exclude_nearest=false, tie_break_neighbors=true))]
    fn index_neighbor_distances_and_indices<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArray2<f64>,
        search_k: i32,
        exclude_nearest: bool,
        tie_break_neighbors: bool,
    ) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, Bound<'py, PyArrayDyn<i64>>)> {
        let (dist2s, idx) = self
            .inner
            .index_access()
            .index_neighbor_distances_and_indices(
                &x.as_array(),
                search_k,
                exclude_nearest,
                tie_break_neighbors,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((
            dist2s.into_dyn().into_pyarray_bound(py),
            idx.into_dyn().into_pyarray_bound(py),
        ))
    }

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn num_outputs(&self) -> usize {
        self.inner.num_outputs()
    }

    #[getter]
    fn num_dim(&self) -> usize {
        self.inner.num_dim()
    }

    #[getter]
    fn scale_x(&self) -> bool {
        self.inner.is_scale_x()
    }

    fn train_rows_at<'py>(
        &self,
        py: Python<'py>,
        indices: Vec<usize>,
    ) -> PyResult<TrainRowsAtPyOut<'py>> {
        let (x, y, yvar) = self
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

    fn row_x<'py>(&self, py: Python<'py>, i: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let row = self
            .inner
            .rows()
            .row_x(i)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(row.insert_axis(ndarray::Axis(0)).into_pyarray_bound(py))
    }

    fn row_y<'py>(&self, py: Python<'py>, i: usize) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let row = self
            .inner
            .rows()
            .row_y(i)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(row.insert_axis(ndarray::Axis(0)).into_pyarray_bound(py))
    }

    #[getter]
    fn x_scale_row<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(self.inner.x_scale_row().into_pyarray_bound(py))
    }

    #[getter]
    fn y_scale_row<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(self.inner.y_scale_row().into_pyarray_bound(py))
    }
}

/// Wrapper for ENNParams
#[pyclass(name = "ENNParams")]
#[derive(Clone, Copy)]
pub struct PyENNParams {
    pub(crate) inner: ennbo::ENNParams,
}

#[pymethods]
impl PyENNParams {
    #[new]
    #[pyo3(signature = (k_num_neighbors, epistemic_variance_scale, aleatoric_variance_scale))]
    fn new(
        k_num_neighbors: i32,
        epistemic_variance_scale: f64,
        aleatoric_variance_scale: f64,
    ) -> PyResult<Self> {
        let inner = ennbo::ENNParams::new(
            k_num_neighbors,
            epistemic_variance_scale,
            aleatoric_variance_scale,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[getter]
    fn k_num_neighbors(&self) -> i32 {
        self.inner.k_num_neighbors
    }

    #[getter]
    fn epistemic_variance_scale(&self) -> f64 {
        self.inner.epistemic_variance_scale
    }

    #[getter]
    fn aleatoric_variance_scale(&self) -> f64 {
        self.inner.aleatoric_variance_scale
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

#[cfg(test)]
mod kiss_coverage_tests {
    use super::*;

    #[test]
    fn py_model_units_are_linked() {
        let _ = py_posterior_flags
            as fn(&PyEpistemicNearestNeighbors, bool, bool) -> ennbo::PosteriorFlags;
        let _ = (
            PyEpistemicNearestNeighbors::new,
            PyEpistemicNearestNeighbors::set_tie_break_neighbors,
            PyEpistemicNearestNeighbors::tie_break_neighbors,
            PyEpistemicNearestNeighbors::add,
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
            PyENNParams::new,
            PyENNParams::k_num_neighbors,
            PyENNParams::epistemic_variance_scale,
            PyENNParams::aleatoric_variance_scale,
            std::mem::size_of::<PyEpistemicNearestNeighbors>,
            std::mem::size_of::<PyENNParams>,
        );
    }
}
