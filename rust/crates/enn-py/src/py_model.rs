use ennbo::traits::PosteriorComputation;
use numpy::{
    IntoPyArray, PyArray2, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::py_args::matrix_view;
use crate::py_params::make_params;

pub(crate) type PosteriorPyOut<'py> = (
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<f64>>,
    Option<Bound<'py, PyArrayDyn<i64>>>,
);

pub(crate) type TrainRowsAtPyOut<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Option<Bound<'py, PyArray2<f64>>>,
);

pub(crate) fn params_list(
    k_values: Vec<i32>,
    epistemic_scales: Vec<f64>,
    aleatoric_scales: Vec<f64>,
    taus: Option<Vec<f64>>,
) -> PyResult<Vec<ennbo::ENNParams>> {
    let n_params = k_values.len();
    let taus = taus.unwrap_or_else(|| vec![0.0; n_params]);
    if epistemic_scales.len() != n_params
        || aleatoric_scales.len() != n_params
        || taus.len() != n_params
    {
        return Err(PyValueError::new_err(
            "k_values, epistemic_scales, aleatoric_scales, and taus must have same length",
        ));
    }
    k_values
        .into_iter()
        .zip(epistemic_scales)
        .zip(aleatoric_scales)
        .zip(taus)
        .map(|(((k, epi), ale), tau)| make_params(k, epi, ale, tau))
        .collect()
}

fn py_posterior_flags(exclude_nearest: bool, observation_noise: bool) -> ennbo::PosteriorFlags {
    ennbo::PosteriorFlags::new()
        .with_exclude_nearest(exclude_nearest)
        .with_observation_noise(observation_noise)
}

#[pyclass(name = "EpistemicNearestNeighbors")]
pub struct PyEpistemicNearestNeighbors {
    pub(crate) inner: ennbo::EpistemicNearestNeighbors,
}

#[pymethods]
impl PyEpistemicNearestNeighbors {
    #[new]
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (train_x, train_y, work_dir, train_yvar=None, weighting=false, y_bounds=None, tied_dims=None, aggregate_duplicates=false))]
    fn new(
        train_x: PyReadonlyArrayDyn<f64>,
        train_y: PyReadonlyArrayDyn<f64>,
        work_dir: &str,
        train_yvar: Option<PyReadonlyArrayDyn<f64>>,
        weighting: bool,
        y_bounds: Option<PyReadonlyArray2<f64>>,
        tied_dims: Option<Vec<Vec<usize>>>,
        aggregate_duplicates: bool,
    ) -> PyResult<Self> {
        let train_x = matrix_view("train_x", &train_x)?.to_owned();
        let train_y = matrix_view("train_y", &train_y)?.to_owned();
        let layout = ennbo::EnnLayout::new(work_dir, weighting)
            .map_err(|e| PyValueError::new_err(e.to_string()))?
            .with_aggregate_duplicates(aggregate_duplicates);
        let y_bounds = y_bounds.map(|v| v.as_array().to_owned());
        let train_yvar = train_yvar
            .as_ref()
            .map(|v| matrix_view("train_yvar", v).map(|m| m.to_owned()))
            .transpose()?;
        let inner = ennbo::EpistemicNearestNeighbors::new_configured(
            train_x,
            train_y,
            train_yvar,
            layout,
            y_bounds,
            tied_dims.unwrap_or_default(),
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[staticmethod]
    fn open(work_dir: &str) -> PyResult<Self> {
        let inner = ennbo::EpistemicNearestNeighbors::open(work_dir)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (x_scale, rebuild=false))]
    fn set_metric_scale(&mut self, x_scale: PyReadonlyArray1<f64>, rebuild: bool) -> PyResult<()> {
        self.inner
            .set_metric_scale(x_scale.as_array().to_owned(), rebuild)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    #[pyo3(signature = (train_x, train_y, train_yvar=None))]
    fn add(
        &mut self,
        train_x: PyReadonlyArrayDyn<f64>,
        train_y: PyReadonlyArrayDyn<f64>,
        train_yvar: Option<PyReadonlyArrayDyn<f64>>,
    ) -> PyResult<()> {
        let train_yvar = train_yvar
            .as_ref()
            .map(|v| matrix_view("train_yvar", v))
            .transpose()?;
        self.inner
            .add(
                &matrix_view("train_x", &train_x)?,
                &matrix_view("train_y", &train_y)?,
                train_yvar.as_ref(),
            )
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
    #[pyo3(signature = (x, params, exclude_nearest=false, observation_noise=false))]
    fn posterior<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArrayDyn<f64>,
        params: PyENNParams,
        exclude_nearest: bool,
        observation_noise: bool,
    ) -> PyResult<PosteriorPyOut<'py>> {
        let flags = py_posterior_flags(exclude_nearest, observation_noise);
        let out = self
            .inner
            .posterior(&matrix_view("x", &x)?, &params.inner, &flags)
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
    #[pyo3(signature = (x, paramss, exclude_nearest=false, observation_noise=false))]
    fn batch_posterior<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArrayDyn<f64>,
        paramss: Vec<PyENNParams>,
        exclude_nearest: bool,
        observation_noise: bool,
    ) -> PyResult<PosteriorPyOut<'py>> {
        let paramss: Vec<ennbo::ENNParams> = paramss.into_iter().map(|p| p.inner).collect();

        let flags = py_posterior_flags(exclude_nearest, observation_noise);

        let out = self
            .inner
            .batch_posterior(&matrix_view("x", &x)?, &paramss, &flags)
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
    #[pyo3(signature = (x, params, function_seeds, exclude_nearest=false, observation_noise=false))]
    fn posterior_function_draw<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArrayDyn<f64>,
        params: PyENNParams,
        function_seeds: Vec<i64>,
        exclude_nearest: bool,
        observation_noise: bool,
    ) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, Vec<Vec<usize>>)> {
        let flags = py_posterior_flags(exclude_nearest, observation_noise);
        let (draws, idx) = self
            .inner
            .posterior_function_draw(
                &matrix_view("x", &x)?,
                &params.inner,
                &function_seeds,
                &flags,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((draws.into_dyn().into_pyarray_bound(py), idx))
    }

    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (x_whatif, y_whatif, x, params, exclude_nearest=false, observation_noise=false))]
    fn conditional_posterior<'py>(
        &self,
        py: Python<'py>,
        x_whatif: PyReadonlyArrayDyn<f64>,
        y_whatif: PyReadonlyArrayDyn<f64>,
        x: PyReadonlyArrayDyn<f64>,
        params: PyENNParams,
        exclude_nearest: bool,
        observation_noise: bool,
    ) -> PyResult<PosteriorPyOut<'py>> {
        let flags = py_posterior_flags(exclude_nearest, observation_noise);
        let out = self
            .inner
            .conditional_posterior(
                &matrix_view("x_whatif", &x_whatif)?,
                &matrix_view("y_whatif", &y_whatif)?,
                &matrix_view("x", &x)?,
                &params.inner,
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

    #[allow(clippy::too_many_arguments, clippy::type_complexity)]
    #[pyo3(signature = (x_whatif, y_whatif, x, params, function_seeds, exclude_nearest=false, observation_noise=false))]
    fn conditional_posterior_function_draw<'py>(
        &self,
        py: Python<'py>,
        x_whatif: PyReadonlyArrayDyn<f64>,
        y_whatif: PyReadonlyArrayDyn<f64>,
        x: PyReadonlyArrayDyn<f64>,
        params: PyENNParams,
        function_seeds: Vec<i64>,
        exclude_nearest: bool,
        observation_noise: bool,
    ) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, Vec<Vec<usize>>)> {
        let flags = py_posterior_flags(exclude_nearest, observation_noise);
        let (draws, idx) = self
            .inner
            .conditional_posterior_function_draw(
                &matrix_view("x_whatif", &x_whatif)?,
                &matrix_view("y_whatif", &y_whatif)?,
                &matrix_view("x", &x)?,
                &params.inner,
                &function_seeds,
                &flags,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((draws.into_dyn().into_pyarray_bound(py), idx))
    }

    #[pyo3(signature = (x, k, exclude_nearest=false))]
    fn neighbors<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArrayDyn<f64>,
        k: i32,
        exclude_nearest: bool,
    ) -> PyResult<Bound<'py, PyArrayDyn<usize>>> {
        let result = self
            .inner
            .neighbors(&matrix_view("x", &x)?, k, exclude_nearest)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(result.into_dyn().into_pyarray_bound(py))
    }

    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (x, search_k, exclude_nearest=false))]
    fn neighbor_distances_and_indices<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArrayDyn<f64>,
        search_k: i32,
        exclude_nearest: bool,
    ) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, Bound<'py, PyArrayDyn<i64>>)> {
        let (dist2s, idx) = self
            .inner
            .index_access()
            .neighbor_distances_and_indices(&matrix_view("x", &x)?, search_k, exclude_nearest)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((
            dist2s.into_dyn().into_pyarray_bound(py),
            idx.into_dyn().into_pyarray_bound(py),
        ))
    }

    #[allow(clippy::type_complexity)]
    #[pyo3(signature = (x, search_k, exclude_nearest=false))]
    fn index_neighbor_distances_and_indices<'py>(
        &self,
        py: Python<'py>,
        x: PyReadonlyArrayDyn<f64>,
        search_k: i32,
        exclude_nearest: bool,
    ) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, Bound<'py, PyArrayDyn<i64>>)> {
        let (dist2s, idx) = self
            .inner
            .index_access()
            .index_neighbor_distances_and_indices(&matrix_view("x", &x)?, search_k, exclude_nearest)
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
    fn weighting(&self) -> bool {
        self.inner.weighting()
    }

    #[getter]
    fn aggregate_duplicates(&self) -> bool {
        self.inner.aggregates_duplicates()
    }

    #[getter]
    fn metric_learning_auto(&self) -> bool {
        self.inner.metric_learning_auto()
    }

    fn tied_groups(&self) -> Vec<Vec<usize>> {
        self.inner.tied_groups().to_vec()
    }

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
            .row_y_natural(i)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(row.insert_axis(ndarray::Axis(0)).into_pyarray_bound(py))
    }

    #[getter]
    fn y_bounds<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray2<f64>>> {
        Ok(self.inner.y_bounds().clone().into_pyarray_bound(py))
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

#[pyfunction]
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

pub use crate::py_params::PyENNParams;

#[cfg(test)]
mod kiss_coverage_tests {
    use super::*;

    #[test]
    fn py_model_units_are_linked() {
        let _ = py_posterior_flags as fn(bool, bool) -> ennbo::PosteriorFlags;
        let _ = (
            PyEpistemicNearestNeighbors::new,
            PyEpistemicNearestNeighbors::open,
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
            PyEpistemicNearestNeighbors::train_rows_at,
            PyEpistemicNearestNeighbors::row_x,
            PyEpistemicNearestNeighbors::row_y,
            PyEpistemicNearestNeighbors::x_scale_row,
            PyEpistemicNearestNeighbors::y_scale_row,
            PyEpistemicNearestNeighbors::y_bounds,
            train_rows_at_warped,
            PyENNParams::new,
            PyENNParams::k_num_neighbors,
            PyENNParams::epistemic_variance_scale,
            PyENNParams::aleatoric_variance_scale,
            PyENNParams::calibrator,
            PyEpistemicNearestNeighbors::__len__,
            params_list,
            std::mem::size_of::<PyEpistemicNearestNeighbors>,
            std::mem::size_of::<PyENNParams>,
        );
    }
}
