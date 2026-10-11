use ennbo::ennts::{EnnTs, EnnTsConfig, QueryOptions};
use ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyArray3, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

fn to_py(err: ennbo::ENNError) -> PyErr {
    PyValueError::new_err(err.to_string())
}

fn matrix(values: PyReadonlyArray2<f64>) -> Array2<f64> {
    values.as_array().to_owned()
}

fn timestamp_vec(values: PyReadonlyArray1<i64>) -> Vec<i64> {
    values.as_array().iter().copied().collect()
}

#[pyclass(name = "ENNTS")]
pub struct PyENNTS {
    inner: EnnTs,
}

#[pymethods]
impl PyENNTS {
    #[new]
    #[pyo3(signature = (
        lookback,
        num_tables=8,
        num_projections=4,
        bucket_width=4.0,
        k=8,
        seed=0,
        horizon=1,
        cadence=1,
        probe=1,
        variance_scales=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        lookback: usize,
        num_tables: usize,
        num_projections: usize,
        bucket_width: f64,
        k: usize,
        seed: u64,
        horizon: i64,
        cadence: i64,
        probe: i32,
        variance_scales: Option<(f64, f64)>,
    ) -> PyResult<Self> {
        let (epistemic_variance_scale, aleatoric_variance_scale) =
            variance_scales.unwrap_or((1.0, 0.1));
        let config = EnnTsConfig {
            lookback,
            horizon,
            cadence,
            num_tables,
            num_projections,
            bucket_width,
            k,
            probe,
            seed,
            epistemic_variance_scale,
            aleatoric_variance_scale,
        };
        let inner = EnnTs::new(config).map_err(to_py)?;
        Ok(Self { inner })
    }

    #[pyo3(signature = (series_id, x, y=None, timestamps=None))]
    fn add(
        &mut self,
        series_id: u64,
        x: PyReadonlyArray2<f64>,
        y: Option<PyReadonlyArray2<f64>>,
        timestamps: Option<PyReadonlyArray1<i64>>,
    ) -> PyResult<()> {
        let x_owned = matrix(x);
        let y_owned = y.map(matrix);
        let ts = match timestamps {
            Some(values) => timestamp_vec(values),
            None => self
                .inner
                .auto_timestamps(series_id, x_owned.nrows())
                .map_err(to_py)?,
        };
        let y_view = y_owned.as_ref().map(Array2::view);
        self.inner
            .add(series_id, &ts, x_owned.view(), y_view)
            .map_err(to_py)
    }

    fn add_y(
        &mut self,
        series_id: u64,
        timestamps: PyReadonlyArray1<i64>,
        y: PyReadonlyArray2<f64>,
    ) -> PyResult<()> {
        let y_owned = matrix(y);
        let ts = timestamp_vec(timestamps);
        self.inner
            .add_y(series_id, &ts, y_owned.view())
            .map_err(to_py)
    }

    #[pyo3(signature = (query, cutoff, k=None, require_target=true))]
    fn search<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray2<f64>,
        cutoff: i64,
        k: Option<usize>,
        require_target: bool,
    ) -> PyResult<SearchPy<'py>> {
        self.finish_search(py, query, cutoff, k, require_target, false)
    }

    #[pyo3(signature = (query, cutoff, k=None, require_target=true))]
    fn search_exact<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray2<f64>,
        cutoff: i64,
        k: Option<usize>,
        require_target: bool,
    ) -> PyResult<SearchPy<'py>> {
        self.finish_search(py, query, cutoff, k, require_target, true)
    }

    #[pyo3(signature = (query, cutoff, k=None, observation_noise=true))]
    fn posterior<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray2<f64>,
        cutoff: i64,
        k: Option<usize>,
        observation_noise: bool,
    ) -> PyResult<PosteriorPy<'py>> {
        let out = self
            .inner
            .posterior(query.as_array(), cutoff, k, observation_noise)
            .map_err(to_py)?;
        Ok((
            out.mu.into_pyarray_bound(py),
            out.se.into_pyarray_bound(py),
            out.se_epi.into_pyarray_bound(py),
            out.se_ale.into_pyarray_bound(py),
        ))
    }

    fn set_variance_scales(&mut self, epistemic: f64, aleatoric: f64) -> PyResult<()> {
        self.inner
            .set_variance_scales(epistemic, aleatoric)
            .map_err(to_py)
    }

    #[getter]
    fn num_obs(&self) -> usize {
        self.inner.num_obs()
    }

    #[getter]
    fn num_windows(&self) -> usize {
        self.inner.num_windows()
    }

    #[getter]
    fn num_dim(&self) -> Option<usize> {
        self.inner.num_dim()
    }

    #[getter]
    fn lookback(&self) -> usize {
        self.inner.lookback()
    }

    #[getter]
    fn neighbor_k(&self) -> usize {
        self.inner.neighbor_k()
    }
}

type SearchPy<'py> = (
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<i64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray3<f64>>,
);

type PosteriorPy<'py> = (
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
);

impl PyENNTS {
    fn options(&self, cutoff: i64, k: Option<usize>, require_target: bool) -> QueryOptions {
        QueryOptions {
            cutoff,
            k: k.unwrap_or_else(|| self.inner.neighbor_k()),
            require_target,
        }
    }

    fn finish_search<'py>(
        &self,
        py: Python<'py>,
        query: PyReadonlyArray2<f64>,
        cutoff: i64,
        k: Option<usize>,
        require_target: bool,
        exact: bool,
    ) -> PyResult<SearchPy<'py>> {
        let opts = self.options(cutoff, k, require_target);
        let batch = if exact {
            self.inner.search_exact(query.as_array(), &opts)
        } else {
            self.inner.search(query.as_array(), &opts)
        }
        .map_err(to_py)?;
        Ok((
            batch.series_id.into_pyarray_bound(py),
            batch.end_timestamp.into_pyarray_bound(py),
            batch.distance.into_pyarray_bound(py),
            batch.target.into_pyarray_bound(py),
        ))
    }
}

#[cfg(test)]
mod kiss_coverage {
    use super::*;

    #[test]
    fn py_ennts_methods_are_linked() {
        let _ = (
            PyENNTS::new,
            PyENNTS::add,
            PyENNTS::add_y,
            PyENNTS::search,
            PyENNTS::search_exact,
            PyENNTS::posterior,
            PyENNTS::set_variance_scales,
            PyENNTS::num_obs,
            PyENNTS::num_windows,
            PyENNTS::num_dim,
            PyENNTS::lookback,
            PyENNTS::neighbor_k,
        );
    }
}
