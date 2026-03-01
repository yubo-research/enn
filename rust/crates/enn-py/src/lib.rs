//! Python bindings for ENN core algorithms using PyO3.

#![allow(clippy::useless_conversion)]

use ndarray::{Array1, IxDyn};
use numpy::{
    IntoPyArray, PyArray1, PyArrayDyn, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArrayDyn,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::wrap_pymodule;

use enn_core::traits::PosteriorComputation;

/// Hypervolume calculation module
#[pymodule]
fn hypervolume(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hypervolume_2d_max_py, m)?)?;
    Ok(())
}

/// Python wrapper for hypervolume_2d_max
#[pyfunction(name = "hypervolume_2d_max")]
fn hypervolume_2d_max_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray2<f64>,
    ref_point: PyReadonlyArray1<f64>,
) -> PyResult<f64> {
    let y_arr = y.as_array();
    let ref_arr = ref_point.as_array();

    // Release GIL for computation
    let result = py.allow_threads(|| enn_core::hypervolume_2d_max(&y_arr, &ref_arr));

    result.map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Hash-based RNG module
#[pymodule]
fn hash(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(normal_hash_batch_multi_seed_fast_py, m)?)?;
    Ok(())
}

/// Python wrapper for normal_hash_batch_multi_seed_fast
#[pyfunction(name = "normal_hash_batch_multi_seed_fast")]
fn normal_hash_batch_multi_seed_fast_py<'py>(
    py: Python<'py>,
    function_seeds: PyReadonlyArray1<i64>,
    data_indices: PyReadonlyArrayDyn<i64>,
    num_metrics: i64,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let seeds_arr = function_seeds.as_array();
    let indices_arr = data_indices.as_array();

    let seeds: Vec<i64> = seeds_arr.iter().copied().collect();
    let indices: Vec<i64> = indices_arr.iter().copied().collect();
    let input_shape = indices_arr.shape().to_vec();
    let mut output_shape = Vec::with_capacity(2 + input_shape.len());
    output_shape.push(seeds.len());
    output_shape.extend(input_shape.iter().copied());
    output_shape.push(num_metrics.max(0) as usize);

    // Release GIL for computation
    let result = py.allow_threads(|| {
        enn_core::normal_hash_batch_multi_seed_fast(&seeds, &indices, num_metrics)
    });

    match result {
        Ok(arr) => {
            // Reshape from (num_seeds, flattened_indices, num_metrics)
            // to (num_seeds, *data_indices.shape, num_metrics) for API parity.
            let reshaped = arr
                .into_shape_with_order(IxDyn(&output_shape))
                .map_err(|e| PyValueError::new_err(format!("Shape error: {}", e)))?;
            Ok(reshaped.into_pyarray_bound(py))
        }
        Err(e) => Err(PyValueError::new_err(e.to_string())),
    }
}

/// Utility functions module
#[pymodule]
fn util(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(standardize_y_py, m)?)?;
    m.add_function(wrap_pyfunction!(pareto_front_2d_maximize_py, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_sobol_indices_py, m)?)?;
    Ok(())
}

/// Python wrapper for standardize_y
#[pyfunction(name = "standardize_y")]
fn standardize_y_py<'py>(
    py: Python<'py>,
    y: PyReadonlyArray1<f64>,
) -> PyResult<(f64, f64)> {
    let y_arr = y.as_array();

    // Release GIL for computation
    let (center, scale) = py.allow_threads(|| enn_core::standardize_y(&y_arr));

    Ok((center, scale))
}

/// Python wrapper for pareto_front_2d_maximize
#[pyfunction(name = "pareto_front_2d_maximize")]
fn pareto_front_2d_maximize_py<'py>(
    py: Python<'py>,
    a: PyReadonlyArray1<f64>,
    b: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<usize>>> {
    let a_arr = a.as_array();
    let b_arr = b.as_array();

    // Release GIL for computation
    let result = py.allow_threads(|| enn_core::pareto_front_2d_maximize(&a_arr, &b_arr, None));

    let front = ndarray::Array1::from_vec(result);
    Ok(front.into_pyarray_bound(py))
}

/// Python wrapper for calculate_sobol_indices
#[pyfunction(name = "calculate_sobol_indices")]
fn calculate_sobol_indices_py<'py>(
    py: Python<'py>,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray1<f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let x_arr = x.as_array();
    let y_arr = y.as_array();

    let sobol = py.allow_threads(|| enn_core::calculate_sobol_indices(&x_arr, &y_arr));
    Ok(Array1::from_vec(sobol.to_vec()).into_pyarray_bound(py))
}

#[pyclass(name = "EpistemicNearestNeighbors")]
struct PyEpistemicNearestNeighbors {
    inner: enn_core::EpistemicNearestNeighbors,
}

type PosteriorPyOut<'py> = (
    Bound<'py, PyArrayDyn<f64>>,
    Bound<'py, PyArrayDyn<f64>>,
    Option<Vec<Vec<usize>>>,
);

#[pymethods]
impl PyEpistemicNearestNeighbors {
    #[new]
    #[pyo3(signature = (train_x, train_y, train_yvar=None, scale_x=false, index_driver="Exact"))]
    fn new(
        train_x: PyReadonlyArray2<f64>,
        train_y: PyReadonlyArray2<f64>,
        train_yvar: Option<PyReadonlyArray2<f64>>,
        scale_x: bool,
        index_driver: &str,
    ) -> PyResult<Self> {
        let driver = match index_driver {
            "Exact" | "exact" | "FLAT" | "flat" => enn_core::IndexDriver::Exact,
            "KDTree" | "kdtree" | "HNSW" | "hnsw" => enn_core::IndexDriver::KDTree,
            _ => {
                return Err(PyValueError::new_err(format!(
                    "Unknown index_driver: {index_driver}"
                )))
            }
        };
        let model = enn_core::EpistemicNearestNeighbors::new(
            train_x.as_array().to_owned(),
            train_y.as_array().to_owned(),
            train_yvar.map(|v| v.as_array().to_owned()),
            scale_x,
            driver,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner: model })
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
            .add(
                &x.as_array(),
                &y.as_array(),
                yvar_arr.as_ref(),
            )
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
        let params = enn_core::ENNParams::new(
            k_num_neighbors,
            epistemic_variance_scale,
            aleatoric_variance_scale,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let flags = enn_core::PosteriorFlags::new()
            .with_exclude_nearest(exclude_nearest)
            .with_observation_noise(observation_noise);
        let out = self
            .inner
            .posterior(&x.as_array(), &params, &flags)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((
            out.mu.into_pyarray_bound(py),
            out.se.into_pyarray_bound(py),
            out.idx,
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
    ) -> PyResult<(Bound<'py, PyArrayDyn<f64>>, Bound<'py, PyArrayDyn<f64>>)> {
        // Build params list
        let n_params = k_values.len();
        if epistemic_scales.len() != n_params || aleatoric_scales.len() != n_params {
            return Err(PyValueError::new_err(
                "k_values, epistemic_scales, and aleatoric_scales must have same length"
            ));
        }

        let mut paramss = Vec::with_capacity(n_params);
        for i in 0..n_params {
            let params = enn_core::ENNParams::new(
                k_values[i],
                epistemic_scales[i],
                aleatoric_scales[i],
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
            paramss.push(params);
        }

        let flags = enn_core::PosteriorFlags::new()
            .with_exclude_nearest(exclude_nearest)
            .with_observation_noise(observation_noise);

        let out = self
            .inner
            .batch_posterior(&x.as_array(), &paramss, &flags)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((
            out.mu.into_pyarray_bound(py),
            out.se.into_pyarray_bound(py),
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
        let params = enn_core::ENNParams::new(
            k_num_neighbors,
            epistemic_variance_scale,
            aleatoric_variance_scale,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let flags = enn_core::PosteriorFlags::new()
            .with_exclude_nearest(exclude_nearest)
            .with_observation_noise(observation_noise);
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
        let params = enn_core::ENNParams::new(
            k_num_neighbors,
            epistemic_variance_scale,
            aleatoric_variance_scale,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let flags = enn_core::PosteriorFlags::new()
            .with_exclude_nearest(exclude_nearest)
            .with_observation_noise(observation_noise);
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
            out.idx,
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
        let params = enn_core::ENNParams::new(
            k_num_neighbors,
            epistemic_variance_scale,
            aleatoric_variance_scale,
        )
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let flags = enn_core::PosteriorFlags::new()
            .with_exclude_nearest(exclude_nearest)
            .with_observation_noise(observation_noise);
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

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    #[getter]
    fn num_outputs(&self) -> usize {
        self.inner.num_outputs()
    }
}

/// ENN model module
#[pymodule]
fn model(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEpistemicNearestNeighbors>()?;
    Ok(())
}

/// Main module
#[pymodule]
fn enn_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(hypervolume))?;
    m.add_wrapped(wrap_pymodule!(hash))?;
    m.add_wrapped(wrap_pymodule!(util))?;
    m.add_wrapped(wrap_pymodule!(model))?;
    Ok(())
}
