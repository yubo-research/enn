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

/// Parameter fitting module
#[pymodule]
fn fit(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(enn_fit_py, m)?)?;
    m.add_function(wrap_pyfunction!(subsample_loglik_py, m)?)?;
    Ok(())
}

/// Python wrapper for enn_fit
#[allow(clippy::too_many_arguments)]
#[pyfunction(name = "enn_fit")]
#[pyo3(signature = (model, k, num_fit_candidates, num_fit_samples, seed, params_warm_start=None, infer_aleatoric_variance_scale=true))]
fn enn_fit_py(
    model: &PyEpistemicNearestNeighbors,
    k: i32,
    num_fit_candidates: usize,
    num_fit_samples: usize,
    seed: u64,
    params_warm_start: Option<PyENNParams>,
    infer_aleatoric_variance_scale: bool,
) -> PyResult<PyENNParams> {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    let mut rng = StdRng::seed_from_u64(seed);

    let warm_start = params_warm_start.as_ref().map(|p| p.inner);

    let result = enn_core::enn_fit(
        &model.inner,
        k,
        num_fit_candidates,
        num_fit_samples,
        &mut rng,
        warm_start.as_ref(),
        infer_aleatoric_variance_scale,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyENNParams { inner: result })
}

/// Python wrapper for subsample_loglik
#[allow(clippy::too_many_arguments)]
#[pyfunction(name = "subsample_loglik")]
#[pyo3(signature = (model, x, y, k_values, epistemic_scales, aleatoric_scales, p, seed, y_std=None))]
fn subsample_loglik_py(
    model: &PyEpistemicNearestNeighbors,
    x: PyReadonlyArray2<f64>,
    y: PyReadonlyArray2<f64>,
    k_values: Vec<i32>,
    epistemic_scales: Vec<f64>,
    aleatoric_scales: Vec<f64>,
    p: usize,
    seed: u64,
    y_std: Option<PyReadonlyArray1<f64>>,
) -> PyResult<Vec<f64>> {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    let mut rng = StdRng::seed_from_u64(seed);

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

    let y_std_arr = y_std.as_ref().map(|v| v.as_array());

    let result = enn_core::subsample_loglik(
        &model.inner,
        &x.as_array(),
        &y.as_array(),
        &paramss,
        p,
        &mut rng,
        y_std_arr.as_ref(),
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(result)
}

/// Wrapper for ENNParams
#[pyclass(name = "ENNParams")]
#[derive(Clone, Copy)]
struct PyENNParams {
    inner: enn_core::ENNParams,
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
        let inner = enn_core::ENNParams::new(
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

/// ENN model module
#[pymodule]
fn model(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEpistemicNearestNeighbors>()?;
    m.add_class::<PyENNParams>()?;
    Ok(())
}

/// Optimizer module
#[pymodule]
fn optimizer(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyOptimizer>()?;
    m.add_function(wrap_pyfunction!(create_optimizer_enn_py, m)?)?;
    m.add_function(wrap_pyfunction!(create_optimizer_zero_py, m)?)?;
    m.add_function(wrap_pyfunction!(create_optimizer_lhd_py, m)?)?;
    Ok(())
}

/// Python wrapper for Optimizer
#[pyclass(name = "Optimizer")]
struct PyOptimizer {
    inner: enn_core::Optimizer,
}

#[pymethods]
impl PyOptimizer {
    /// Ask for candidate points
    #[pyo3(signature = (num_arms, seed))]
    fn ask<'py>(
        &mut self,
        py: Python<'py>,
        num_arms: usize,
        seed: u64,
    ) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let mut rng = StdRng::seed_from_u64(seed);

        let result = self
            .inner
            .ask(num_arms, &mut rng)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        Ok(result.into_dyn().into_pyarray_bound(py))
    }

    /// Tell observations
    #[pyo3(signature = (x, y, seed))]
    fn tell(
        &mut self,
        x: PyReadonlyArray2<f64>,
        y: PyReadonlyArray2<f64>,
        seed: u64,
    ) -> PyResult<()> {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let mut rng = StdRng::seed_from_u64(seed);

        self.inner
            .tell(&x.as_array(), &y.as_array(), &mut rng)
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Get init progress if in initialization phase
    fn init_progress(&self) -> Option<(usize, usize)> {
        self.inner.init_progress()
    }

    /// Get current telemetry
    fn telemetry(&self) -> PyTelemetry {
        let t = self.inner.telemetry();
        PyTelemetry {
            dt_fit: t.dt_fit,
            dt_gen: t.dt_gen,
            dt_sel: t.dt_sel,
            dt_tell: t.dt_tell,
        }
    }
}

/// Telemetry data structure for Python
#[pyclass(name = "Telemetry")]
#[derive(Clone, Copy)]
struct PyTelemetry {
    #[pyo3(get)]
    dt_fit: f64,
    #[pyo3(get)]
    dt_gen: f64,
    #[pyo3(get)]
    dt_sel: f64,
    #[pyo3(get)]
    dt_tell: f64,
}

/// Create TuRBO-ENN optimizer
#[pyfunction(name = "create_optimizer_enn")]
#[pyo3(signature = (bounds, k=10, num_init=10, seed=42))]
fn create_optimizer_enn_py(
    bounds: PyReadonlyArray2<f64>,
    k: i32,
    num_init: usize,
    seed: u64,
) -> PyResult<PyOptimizer> {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    let mut rng = StdRng::seed_from_u64(seed);

    let optimizer = enn_core::create_optimizer_enn(
        bounds.as_array().to_owned(),
        k,
        num_init,
        &mut rng,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyOptimizer { inner: optimizer })
}

/// Create TuRBO-ZERO optimizer
#[pyfunction(name = "create_optimizer_zero")]
#[pyo3(signature = (bounds, num_init=10, seed=42))]
fn create_optimizer_zero_py(
    bounds: PyReadonlyArray2<f64>,
    num_init: usize,
    seed: u64,
) -> PyResult<PyOptimizer> {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    let mut rng = StdRng::seed_from_u64(seed);

    let optimizer = enn_core::create_optimizer_zero(
        bounds.as_array().to_owned(),
        num_init,
        &mut rng,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyOptimizer { inner: optimizer })
}

/// Create LHD-only optimizer
#[pyfunction(name = "create_optimizer_lhd")]
#[pyo3(signature = (bounds, num_init=10, seed=42))]
fn create_optimizer_lhd_py(
    bounds: PyReadonlyArray2<f64>,
    num_init: usize,
    seed: u64,
) -> PyResult<PyOptimizer> {
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    let mut rng = StdRng::seed_from_u64(seed);

    let optimizer = enn_core::create_optimizer_lhd(
        bounds.as_array().to_owned(),
        num_init,
        &mut rng,
    )
    .map_err(|e| PyValueError::new_err(e.to_string()))?;

    Ok(PyOptimizer { inner: optimizer })
}

/// Main module
#[pymodule]
fn enn_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pymodule!(hypervolume))?;
    m.add_wrapped(wrap_pymodule!(hash))?;
    m.add_wrapped(wrap_pymodule!(util))?;
    m.add_wrapped(wrap_pymodule!(model))?;
    m.add_wrapped(wrap_pymodule!(fit))?;
    m.add_wrapped(wrap_pymodule!(optimizer))?;
    Ok(())
}
