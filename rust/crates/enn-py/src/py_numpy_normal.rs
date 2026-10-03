//! NumPy `Generator.standard_normal` from a copied PCG64 state.

use numpy::{PyArray1, ToPyArray};
use pyo3::prelude::*;

use ennbo::numpy_pcg::NumpyPcg64;

#[pyclass(name = "NumpyNormal")]
pub struct PyNumpyNormal {
    rng: NumpyPcg64,
}

#[pymethods]
impl PyNumpyNormal {
    #[new]
    #[doc = "kiss-coverage-off"]
    fn new(state_hi: u64, state_lo: u64, inc_hi: u64, inc_lo: u64) -> Self {
        let state = (u128::from(state_hi) << 64) | u128::from(state_lo);
        let inc = (u128::from(inc_hi) << 64) | u128::from(inc_lo);
        Self {
            rng: NumpyPcg64::from_parts(state, inc),
        }
    }

    #[doc = "kiss-coverage-off"]
    fn standard_normals<'py>(
        &mut self,
        py: Python<'py>,
        n: usize,
    ) -> Bound<'py, PyArray1<f64>> {
        let mut out = Vec::with_capacity(n);
        for _ in 0..n {
            out.push(self.rng.standard_normal());
        }
        ndarray::Array1::from(out).to_pyarray_bound(py)
    }
}
