//! Benchmark objectives. Noise, when requested, is drawn in Rust from a seed.

use ndarray::{Array1, Array2, ArrayView2};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::error::ENNError;

const ACKLEY_BOUND: f64 = 32.768;

fn standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    let u1 = rng.gen::<f64>().clamp(1e-12, 1.0 - 1e-12);
    let u2 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Ackley objective. The Python wrapper holds this value and nothing else.
pub struct Ackley {
    noise: f64,
    rng: StdRng,
}

impl Ackley {
    /// `seed` initializes the noise generator once.
    pub fn new(noise: f64, seed: u64) -> Self {
        Self {
            noise,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn noise(&self) -> f64 {
        self.noise
    }

    pub fn bounds(&self) -> [f64; 2] {
        [-ACKLEY_BOUND, ACKLEY_BOUND]
    }

    /// One row per input. 1-D input is `(1, d)` after the caller reshapes.
    pub fn evaluate(&mut self, x: ArrayView2<f64>) -> Array1<f64> {
        let mut base = ackley_core(x, 20.0, 0.2, 2.0 * std::f64::consts::PI);
        base.mapv_inplace(|v| -v);
        if self.noise != 0.0 {
            for value in base.iter_mut() {
                *value += self.noise * standard_normal(&mut self.rng);
            }
        }
        base
    }
}

/// Two Ackley objectives on even-dimensional inputs, split in half.
pub struct DoubleAckley {
    noise: f64,
    rng: StdRng,
}

impl DoubleAckley {
    pub fn new(noise: f64, seed: u64) -> Self {
        Self {
            noise,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn noise(&self) -> f64 {
        self.noise
    }

    pub fn bounds(&self) -> [f64; 2] {
        [-ACKLEY_BOUND, ACKLEY_BOUND]
    }

    pub fn evaluate(&mut self, x: ArrayView2<f64>) -> Result<Array2<f64>, ENNError> {
        let d = x.ncols();
        if !d.is_multiple_of(2) {
            return Err(ENNError::InvalidParameter(
                "num_dim must be even for DoubleAckley".into(),
            ));
        }
        let mid = d / 2;
        let left = ackley_core(x.slice(ndarray::s![.., ..mid]), 20.0, 0.2, 2.0 * std::f64::consts::PI);
        let right = ackley_core(x.slice(ndarray::s![.., mid..]), 20.0, 0.2, 2.0 * std::f64::consts::PI);
        let mut out = Array2::zeros((x.nrows(), 2));
        for i in 0..x.nrows() {
            out[[i, 0]] = -left[i];
            out[[i, 1]] = -right[i];
            if self.noise != 0.0 {
                out[[i, 0]] += self.noise * standard_normal(&mut self.rng);
                out[[i, 1]] += self.noise * standard_normal(&mut self.rng);
            }
        }
        Ok(out)
    }
}

fn ackley_row(x: &[f64], a: f64, b: f64, c: f64) -> f64 {
    let mean_sq = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let mean_cos = x.iter().map(|v| (c * v).cos()).sum::<f64>() / x.len() as f64;
    -a * (-b * mean_sq.sqrt()).exp() - mean_cos.exp() + a + std::f64::consts::E
}

pub fn ackley_core(x: ArrayView2<f64>, a: f64, b: f64, c: f64) -> Array1<f64> {
    let mut out = Array1::zeros(x.nrows());
    for i in 0..x.nrows() {
        let shifted: Vec<f64> = x.row(i).iter().map(|v| v - 1.0).collect();
        out[i] = ackley_row(&shifted, a, b, c);
    }
    out
}

pub fn separable_unimodal(x: ArrayView2<f64>) -> Array2<f64> {
    let mut out = Array2::zeros((x.nrows(), 2));
    for i in 0..x.nrows() {
        let x1 = x[[i, 0]];
        let x2 = x[[i, 1]];
        out[[i, 0]] = 500_000.0 - 8.0 * (x1 - 120.0).powi(2);
        out[[i, 1]] = 12.5 - 110.0 * (x2 - 0.91).powi(2);
    }
    out
}
