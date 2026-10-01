//! Benchmark objectives. Noise, when requested, is drawn in Rust from a seed.

use ndarray::{Array1, Array2, ArrayView2};

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
