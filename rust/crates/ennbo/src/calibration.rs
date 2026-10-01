//! Post-hoc affine calibration of an ENN posterior.

use ndarray::{Array1, Array2, ArrayView2};

use crate::error::ENNError;

const BOUND_EPS: f64 = 1e-3;
const BOUND_SE_MULT: f64 = 0.25;

#[derive(Clone, Debug)]
pub struct AffineCalibrator {
    pub a: Array1<f64>,
    pub b: Array1<f64>,
    pub c: Array1<f64>,
}

impl AffineCalibrator {
    pub fn identity(num_metrics: usize) -> Self {
        Self {
            a: Array1::zeros(num_metrics),
            b: Array1::ones(num_metrics),
            c: Array1::ones(num_metrics),
        }
    }

    pub fn fit(mu: ArrayView2<f64>, y: ArrayView2<f64>, se: Option<ArrayView2<f64>>) -> Result<Self, ENNError> {
        if mu.shape() != y.shape() {
            return Err(ENNError::InvalidShape { expected: y.shape().to_vec(), got: mu.shape().to_vec() });
        }
        let (n, m) = (mu.nrows(), mu.ncols());
        let mut a = Array1::zeros(m);
        let mut b = Array1::ones(m);
        if n >= 2 {
            for j in 0..m {
                let (s_mu, s_y, s_mumu, s_muy) = col_sums(&mu, &y, j);
                let det = n as f64 * s_mumu - s_mu * s_mu;
                let scale = (n as f64 * s_mumu).abs().max(1.0);
                if det.is_finite() && det.abs() >= 1e-18 * scale {
                    let aj = (s_mumu * s_y - s_mu * s_muy) / det;
                    let bj = (n as f64 * s_muy - s_mu * s_y) / det;
                    if aj.is_finite() && bj.is_finite() {
                        a[j] = aj;
                        b[j] = bj;
                    }
                }
            }
        }
        let c = match se {
            Some(se) if se.shape() == mu.shape() => residual_c(&mu, &se, &y, &a, &b),
            Some(se) => {
                return Err(ENNError::InvalidShape { expected: mu.shape().to_vec(), got: se.shape().to_vec() });
            }
            None => Array1::ones(m),
        };
        Ok(Self { a, b, c })
    }

    pub fn map_mu(&self, mu: &Array2<f64>, y_bounds: Option<&Array2<f64>>, se: Option<&Array2<f64>>) -> Array2<f64> {
        let mut out = Array2::zeros(mu.raw_dim());
        for i in 0..mu.nrows() {
            for j in 0..mu.ncols() {
                out[[i, j]] = self.a[j] + self.b[j] * mu[[i, j]];
            }
        }
        project_mu(&mut out, y_bounds, se);
        out
    }

    pub fn apply_se(&self, se_epi: &Array2<f64>, se_ale: &Array2<f64>) -> (Array2<f64>, Array2<f64>, Array2<f64>) {
        let mut epi = se_epi.clone();
        let mut ale = se_ale.clone();
        for i in 0..epi.nrows() {
            for j in 0..epi.ncols() {
                epi[[i, j]] *= self.c[j];
                ale[[i, j]] *= self.c[j];
            }
        }
        let mut se = Array2::zeros(epi.raw_dim());
        for i in 0..se.nrows() {
            for j in 0..se.ncols() {
                se[[i, j]] = (epi[[i, j]] * epi[[i, j]] + ale[[i, j]] * ale[[i, j]]).sqrt();
            }
        }
        (se, epi, ale)
    }
}

fn col_sums(mu: &ArrayView2<f64>, y: &ArrayView2<f64>, j: usize) -> (f64, f64, f64, f64) {
    let mut s_mu = 0.0;
    let mut s_y = 0.0;
    let mut s_mumu = 0.0;
    let mut s_muy = 0.0;
    for i in 0..mu.nrows() {
        let mj = mu[[i, j]];
        let yj = y[[i, j]];
        s_mu += mj;
        s_y += yj;
        s_mumu += mj * mj;
        s_muy += mj * yj;
    }
    (s_mu, s_y, s_mumu, s_muy)
}

fn residual_c(mu: &ArrayView2<f64>, se: &ArrayView2<f64>, y: &ArrayView2<f64>, a: &Array1<f64>, b: &Array1<f64>) -> Array1<f64> {
    let m = mu.ncols();
    let mut c = Array1::ones(m);
    for j in 0..m {
        let mut ss_r = 0.0;
        let mut ss_s = 0.0;
        let n = mu.nrows() as f64;
        for i in 0..mu.nrows() {
            let pred = a[j] + b[j] * mu[[i, j]];
            let resid = y[[i, j]] - pred;
            ss_r += resid * resid;
            ss_s += se[[i, j]] * se[[i, j]];
        }
        let rms_r = (ss_r / n).sqrt();
        let rms_s = (ss_s / n).sqrt();
        c[j] = if rms_s.is_finite() && rms_s > 0.0 && rms_r.is_finite() { rms_r / rms_s } else { 1.0 };
    }
    c
}

fn margin(se: Option<f64>, span: f64) -> f64 {
    let floor = if span.is_finite() { BOUND_EPS.max(1e-6 * span) } else { BOUND_EPS };
    match se {
        Some(s) => floor.max(BOUND_SE_MULT * s.abs()),
        None => floor,
    }
}

pub fn project_mu(out: &mut Array2<f64>, y_bounds: Option<&Array2<f64>>, se: Option<&Array2<f64>>) {
    let Some(bounds) = y_bounds else { return };
    if bounds.is_empty() {
        return;
    }
    let open = (0..bounds.nrows()).all(|j| bounds[[j, 0]] == f64::NEG_INFINITY && bounds[[j, 1]] == f64::INFINITY);
    if open {
        return;
    }
    for j in 0..bounds.nrows().min(out.ncols()) {
        let lo = bounds[[j, 0]];
        let hi = bounds[[j, 1]];
        for i in 0..out.nrows() {
            let se_ij = se.map(|s| s[[i, j]]);
            let col = out[[i, j]];
            out[[i, j]] = project_scalar(col, lo, hi, se_ij);
        }
    }
}

pub fn map_mu_with(
    a: &[f64],
    b: &[f64],
    mu: ArrayView2<f64>,
    y_bounds: Option<&Array2<f64>>,
    se: Option<&Array2<f64>>,
) -> Result<Array2<f64>, ENNError> {
    check_cols(a.len(), mu.ncols())?;
    check_cols(b.len(), mu.ncols())?;
    let mut out = Array2::zeros(mu.raw_dim());
    for i in 0..mu.nrows() {
        for j in 0..mu.ncols() {
            out[[i, j]] = a[j] + b[j] * mu[[i, j]];
        }
    }
    project_mu(&mut out, y_bounds, se);
    Ok(out)
}

pub fn map_draws_with(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    draws: ArrayView2<f64>,
    mu: ArrayView2<f64>,
    y_bounds: Option<&Array2<f64>>,
    se: Option<&Array2<f64>>,
) -> Result<Array2<f64>, ENNError> {
    check_cols(a.len(), draws.ncols())?;
    check_cols(b.len(), draws.ncols())?;
    check_cols(c.len(), draws.ncols())?;
    if mu.shape() != draws.shape() {
        return Err(ENNError::InvalidShape { expected: draws.shape().to_vec(), got: mu.shape().to_vec() });
    }
    let se_scaled = se.map(|s| {
        let mut sp = s.to_owned();
        for i in 0..sp.nrows() {
            for j in 0..sp.ncols().min(c.len()) {
                sp[[i, j]] *= c[j];
            }
        }
        sp
    });
    let mut out = Array2::zeros(draws.raw_dim());
    for i in 0..draws.nrows() {
        for j in 0..draws.ncols() {
            out[[i, j]] = a[j] + b[j] * mu[[i, j]] + c[j] * (draws[[i, j]] - mu[[i, j]]);
        }
    }
    project_mu(&mut out, y_bounds, se_scaled.as_ref());
    Ok(out)
}

#[allow(clippy::type_complexity)]
pub fn apply_normal(
    a: &[f64],
    b: &[f64],
    c: &[f64],
    mu: ArrayView2<f64>,
    se_epi: ArrayView2<f64>,
    se_ale: ArrayView2<f64>,
    y_bounds: Option<&Array2<f64>>,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>), ENNError> {
    check_cols(c.len(), mu.ncols())?;
    if se_epi.shape() != mu.shape() || se_ale.shape() != mu.shape() {
        return Err(ENNError::InvalidShape { expected: mu.shape().to_vec(), got: se_epi.shape().to_vec() });
    }
    let mut epi = se_epi.to_owned();
    let mut ale = se_ale.to_owned();
    for i in 0..epi.nrows() {
        for j in 0..epi.ncols() {
            epi[[i, j]] *= c[j];
            ale[[i, j]] *= c[j];
        }
    }
    let mut se = Array2::zeros(mu.raw_dim());
    for i in 0..se.nrows() {
        for j in 0..se.ncols() {
            se[[i, j]] = (epi[[i, j]] * epi[[i, j]] + ale[[i, j]] * ale[[i, j]]).sqrt();
        }
    }
    let mu_p = map_mu_with(a, b, mu, y_bounds, Some(&se))?;
    Ok((mu_p, se, epi, ale))
}

fn check_cols(got: usize, expect: usize) -> Result<(), ENNError> {
    if got == expect {
        Ok(())
    } else {
        Err(ENNError::InvalidShape { expected: vec![expect], got: vec![got] })
    }
}

fn project_scalar(col: f64, lo: f64, hi: f64, se: Option<f64>) -> f64 {
    if lo.is_finite() && hi.is_finite() {
        let eps = margin(se, hi - lo);
        let lo_b = lo + eps;
        let hi_b = hi - eps;
        if lo_b >= hi_b { 0.5 * (lo + hi) } else { col.clamp(lo_b, hi_b) }
    } else if lo.is_finite() {
        col.max(lo + margin(se, f64::INFINITY))
    } else if hi.is_finite() {
        col.min(hi - margin(se, f64::INFINITY))
    } else {
        col
    }
}
