//! `ENNNormal.sample` and `confidence_interval` with the y-bounds warp.

use ndarray::{Array2, ArrayD};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::y_bounds::{d_inv_dz, inv_y, is_identity_bounds, warp_y};
use crate::error::ENNError;

fn erf_approx(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs());
    let poly = t * (0.254_829_592
        + t * (-0.284_496_736 + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let y = 1.0 - poly * (-x * x).exp();
    if x < 0.0 { -y } else { y }
}

pub fn z_crit(level: f64) -> Result<f64, ENNError> {
    if !(0.0 < level && level < 1.0) {
        return Err(ENNError::InvalidParameter(format!("level must be in (0, 1), got {level}")));
    }
    let target = 0.5 * (1.0 + level);
    let cdf = |z: f64| 0.5 * (1.0 + erf_approx(z / std::f64::consts::SQRT_2));
    let mut lo = 0.0;
    let mut hi = 8.0;
    while cdf(hi) < target {
        hi *= 2.0;
    }
    for _ in 0..80 {
        let mid = 0.5 * (lo + hi);
        if cdf(mid) < target { lo = mid; } else { hi = mid; }
    }
    Ok(0.5 * (lo + hi))
}

fn normals(shape_len: usize, seed: u64, clip: Option<f64>) -> Vec<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..shape_len)
        .map(|_| {
            let u1 = rng.gen::<f64>().clamp(1e-12, 1.0 - 1e-12);
            let u2 = rng.gen::<f64>();
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            match clip {
                Some(c) => z.clamp(-c, c),
                None => z,
            }
        })
        .collect()
}

pub fn sample_normal(mu: &ArrayD<f64>, se: &ArrayD<f64>, y_bounds: Option<&Array2<f64>>, num_samples: usize, seed: u64, clip: Option<f64>) -> Result<ArrayD<f64>, ENNError> {
    let eps = normals(se.len() * num_samples, seed, clip);
    let mut out_shape = se.shape().to_vec();
    out_shape.push(num_samples);
    if y_bounds.map(is_identity_bounds).unwrap_or(true) {
        let mut out = vec![0.0; eps.len()];
        for (i, (&m, &s)) in mu.iter().zip(se.iter()).enumerate() {
            for s_i in 0..num_samples {
                out[i * num_samples + s_i] = m + s * eps[i * num_samples + s_i];
            }
        }
        return Ok(ArrayD::from_shape_vec(out_shape, out).expect("shape"));
    }
    let bounds = y_bounds.expect("bounds");
    let m = bounds.nrows();
    if mu.shape().last().copied() != Some(m) {
        return Err(ENNError::InvalidParameter("mu last axis does not match y_bounds".into()));
    }
    let rows = mu.len() / m;
    let mu2 = mu.to_shape((rows, m)).map_err(|e| ENNError::InvalidParameter(e.to_string()))?.to_owned();
    let z_mu = warp_y(mu2.view(), bounds)?;
    let mut out = vec![0.0; rows * m * num_samples];
    for i in 0..rows {
        for j in 0..m {
            let a = bounds[[j, 0]];
            let b = bounds[[j, 1]];
            let z = z_mu[[i, j]];
            let jac = d_inv_dz(z, a, b).abs();
            let se_ij = se.iter().nth(i * m + j).copied().unwrap_or(0.0);
            let scale = if jac > 0.0 { se_ij / jac } else { 0.0 };
            for s_i in 0..num_samples {
                let z_s = z + scale * eps[(i * m + j) * num_samples + s_i];
                let z_arr = ndarray::Array2::from_elem((1, m), 0.0);
                let mut one = z_arr;
                one[[0, j]] = z_s;
                let y = inv_y(one.view(), bounds);
                out[(i * m + j) * num_samples + s_i] = y[[0, j]];
            }
        }
    }
    Ok(ArrayD::from_shape_vec(out_shape, out).expect("shape"))
}

pub fn confidence_interval(mu: &ArrayD<f64>, se: &ArrayD<f64>, y_bounds: Option<&Array2<f64>>, level: f64) -> Result<(ArrayD<f64>, ArrayD<f64>), ENNError> {
    let z = z_crit(level)?;
    if y_bounds.map(is_identity_bounds).unwrap_or(true) {
        let lo: Vec<f64> = mu.iter().zip(se.iter()).map(|(m, s)| m - z * s).collect();
        let hi: Vec<f64> = mu.iter().zip(se.iter()).map(|(m, s)| m + z * s).collect();
        return Ok((
            ArrayD::from_shape_vec(mu.shape(), lo).expect("shape"),
            ArrayD::from_shape_vec(mu.shape(), hi).expect("shape"),
        ));
    }
    let bounds = y_bounds.expect("bounds");
    let m = bounds.nrows();
    let rows = mu.len() / m;
    let mu2 = mu.to_shape((rows, m)).map_err(|e| ENNError::InvalidParameter(e.to_string()))?.to_owned();
    let z_mu = warp_y(mu2.view(), bounds)?;
    let mut lo = vec![0.0; mu.len()];
    let mut hi = vec![0.0; mu.len()];
    for i in 0..rows {
        for j in 0..m {
            let a = bounds[[j, 0]];
            let b = bounds[[j, 1]];
            let zz = z_mu[[i, j]];
            let jac = d_inv_dz(zz, a, b).abs();
            let se_ij = se.iter().nth(i * m + j).copied().unwrap_or(0.0);
            let scale = if jac > 0.0 { se_ij / jac } else { 0.0 };
            let mut low = ndarray::Array2::zeros((1, m));
            let mut high = ndarray::Array2::zeros((1, m));
            low[[0, j]] = zz - z * scale;
            high[[0, j]] = zz + z * scale;
            lo[i * m + j] = inv_y(low.view(), bounds)[[0, j]];
            hi[i * m + j] = inv_y(high.view(), bounds)[[0, j]];
        }
    }
    Ok((
        ArrayD::from_shape_vec(mu.shape(), lo).expect("shape"),
        ArrayD::from_shape_vec(mu.shape(), hi).expect("shape"),
    ))
}
