//! Diagonal dependence weights from Sobol indices.

use crate::metric_sobol::{
    group_cell_count, group_sobol_index, null_sd, sobol_index, DEPENDENCE_Z, MIN_DEPENDENCE_ROWS,
};

const MIN_VAR: f64 = 1e-24;
pub const DEPENDENCE_FLOOR: f64 = 1e-6;

fn column(y: &[f64], n: usize, m: usize, j: usize) -> Vec<f64> {
    (0..n).map(|i| y[i * m + j]).collect()
}

fn mean_sobol(x: &[f64], n: usize, d: usize, y: &[f64], m: usize) -> Vec<f64> {
    let mut acc = vec![0.0; d];
    for j in 0..m {
        let col = column(y, n, m, j);
        let s = sobol_index(x, n, d, &col, None);
        for dim in 0..d {
            acc[dim] += s[dim];
        }
    }
    for v in &mut acc {
        *v /= m as f64;
    }
    acc
}

fn passing(s: f64, n: usize, cells: Option<usize>) -> f64 {
    if s > DEPENDENCE_Z * null_sd(n, cells) { s } else { 0.0 }
}

fn unit_indices(x: &[f64], n: usize, d: usize, y: &[f64], m: usize, tied: &[Vec<usize>]) -> (Vec<f64>, Vec<usize>) {
    let raw = mean_sobol(x, n, d, y, m);
    let s: Vec<f64> = raw.iter().copied().map(|v| passing(v, n, None)).collect();
    if tied.is_empty() {
        return (s, (0..d).collect());
    }
    let mut in_group = vec![false; d];
    for g in tied {
        for &j in g {
            in_group[j] = true;
        }
    }
    let free: Vec<usize> = (0..d).filter(|&j| !in_group[j]).collect();
    let mut unit = vec![0usize; d];
    for (k, &j) in free.iter().enumerate() {
        unit[j] = k;
    }
    let mut joint = Vec::new();
    for (i, g) in tied.iter().enumerate() {
        for &j in g {
            unit[j] = free.len() + i;
        }
        let mut s_g = 0.0;
        for col_j in 0..m {
            let col = column(y, n, m, col_j);
            s_g += group_sobol_index(x, n, d, g, &col);
        }
        s_g /= m as f64;
        let cells = group_cell_count(x, n, d, g);
        joint.push(passing(s_g, n, Some(cells)));
    }
    let mut out = Vec::with_capacity(free.len() + joint.len());
    for &j in &free {
        out.push(s[j]);
    }
    out.extend(joint);
    (out, unit)
}

fn spread(x: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut out = vec![1.0; d];
    if n == 0 {
        return out;
    }
    for j in 0..d {
        let mut sum = 0.0;
        let mut sumsq = 0.0;
        for i in 0..n {
            let v = x[i * d + j];
            sum += v;
            sumsq += v * v;
        }
        let mean = sum / n as f64;
        let var = sumsq / n as f64 - mean * mean;
        out[j] = if var > MIN_VAR { var } else { 1.0 };
    }
    out
}

pub fn dependence_weights(
    x: &[f64],
    n: usize,
    d: usize,
    y: &[f64],
    m: usize,
    tied: &[Vec<usize>],
    floor: f64,
) -> Vec<f64> {
    let (mut s, unit) = unit_indices(x, n, d, y, m, tied);
    let s_max = s.iter().copied().fold(0.0_f64, f64::max);
    if s_max > 0.0 {
        for v in &mut s {
            *v = v.max(floor * s_max);
        }
    } else {
        s.fill(1.0);
    }
    let mut spr = spread(x, n, d);
    for g in tied {
        for &j in g {
            spr[j] = 1.0;
        }
    }
    let s_sum = s.iter().sum::<f64>();
    let u = s.len() as f64;
    (0..d).map(|j| (u * s[unit[j]] / s_sum) / spr[j]).collect()
}

pub fn too_few_rows(n: usize) -> bool {
    n < MIN_DEPENDENCE_ROWS
}
