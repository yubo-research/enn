//! Binned first-order Sobol indices used by AUTO metric learning.
//!
//! This is the `metric_stream` estimator, not `calculate_sobol_indices`.

pub const MIN_DEPENDENCE_ROWS: usize = 100;
pub const DEPENDENCE_Z: f64 = 3.0;

fn between_share(count: &[f64], total: &[f64], ss_tot: f64, n: usize, b: usize) -> f64 {
    let mut ss_between = 0.0;
    for (c, t) in count.iter().zip(total.iter()) {
        ss_between += t * t / c.max(1.0);
    }
    let ms_within = (ss_tot - ss_between) / (n - b) as f64;
    (ss_between - (b - 1) as f64 * ms_within).max(0.0) / ss_tot
}

fn stable_ranks(col: &[f64]) -> Vec<usize> {
    let n = col.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| col[i].total_cmp(&col[j]).then(i.cmp(&j)));
    let mut rank = vec![0usize; n];
    for (k, &i) in order.iter().enumerate() {
        rank[i] = k;
    }
    rank
}

/// `sobol_index` for one `y` column. `x` is row-major `n * d`.
pub fn sobol_index(x: &[f64], n: usize, d: usize, y: &[f64], num_bins: Option<usize>) -> Vec<f64> {
    let b = num_bins.unwrap_or_else(|| (n as f64).sqrt().floor().max(2.0) as usize);
    let mean = y.iter().sum::<f64>() / n as f64;
    let yc: Vec<f64> = y.iter().map(|v| v - mean).collect();
    let ss_tot = yc.iter().map(|v| v * v).sum::<f64>();
    if n < 2 * b || ss_tot <= 0.0 {
        return vec![0.0; d];
    }
    let mut out = vec![0.0; d];
    for dim in 0..d {
        let col: Vec<f64> = (0..n).map(|i| x[i * d + dim]).collect();
        let ranks = stable_ranks(&col);
        let mut count = vec![0.0; b];
        let mut total = vec![0.0; b];
        for i in 0..n {
            let cell = ranks[i] * b / n;
            count[cell] += 1.0;
            total[cell] += yc[i];
        }
        out[dim] = between_share(&count, &total, ss_tot, n, b);
    }
    out
}

pub fn null_sd(n: usize, num_cells: Option<usize>) -> f64 {
    if n < 3 {
        return f64::INFINITY;
    }
    let b = num_cells.unwrap_or_else(|| (n as f64).sqrt().floor().max(2.0) as usize);
    (2.0 * (b - 1) as f64).sqrt() / n as f64
}

fn cells_of_group(x: &[f64], n: usize, d: usize, group: &[usize]) -> (Vec<usize>, usize) {
    let g = group.len();
    let mut rows: Vec<(Vec<u64>, usize)> = Vec::with_capacity(n);
    for i in 0..n {
        let bits: Vec<u64> = group.iter().map(|&j| x[i * d + j].to_bits()).collect();
        rows.push((bits, i));
    }
    rows.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut ids = vec![0usize; n];
    let mut b = 0usize;
    let mut prev: Option<&Vec<u64>> = None;
    for (bits, i) in &rows {
        if prev != Some(bits) {
            if prev.is_some() {
                b += 1;
            }
            prev = Some(bits);
        }
        ids[*i] = b;
    }
    let n_cells = if n == 0 { 0 } else { b + 1 };
    let _ = g;
    (ids, n_cells)
}

pub fn group_sobol_index(x: &[f64], n: usize, d: usize, group: &[usize], y: &[f64]) -> f64 {
    let (cells, b) = cells_of_group(x, n, d, group);
    let mean = y.iter().sum::<f64>() / n.max(1) as f64;
    let yc: Vec<f64> = y.iter().map(|v| v - mean).collect();
    let ss_tot = yc.iter().map(|v| v * v).sum::<f64>();
    if b < 2 || n < 2 * b || ss_tot <= 0.0 {
        return 0.0;
    }
    let mut count = vec![0.0; b];
    let mut total = vec![0.0; b];
    for i in 0..n {
        count[cells[i]] += 1.0;
        total[cells[i]] += yc[i];
    }
    between_share(&count, &total, ss_tot, n, b)
}

pub fn group_cell_count(x: &[f64], n: usize, d: usize, group: &[usize]) -> usize {
    cells_of_group(x, n, d, group).1
}
