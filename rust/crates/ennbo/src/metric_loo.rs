//! Leave-one-out ENN log-likelihood for a diagonal metric.

const EPS: f64 = 1e-9;
const MIN_VAR: f64 = 1e-24;
const SCALE_GRID: [f64; 13] = [
    0.01, 0.031_622_776_601_683_79, 0.1, 0.316_227_766_016_837_94, 1.0,
    3.162_277_660_168_379_5, 10.0, 31.622_776_601_683_793, 100.0,
    316.227_766_016_837_96, 1000.0, 3_162.277_660_168_379_5, 10_000.0,
];
const NOISE_GRID: [f64; 7] = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0];

fn variance(col: &[f64]) -> f64 {
    let n = col.len() as f64;
    let mean = col.iter().sum::<f64>() / n;
    let var = col.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n;
    if var > MIN_VAR { var } else { 1.0 }
}

fn column_loglik(z: &[f64], ycol: &[f64], a_scaled: &[f64], n: usize, d: usize, k: usize) -> f64 {
    let mut row_a = vec![0.0; n];
    let mut az = vec![0.0; n * d];
    for r in 0..n {
        let mut s = 0.0;
        for j in 0..d {
            let zj = z[r * d + j];
            let azj = a_scaled[j] * zj;
            az[r * d + j] = azj;
            s += azj * zj;
        }
        row_a[r] = s;
    }
    let mut nbr = vec![0usize; n * k];
    let mut d2n = vec![0.0; n * k];
    for i in 0..n {
        let mut dist = Vec::with_capacity(n - 1);
        for r in 0..n {
            if r == i {
                continue;
            }
            let mut dot = 0.0;
            for j in 0..d {
                dot += az[i * d + j] * z[r * d + j];
            }
            dist.push((row_a[i] + row_a[r] - 2.0 * dot, r));
        }
        dist.sort_by(|p, q| p.0.total_cmp(&q.0).then(p.1.cmp(&q.1)));
        for t in 0..k {
            nbr[i * k + t] = dist[t].1;
            d2n[i * k + t] = dist[t].0;
        }
    }
    let mut best = f64::NEG_INFINITY;
    for &scale in &SCALE_GRID {
        for &noise in &NOISE_GRID {
            let mut sum = 0.0;
            for i in 0..n {
                let mut wsum = 0.0;
                let mut wysum = 0.0;
                for t in 0..k {
                    let w = 1.0 / (EPS + (d2n[i * k + t] * scale).max(0.0) + noise);
                    wsum += w;
                    wysum += w * ycol[nbr[i * k + t]];
                }
                let mu = wysum / wsum;
                let var = 1.0 / wsum + noise;
                let err = ycol[i] - mu;
                sum += -0.5 * (2.0 * std::f64::consts::PI * var).ln() - 0.5 * err * err / var;
            }
            best = best.max(sum / n as f64);
        }
    }
    best
}

pub fn loo_loglik(x: &[f64], n: usize, d: usize, y: &[f64], m: usize, a: &[f64], k: usize) -> f64 {
    let kk = k.min(n - 1);
    let mut mean = vec![0.0; d];
    let mut spr = vec![1.0; d];
    for j in 0..d {
        let col: Vec<f64> = (0..n).map(|i| x[i * d + j]).collect();
        mean[j] = col.iter().sum::<f64>() / n as f64;
        spr[j] = variance(&col);
    }
    let mut z = vec![0.0; n * d];
    for i in 0..n {
        for j in 0..d {
            z[i * d + j] = (x[i * d + j] - mean[j]) / spr[j].sqrt();
        }
    }
    let a_scaled: Vec<f64> = (0..d).map(|j| a[j] * spr[j]).collect();
    let mut total = 0.0;
    for col in 0..m {
        let raw: Vec<f64> = (0..n).map(|i| y[i * m + col]).collect();
        let yvar = variance(&raw);
        let ymean = raw.iter().sum::<f64>() / n as f64;
        let yz: Vec<f64> = raw.iter().map(|v| (v - ymean) / yvar.sqrt()).collect();
        total += column_loglik(&z, &yz, &a_scaled, n, d, kk);
    }
    total / m as f64
}
