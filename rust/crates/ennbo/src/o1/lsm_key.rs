use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;

pub(crate) const MAX_AXES: usize = 16;

#[derive(Debug, Clone)]
pub(crate) struct Proj {
    pub m: usize,
    pub d: usize,
    dirs: Vec<f64>,
}

pub(crate) fn gauss(rng: &mut ChaCha8Rng) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-300);
    let u2: f64 = rng.gen();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

impl Proj {
    pub(crate) fn new(d: usize, seed: u64) -> Self {
        let m = d.clamp(1, MAX_AXES);
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut dirs: Vec<f64> = Vec::with_capacity(m * d);
        for _ in 0..m {
            let mut v: Vec<f64> = (0..d).map(|_| gauss(&mut rng)).collect();
            for _ in 0..2 {
                for p in dirs.chunks_exact(d) {
                    let c = dot(&v, p);
                    v.iter_mut().zip(p).for_each(|(a, b)| *a -= c * b);
                }
            }
            let n = dot(&v, &v).sqrt().max(1e-300);
            v.iter_mut().for_each(|a| *a /= n);
            dirs.extend(v);
        }
        Self { m, d, dirs }
    }

    pub(crate) fn axes(d: usize) -> Self {
        let dirs = (0..d * d).map(|i| f64::from(u8::from(i / d == i % d))).collect();
        Self { m: d, d, dirs }
    }

    pub(crate) fn project(&self, x: &[f64], scale: &[f64], out: &mut [f64]) {
        for (o, dir) in out.iter_mut().zip(self.dirs.chunks_exact(self.d)) {
            *o = dir
                .iter()
                .zip(x)
                .zip(scale)
                .map(|((a, b), s)| a * b / s)
                .sum();
        }
    }

    pub(crate) fn key(&self, x: &[f64], frame: &Frame, z: &mut [f64], key: &mut [u64]) {
        self.project(x, &frame.scale, z);
        for (i, (k, v)) in key.iter_mut().zip(z.iter()).enumerate() {
            *k = frame.axis_key(i, *v);
        }
    }
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct Frame {
    pub scale: Vec<f64>,
    #[serde(default)]
    pub center: Vec<f64>,
    #[serde(default)]
    pub width: f64,
}

impl Frame {
    pub(crate) fn axis_key(&self, i: usize, v: f64) -> u64 {
        if self.center.is_empty() {
            return okey(v);
        }
        let u = ((v - self.center[i]) / self.width).clamp(-0.5, 0.5);
        let mag = (u.abs() * 18_446_744_073_709_551_616.0).min(9.223_372_036_854_775e18) as u64;
        if u >= 0.0 {
            (1u64 << 63).saturating_add(mag)
        } else {
            (1u64 << 63) - mag.min(1 << 63)
        }
    }

    pub(crate) fn fit(&mut self, zs: &[f64], m: usize) {
        let n = (zs.len() / m).max(1) as f64;
        let mut lo = vec![f64::INFINITY; m];
        let mut hi = vec![f64::NEG_INFINITY; m];
        let mut mean = vec![0.0; m];
        for z in zs.chunks_exact(m) {
            for i in 0..m {
                lo[i] = lo[i].min(z[i]);
                hi[i] = hi[i].max(z[i]);
                mean[i] += z[i] / n;
            }
        }
        let span = (0..m).map(|i| hi[i] - lo[i]).fold(0.0, f64::max);
        self.center = mean;
        self.width = (4.0 * span).max(1e-300);
    }
}

pub(crate) fn okey(v: f64) -> u64 {
    let b = v.to_bits();
    if b >> 63 == 1 {
        !b
    } else {
        b | (1 << 63)
    }
}

fn less_msb(x: u64, y: u64) -> bool {
    x < y && x < (x ^ y)
}

pub(crate) fn z_cmp(a: &[u64], b: &[u64]) -> Ordering {
    let mut best = 0;
    let mut top = 0u64;
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        let diff = x ^ y;
        if less_msb(top, diff) {
            best = i;
            top = diff;
        }
    }
    a[best].cmp(&b[best])
}

pub(crate) fn box_dist2(z: &[f64], bx: &[f64]) -> f64 {
    let m = z.len();
    let mut acc = 0.0;
    for (i, &v) in z.iter().enumerate() {
        let (lo, hi) = (bx[i], bx[m + i]);
        let g = if v < lo {
            lo - v
        } else if v > hi {
            v - hi
        } else {
            0.0
        };
        acc += g * g;
    }
    acc
}

pub(crate) fn wbox_dist2(q: &[f64], bx: &[f64], w: &[f64]) -> f64 {
    let d = q.len();
    let mut acc = 0.0;
    for (i, (&v, &wi)) in q.iter().zip(w).enumerate() {
        let g = (bx[i] - v).max(v - bx[d + i]).max(0.0);
        acc += wi * g * g;
    }
    acc
}

pub(crate) fn empty_box(m: usize) -> Vec<f64> {
    let mut b = vec![f64::INFINITY; m];
    b.extend(std::iter::repeat_n(f64::NEG_INFINITY, m));
    b
}

pub(crate) fn grow_box(bx: &mut [f64], z: &[f64]) {
    let m = z.len();
    for (i, &v) in z.iter().enumerate() {
        bx[i] = bx[i].min(v);
        bx[m + i] = bx[m + i].max(v);
    }
}

pub(crate) fn union_box(bx: &mut [f64], other: &[f64]) {
    let m = bx.len() / 2;
    for i in 0..m {
        bx[i] = bx[i].min(other[i]);
        bx[m + i] = bx[m + i].max(other[m + i]);
    }
}

pub(crate) fn metric_factor(scale: &[f64], weights: &[f64]) -> f64 {
    scale
        .iter()
        .zip(weights)
        .map(|(s, w)| s * s * w)
        .fold(f64::INFINITY, f64::min)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn okey_preserves_order() {
        let v = [-3.0, -1e-12, -0.0, 0.0, 1e-300, 0.5, 2.0, 1e10];
        for w in v.windows(2) {
            assert!(okey(w[0]) <= okey(w[1]), "{w:?}");
        }
    }

    #[test]
    fn fixed_frame_keys_preserve_order_and_resolve_tiny_offsets() {
        let mut f = Frame { scale: vec![1.0], ..Frame::default() };
        f.fit(&[0.4, 0.6], 1);
        assert!((f.center[0] - 0.5).abs() < 1e-12 && (f.width - 0.8).abs() < 1e-12);
        let v = [-10.0, 0.1, 0.5 - 1e-12, 0.5, 0.5 + 1e-12, 0.9, 10.0];
        for w in v.windows(2) {
            assert!(f.axis_key(0, w[0]) <= f.axis_key(0, w[1]), "{w:?}");
        }
        assert!(f.axis_key(0, 0.5 - 1e-12) < f.axis_key(0, 0.5));
        assert!(f.axis_key(0, 0.5) < f.axis_key(0, 0.5 + 1e-12));
    }

    #[test]
    fn z_cmp_matches_interleaved_bits() {
        let a = [0b10u64, 0b01];
        let b = [0b01u64, 0b10];
        assert_eq!(z_cmp(&a, &b), Ordering::Greater);
        assert_eq!(z_cmp(&a, &a), Ordering::Equal);
    }

    #[test]
    fn projection_is_orthonormal_and_bound_holds() {
        let p = Proj::new(10, 3);
        let s = vec![1.0; 10];
        let x: Vec<f64> = (0..10).map(|i| i as f64 * 0.1).collect();
        let mut z = vec![0.0; p.m];
        p.project(&x, &s, &mut z);
        let n2: f64 = x.iter().map(|v| v * v).sum();
        let z2: f64 = z.iter().map(|v| v * v).sum();
        assert!((n2 - z2).abs() < 1e-9);
        let mut bx = empty_box(p.m);
        grow_box(&mut bx, &z);
        assert_eq!(box_dist2(&z, &bx), 0.0);
    }
}
