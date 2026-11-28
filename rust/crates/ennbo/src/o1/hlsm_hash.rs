use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
pub(crate) use super::hlsm_frame::TableFrame;
use super::lsm_key::gauss;

#[derive(Debug, Clone)]
pub(crate) struct TableHash {
    pub k: usize,
    pub d: usize,
    dirs: Vec<f64>,
}

impl TableHash {
    pub(crate) fn new(d: usize, k: usize, seed: u64) -> Self {
        assert!((1..=64).contains(&k), "hash width k={k} must be in 1..=64");
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut dirs = Vec::with_capacity(k * d);
        assert!(k <= d, "hash width k={k} exceeds the dimension {d}");
        for _ in 0..k {
            let mut v: Vec<f64> = (0..d).map(|_| gauss(&mut rng)).collect();
            for _ in 0..2 {
                for p in dirs.chunks_exact(d) {
                    let c: f64 = v.iter().zip(p).map(|(a, b)| a * b).sum();
                    v.iter_mut().zip(p).for_each(|(a, b)| *a -= c * b);
                }
            }
            let n = v.iter().map(|a| a * a).sum::<f64>().sqrt().max(1e-300);
            dirs.extend(v.iter().map(|a| a / n));
        }
        Self { k, d, dirs }
    }

    pub(crate) fn project(&self, x: &[f64], scale: &[f64], z: &mut [f64]) {
        for (o, dir) in z.iter_mut().zip(self.dirs.chunks_exact(self.d)) {
            *o = dir.iter().zip(x).zip(scale).map(|((a, b), s)| a * b / s).sum();
        }
    }

    pub(crate) fn fit(&self, x: &[f64], scale: &[f64]) -> TableFrame {
        let mut z = vec![0.0; x.len() / self.d * self.k];
        for (r, o) in x.chunks_exact(self.d).zip(z.chunks_exact_mut(self.k)) {
            self.project(r, scale, o);
        }
        TableFrame::from_projections(&z, self.k, scale)
    }

    pub(crate) fn key(&self, x: &[f64], frame: &TableFrame, z: &mut [f64]) -> u64 {
        self.project(x, &frame.scale, z);
        let bits = 64 / self.k;
        let top = ((1u128 << bits) - 1) as f64;
        let mut u = [0u64; 64];
        for i in 0..self.k {
            u[i] = (frame.unit_of(i, z[i]) * top) as u64;
        }
        let mut key = 0u64;
        for b in (0..bits).rev() {
            for &ui in &u[..self.k] {
                key = (key << 1) | ((ui >> b) & 1);
            }
        }
        key << (64 - bits * self.k)
    }

    pub(crate) fn proj_lb2(&self, zq: &[f64], key: u64, frame: &TableFrame) -> f64 {
        let bits = 64 / self.k;
        let top = ((1u128 << bits) - 1) as f64;
        let mut u = [0u64; 64];
        let mut key = key >> (64 - bits * self.k);
        for b in 0..bits {
            for i in (0..self.k).rev() {
                u[i] |= (key & 1) << b;
                key >>= 1;
            }
        }
        let mut s = 0.0;
        for i in 0..self.k {
            let lo = if u[i] == 0 { f64::NEG_INFINITY } else { frame.value_at(i, u[i] as f64 / top) };
            let hi = if u[i] as f64 >= top { f64::INFINITY } else { frame.value_at(i, (u[i] + 1) as f64 / top) };
            let g = (lo - zq[i]).max(zq[i] - hi).max(0.0);
            s += g * g;
        }
        s
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct CodeBook {
    pub cuts: Vec<f64>,
    pub mean: Vec<f64>,
    pub var: Vec<f64>,
}

pub(crate) fn code_bytes(d: usize) -> usize {
    d.div_ceil(4)
}

impl CodeBook {
    pub(crate) fn fit(x: &[f64], d: usize) -> Self {
        let n = x.len() / d;
        assert!(n > 0, "code book needs at least one row");
        let mut cuts = vec![0.0; 3 * d];
        let mut col = vec![0.0; n];
        for j in 0..d {
            col.iter_mut().zip(x.iter().skip(j).step_by(d)).for_each(|(c, v)| *c = *v);
            col.sort_by(f64::total_cmp);
            for q in 0..3 {
                cuts[3 * j + q] = col[((q + 1) * n / 4).min(n - 1)];
            }
        }
        let mut book = Self { cuts, mean: vec![0.0; 4 * d], var: vec![0.0; 4 * d] };
        book.fit_moments(x, d);
        book
    }

    fn fit_moments(&mut self, x: &[f64], d: usize) {
        let mut cnt = vec![0usize; 4 * d];
        let mut sum = vec![0.0; 4 * d];
        let mut sq = vec![0.0; 4 * d];
        for r in x.chunks_exact(d) {
            for (j, &v) in r.iter().enumerate() {
                let at = 4 * j + self.bin(j, v);
                cnt[at] += 1;
                sum[at] += v;
                sq[at] += v * v;
            }
        }
        for at in 0..4 * d {
            let j = at / 4;
            if cnt[at] == 0 {
                self.mean[at] = self.cuts[3 * j + (at % 4).min(2)];
                continue;
            }
            let m = sum[at] / cnt[at] as f64;
            self.mean[at] = m;
            self.var[at] = (sq[at] / cnt[at] as f64 - m * m).max(0.0);
        }
    }

    fn bin(&self, j: usize, v: f64) -> usize {
        self.cuts[3 * j..3 * j + 3].iter().filter(|&&c| c <= v).count()
    }

    pub(crate) fn encode(&self, x: &[f64], out: &mut [u8]) {
        out.iter_mut().for_each(|b| *b = 0);
        for (j, &v) in x.iter().enumerate() {
            out[j / 4] |= (self.bin(j, v) as u8) << (2 * (j % 4));
        }
    }

    pub(crate) fn lut_lb(&self, query: &[f64], weights: &[f64]) -> Vec<f64> {
        let mut out = vec![0.0; self.mean.len()];
        for (at, o) in out.iter_mut().enumerate() {
            let (j, c) = (at / 4, at % 4);
            let lo = if c == 0 { f64::NEG_INFINITY } else { self.cuts[3 * j + c - 1] };
            let hi = if c == 3 { f64::INFINITY } else { self.cuts[3 * j + c] };
            let g = (lo - query[j]).max(query[j] - hi).max(0.0);
            *o = weights[j] * g * g;
        }
        out
    }

    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        self.cuts.iter().chain(&self.mean).chain(&self.var).flat_map(|v| v.to_le_bytes()).collect()
    }

    pub(crate) fn from_bytes(b: &[u8], d: usize) -> Self {
        let v: Vec<f64> = b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap_or([0; 8]))).collect();
        assert_eq!(v.len(), 11 * d, "code book record has the wrong length");
        Self { cuts: v[..3 * d].to_vec(), mean: v[3 * d..7 * d].to_vec(), var: v[7 * d..].to_vec() }
    }
}

pub(crate) fn lut_score(lut: &[f64], code: &[u8], d: usize) -> f64 {
    (0..d).map(|j| lut[4 * j + usize::from((code[j / 4] >> (2 * (j % 4))) & 3)]).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_keys_share_long_prefixes_for_near_points() {
        let h = TableHash::new(3, 3, 7);
        let x: Vec<f64> = (0..300).map(|i| (i % 97) as f64 / 97.0).collect();
        let f = h.fit(&x, &[1.0; 3]);
        let mut z = vec![0.0; 3];
        let a = h.key(&[0.5, 0.5, 0.5], &f, &mut z);
        let b = h.key(&[0.5001, 0.5, 0.5], &f, &mut z);
        let c = h.key(&[0.0, 1.0, 0.0], &f, &mut z);
        assert!((a ^ b).leading_zeros() > (a ^ c).leading_zeros());
    }

    #[test]
    fn projection_and_code_bounds_never_exceed_the_true_distance() {
        let d = 6;
        let x: Vec<f64> = (0..600).map(|i| ((i * 53) % 101) as f64 / 101.0).collect();
        let h = TableHash::new(d, 3, 2);
        let scale = [1.0, 2.0, 0.5, 1.0, 1.0, 3.0];
        let f = h.fit(&x, &scale);
        let book = CodeBook::fit(&x, d);
        let w: Vec<f64> = scale.iter().map(|s| 1.0 / (s * s)).collect();
        let q = [0.3, 0.9, 0.1, 0.5, 0.7, 0.2];
        let mut zq = vec![0.0; 3];
        h.project(&q, &scale, &mut zq);
        let lut = book.lut_lb(&q, &w);
        let mut z = vec![0.0; 3];
        let mut code = vec![0u8; code_bytes(d)];
        let mut tight = 0;
        for r in x.chunks_exact(d) {
            let truth: f64 = r.iter().zip(&q).zip(&w).map(|((a, b), c)| c * (a - b) * (a - b)).sum();
            let hb = h.proj_lb2(&zq, h.key(r, &f, &mut z), &f);
            book.encode(r, &mut code);
            let cb = lut_score(&lut, &code, d);
            assert!(hb <= truth * (1.0 + 1e-9) + 1e-12 && cb <= truth * (1.0 + 1e-9) + 1e-12);
            tight += usize::from(hb > 0.5 * truth);
        }
        assert!(tight > 0);
    }

    #[test]
    fn codes_round_trip_and_score_near_rows_lower() {
        let d = 5;
        let x: Vec<f64> = (0..4000).map(|i| ((i * 37) % 1000) as f64 / 1000.0).collect();
        let book = CodeBook::fit(&x, d);
        assert_eq!(CodeBook::from_bytes(&book.to_bytes(), d), book);
        let mut near = vec![0u8; code_bytes(d)];
        let mut far = near.clone();
        book.encode(&[0.1; 5], &mut near);
        book.encode(&[0.9; 5], &mut far);
        let lut = book.lut_lb(&[0.1; 5], &[1.0; 5]);
        assert!(lut_score(&lut, &near, d) < lut_score(&lut, &far, d));
    }
}
