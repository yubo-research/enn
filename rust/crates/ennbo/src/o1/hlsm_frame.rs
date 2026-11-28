use serde::{Deserialize, Serialize};

pub(crate) const KNOTS: usize = 64;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct TableFrame {
    pub scale: Vec<f64>,
    pub knots: Vec<Vec<f64>>,
}

impl TableFrame {
    pub(crate) fn from_projections(z: &[f64], k: usize, scale: &[f64]) -> Self {
        let knots = (0..k).map(|i| column_knots(z.iter().skip(i).step_by(k).copied().filter(|v| v.is_finite()).collect())).collect();
        Self { scale: scale.to_vec(), knots }
    }

    pub(crate) fn unit_of(&self, i: usize, z: f64) -> f64 {
        let k = &self.knots[i];
        let m = k.len() - 1;
        if z.is_nan() || z <= k[0] {
            return 0.0;
        }
        if z >= k[m] {
            return 1.0;
        }
        let j = (k.partition_point(|&v| v <= z) - 1).min(m - 1);
        let span = k[j + 1] - k[j];
        let f = if span > 0.0 { ((z - k[j]) / span).clamp(0.0, 1.0) } else { 0.0 };
        (j as f64 + f) / m as f64
    }

    pub(crate) fn value_at(&self, i: usize, t: f64) -> f64 {
        let k = &self.knots[i];
        let m = k.len() - 1;
        let s = t.clamp(0.0, 1.0) * m as f64;
        let j = (s.floor() as usize).min(m - 1);
        k[j] + (s - j as f64).clamp(0.0, 1.0) * (k[j + 1] - k[j])
    }

    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let head = [self.scale.len() as f64, self.knots.len() as f64, self.knots.first().map_or(0, Vec::len) as f64];
        head.iter().chain(&self.scale).chain(self.knots.iter().flatten()).flat_map(|v| v.to_le_bytes()).collect()
    }

    pub(crate) fn from_bytes(b: &[u8]) -> Option<Self> {
        let v: Vec<f64> = b.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap_or([0; 8]))).collect();
        let (d, k, m) = (*v.first()? as usize, *v.get(1)? as usize, *v.get(2)? as usize);
        if m < 2 || v.len() != 3 + d + k * m {
            return None;
        }
        let knots = v[3 + d..].chunks_exact(m).map(<[f64]>::to_vec).collect();
        Some(Self { scale: v[3..3 + d].to_vec(), knots })
    }
}

fn column_knots(mut col: Vec<f64>) -> Vec<f64> {
    if col.is_empty() {
        return vec![-1.0, 1.0];
    }
    col.sort_by(f64::total_cmp);
    let n = col.len();
    let q: Vec<f64> = (0..=KNOTS).map(|j| col[j * (n - 1) / KNOTS]).collect();
    let w = (q[KNOTS] - q[0]).max(1e-12 * q[0].abs().max(1e-300));
    let mut k = Vec::with_capacity(KNOTS + 3);
    k.push(q[0] - 0.5 * w);
    k.extend_from_slice(&q);
    k.push(q[KNOTS] + 0.5 * w);
    k
}

#[cfg(test)]
mod tests {
    use super::TableFrame;

    #[test]
    fn quantile_frame_spreads_dense_rows_and_round_trips() {
        let z: Vec<f64> = (0..2000).map(|i| if i < 1000 { 1e-9 * i as f64 } else { i as f64 }).collect();
        let f = TableFrame::from_projections(&z, 1, &[1.0]);
        let (a, b) = (f.unit_of(0, 1e-9 * 100.0), f.unit_of(0, 1e-9 * 900.0));
        assert!(b - a > 0.3, "dense half gets a large share of the unit range: {a} {b}");
        for t in [0.0, 0.1, 0.37, 0.5, 0.99, 1.0] {
            assert!((f.unit_of(0, f.value_at(0, t)) - t).abs() < 1e-9);
        }
        assert_eq!(TableFrame::from_bytes(&f.to_bytes()), Some(f));
    }
}
