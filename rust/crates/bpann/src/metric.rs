//! Metric changes for a live BPANN index (metric learning mode).
//!
//! Every coordinate the index stores is `x_d / x_scale_d` or a mean of such
//! values, so a new diagonal metric maps each stored value by
//! `v_d * old_scale_d / new_scale_d` exactly. [`BpannBackend::rescale_metric`]
//! applies that map in place and keeps the partition; [`BpannBackend::rebuild_metric`]
//! also re-partitions every row under the new metric, in place, by moving page and
//! row references a little per later insert ([`crate::index::tree_morph`]).

use std::fs;
use std::sync::Arc;

use ndarray::Array1;

use crate::backend::BpannBackend;
use crate::error::BpannError;
use crate::index::page::Page;
use crate::index::{BpannIndex, IncrementalIndex};

/// Present in `work_dir` while the index is expressed in a caller-set metric.
pub const METRIC_MARKER_FILE: &str = "metric_x_scale.json";

fn scale_f32(values: &mut [f32], ratio: &[f64]) {
    for (v, &r) in values.iter_mut().zip(ratio) {
        *v = (f64::from(*v) * r) as f32;
    }
}

fn rescale_page(page: &mut Page, ratio: &[f64]) {
    match page {
        Page::Internal { centroids, .. } => {
            centroids.iter_mut().for_each(|c| scale_f32(c, ratio));
        }
        Page::Leaf {
            vectors,
            stored_centroid,
            ..
        } => {
            vectors.iter_mut().for_each(|v| scale_f32(v, ratio));
            if let Some(c) = stored_centroid.as_mut() {
                scale_f32(c, ratio);
            }
        }
    }
}

pub fn rescale_index_coords(index: &mut BpannIndex, ratio: &[f64]) {
    index.pages.iter_mut().for_each(|p| rescale_page(p, ratio));
}

impl IncrementalIndex {
    /// Multiply every stored coordinate by `ratio`; a re-partition in progress
    /// carries on under the new coordinates.
    pub fn rescale_coords(&mut self, ratio: &[f64]) {
        for index in &mut self.indices {
            self.counts.write_page_centroids(index);
            rescale_index_coords(index, ratio);
        }
        self.counts.scale_rows(ratio);
    }
}

impl BpannBackend {
    fn validate_metric_scale(&self, x_scale: &Array1<f64>) -> Result<(), BpannError> {
        if x_scale.len() != self.num_dim {
            return Err(BpannError::InvalidShape {
                expected: vec![self.num_dim],
                got: vec![x_scale.len()],
            });
        }
        if !x_scale.iter().all(|s| s.is_finite() && *s > 0.0) {
            return Err(BpannError::InvalidParameter(
                "metric x_scale entries must be finite and > 0".to_string(),
            ));
        }
        Ok(())
    }

    fn current_scale(&self, j: usize) -> f64 {
        if self.scale_x {
            self.x_scale[j]
        } else {
            1.0
        }
    }

    fn write_metric_marker(&self) -> Result<(), BpannError> {
        let body = serde_json::to_string(&self.x_scale.to_vec())
            .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        fs::write(self.work_dir.join(METRIC_MARKER_FILE), body)
            .map_err(|e| BpannError::InvalidParameter(e.to_string()))
    }

    fn rescale_small_n_cache(&self, ratio: &[f64]) {
        let mut guard = self.small_n_x_cache.lock().expect("small_n_x_cache");
        if let Some((n, data)) = guard.take() {
            let mut flat = data.to_vec();
            for row in flat.chunks_mut(self.num_dim) {
                scale_f32(row, ratio);
            }
            *guard = Some((n, Arc::from(flat.into_boxed_slice())));
        }
    }

    /// Switch to metric `x_scale` (distance uses `x / x_scale`) by rescaling the
    /// stored index in place. The partition is kept, so no rows are re-read.
    pub fn rescale_metric(&mut self, x_scale: &Array1<f64>) -> Result<(), BpannError> {
        self.validate_metric_scale(x_scale)?;
        let ratio: Vec<f64> = (0..self.num_dim)
            .map(|j| self.current_scale(j) / x_scale[j])
            .collect();
        self.index.rescale_coords(&ratio);
        self.rescale_small_n_cache(&ratio);
        self.scale_x = true;
        self.x_scale = x_scale.to_owned();
        self.write_metric_marker()
    }

    /// Switch to metric `x_scale` by rescaling in place, then re-partition every
    /// row under it, in place, during the following inserts. The index answers
    /// queries under the new metric immediately.
    pub fn rebuild_metric(&mut self, x_scale: &Array1<f64>) -> Result<(), BpannError> {
        self.rescale_metric(x_scale)?;
        self.index.start_morph();
        Ok(())
    }

    /// On reopen, an index persisted under a caller-set metric cannot be matched
    /// to the reopened (identity) metric, so it is discarded and rebuilt lazily.
    pub fn discard_metric_index_on_reopen(&mut self) -> Result<bool, BpannError> {
        let marker = self.work_dir.join(METRIC_MARKER_FILE);
        if !marker.exists() {
            return Ok(false);
        }
        self.mark_index_stale();
        fs::remove_file(&marker).map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array2;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use tempfile::TempDir;

    fn random_rows(n: usize, d: usize, seed: u64) -> Array2<f64> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        Array2::from_shape_fn((n, d), |_| rng.gen::<f64>())
    }

    fn backend_with_rows(dir: &TempDir, x: &Array2<f64>) -> BpannBackend {
        let y = Array2::zeros((x.nrows(), 1));
        let mut b = BpannBackend::new_empty(dir.path().to_path_buf(), x.ncols(), 1).unwrap();
        for lo in (0..x.nrows()).step_by(1500) {
            let hi = (lo + 1500).min(x.nrows());
            b.append_rows(
                &x.slice(ndarray::s![lo..hi, ..]),
                &y.slice(ndarray::s![lo..hi, ..]),
                None,
            )
            .unwrap();
            b.ensure_index_sync().unwrap();
        }
        b
    }

    fn page_coords(p: &Page) -> Vec<Vec<f32>> {
        match p {
            Page::Internal { centroids, .. } => centroids.clone(),
            Page::Leaf {
                vectors,
                stored_centroid,
                ..
            } => vectors.iter().chain(stored_centroid.iter()).cloned().collect(),
        }
    }

    fn all_coords(b: &BpannBackend) -> Vec<Vec<f32>> {
        let mut out = Vec::new();
        for index in &b.index.indices {
            let mut index = index.clone();
            b.index.counts.write_page_centroids(&mut index);
            out.extend(index.pages.iter().flat_map(page_coords));
        }
        out
    }

    #[test]
    fn rescale_maps_every_stored_coordinate_by_scale_ratio() {
        let x = random_rows(4500, 3, 7);
        let scale = Array1::from(vec![0.5, 2.0, 1.25]);
        let dir_a = TempDir::new().unwrap();
        let mut a = backend_with_rows(&dir_a, &x);
        let dir_b = TempDir::new().unwrap();
        let b = backend_with_rows(&dir_b, &x);
        a.rescale_metric(&scale).unwrap();
        let (ca, cb) = (all_coords(&a), all_coords(&b));
        assert!(!ca.is_empty());
        assert_eq!(ca.len(), cb.len());
        for (u, v) in ca.iter().zip(cb.iter()) {
            for j in 0..3 {
                let expect = v[j] / scale[j] as f32;
                assert!((u[j] - expect).abs() <= 1e-5 * (1.0 + expect.abs()));
            }
        }
        assert!(dir_a.path().join(METRIC_MARKER_FILE).exists());
    }

    #[test]
    fn rescale_and_rebuild_give_exact_neighbors_small_n() {
        let x = random_rows(3000, 4, 11);
        let scale = Array1::from(vec![0.2, 1.0, 3.0, 0.7]);
        let q = random_rows(20, 4, 12);
        let dir_a = TempDir::new().unwrap();
        let mut a = backend_with_rows(&dir_a, &x);
        let _ = a.search(&q.view(), 5, false).unwrap();
        a.rescale_metric(&scale).unwrap();
        let (_, ia) = a.search(&q.view(), 5, false).unwrap();
        let dir_b = TempDir::new().unwrap();
        let mut b = backend_with_rows(&dir_b, &x);
        b.rebuild_metric(&scale).unwrap();
        let (_, ib) = b.search(&q.view(), 5, false).unwrap();
        for r in 0..q.nrows() {
            let mut d: Vec<(f64, i64)> = (0..x.nrows())
                .map(|i| {
                    let s: f64 = (0..4).map(|j| ((q[[r, j]] - x[[i, j]]) / scale[j]).powi(2)).sum();
                    (s, i as i64)
                })
                .collect();
            d.sort_by(|u, v| u.0.partial_cmp(&v.0).unwrap());
            let exact: Vec<i64> = d.iter().take(5).map(|t| t.1).collect();
            assert_eq!(ia.row(r).to_vec(), exact);
            assert_eq!(ib.row(r).to_vec(), exact);
        }
    }

    fn assert_exact_top5(b: &mut BpannBackend, x: &Array2<f64>, n: usize, scale: &[f64], q: &Array2<f64>) {
        let (_, ids) = b.search(&q.view(), 5, false).unwrap();
        for r in 0..q.nrows() {
            let mut d: Vec<(f64, i64)> = (0..n)
                .map(|i| {
                    let s: f64 = (0..x.ncols()).map(|j| ((q[[r, j]] - x[[i, j]]) / scale[j]).powi(2)).sum();
                    (s, i as i64)
                })
                .collect();
            d.sort_by(|u, v| u.0.total_cmp(&v.0));
            assert_eq!(ids.row(r).to_vec(), d.iter().take(5).map(|t| t.1).collect::<Vec<_>>());
        }
    }

    #[test]
    fn large_metric_change_gives_exact_neighbors_during_and_after_the_morph() {
        let (n0, scale) = (12_000, [0.3, 1.0, 2.5]);
        let x = random_rows(n0 + n0 / 8, 3, 21);
        let q = random_rows(10, 3, 22);
        let dir = TempDir::new().unwrap();
        let mut b = backend_with_rows(&dir, &x.slice(ndarray::s![..n0, ..]).to_owned());
        b.rebuild_metric(&Array1::from(scale.to_vec())).unwrap();
        assert!(b.index.morphing());
        assert_exact_top5(&mut b, &x, n0, &scale, &q);
        let y = Array2::zeros((500, 1));
        let mut n = n0;
        while b.index.morphing() {
            assert!(n + 500 <= x.nrows(), "morph still running at {n} rows");
            assert_exact_top5(&mut b, &x, n, &scale, &q);
            b.append_rows(&x.slice(ndarray::s![n..n + 500, ..]), &y.view(), None).unwrap();
            b.ensure_index_sync().unwrap();
            n += 500;
        }
        assert_exact_top5(&mut b, &x, n, &scale, &q);
    }

    #[test]
    fn rejects_bad_scale_and_discards_on_reopen() {
        let x = random_rows(50, 2, 3);
        let dir = TempDir::new().unwrap();
        let mut b = backend_with_rows(&dir, &x);
        assert!(b.rescale_metric(&Array1::from(vec![1.0])).is_err());
        assert!(b.rescale_metric(&Array1::from(vec![1.0, 0.0])).is_err());
        b.rescale_metric(&Array1::from(vec![1.0, 2.0])).unwrap();
        assert!(b.discard_metric_index_on_reopen().unwrap());
        assert_eq!(b.indexed_rows(), 0);
        assert!(!b.discard_metric_index_on_reopen().unwrap());
    }
}
