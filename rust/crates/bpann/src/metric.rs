use ndarray::Array1;

use crate::backend::BpannBackend;
use crate::error::BpannError;
use crate::index::IncrementalIndex;
use crate::store_file::{bpann_read_json, bpann_write_json};

pub const METRIC_MARKER_FILE: &str = "metric_x_scale.json";

impl IncrementalIndex {
    pub fn rescale_coords(
        &mut self,
        ratio: &[f64],
        coords: impl FnMut(u32, &mut Vec<f32>) -> Result<(), BpannError>,
    ) -> Result<(), BpannError> {
        match self.tree.as_mut() {
            Some(tree) => tree.scale_rows(ratio, coords),
            None => Ok(()),
        }
    }
}

#[cfg(test)]
impl IncrementalIndex {
    /// Test helper: rescale from unit scale, re-reading rows from `store`.
    pub(crate) fn rescale_unit(
        &mut self,
        ratio: &[f64],
        store: &crate::mmap_store::MmapColumnStore,
    ) {
        let tree_dim = crate::index::tree_store::pad_dim(ratio.len());
        let x_scale: Vec<f64> = ratio.iter().map(|r| 1.0 / r).collect();
        self.rescale_coords(ratio, |row, buf| {
            crate::distance::bpann_row_to_f32(
                store.mmap_row_slice(row as usize)?,
                true,
                &x_scale,
                buf,
            );
            buf.resize(tree_dim, 0.0);
            Ok(())
        })
        .unwrap();
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
        bpann_write_json(
            &self.work_dir.join(METRIC_MARKER_FILE),
            &self.x_scale.to_vec(),
        )
    }

    pub fn rescale_metric(&mut self, x_scale: &Array1<f64>) -> Result<(), BpannError> {
        self.validate_metric_scale(x_scale)?;
        let ratio: Vec<f64> = (0..self.num_dim)
            .map(|j| self.current_scale(j) / x_scale[j])
            .collect();
        let tree_dim = crate::index::tree_store::pad_dim(self.num_dim);
        let (train_x, new_scale) = (&self.train_x, x_scale.to_vec());
        self.index.rescale_coords(&ratio, |row, buf| {
            crate::distance::bpann_row_to_f32(
                train_x.mmap_row_slice(row as usize)?,
                true,
                &new_scale,
                buf,
            );
            buf.resize(tree_dim, 0.0);
            Ok(())
        })?;
        *self.small_n_x_cache.lock().expect("small_n_x_cache") = None;
        self.scale_x = true;
        self.x_scale = x_scale.to_owned();
        self.write_metric_marker()
    }

    pub fn rebuild_metric(&mut self, x_scale: &Array1<f64>) -> Result<(), BpannError> {
        self.rescale_metric(x_scale)?;
        self.index.start_morph();
        Ok(())
    }

    pub fn restore_metric_on_reopen(&mut self) -> Result<bool, BpannError> {
        let marker = self.work_dir.join(METRIC_MARKER_FILE);
        let Some(x_scale) = bpann_read_json::<Vec<f64>>(&marker)? else {
            return Ok(false);
        };
        let x_scale = Array1::from(x_scale);
        self.validate_metric_scale(&x_scale)?;
        self.scale_x = true;
        self.x_scale = x_scale;
        self.mark_index_stale();
        Ok(true)
    }

    pub fn metric_scale(&self) -> Option<&Array1<f64>> {
        self.scale_x.then_some(&self.x_scale)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::page::Page;
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
            } => vectors
                .iter()
                .chain(stored_centroid.iter())
                .cloned()
                .collect(),
        }
    }

    fn all_coords(b: &BpannBackend) -> Vec<Vec<f32>> {
        let tree = b.index.tree.as_ref().expect("tree");
        let mut out = Vec::new();
        for id in 0..tree.num_pages() as u32 {
            out.extend(page_coords(&tree.page(id)));
            if tree.store.is_leaf(id) {
                out.extend(
                    tree.store
                        .leaf_rows(id)
                        .chunks_exact(tree.store.num_dim())
                        .map(<[f32]>::to_vec),
                );
            }
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
                // Leaf rows are int8-decoded: allow a code step.
                assert!((u[j] - expect).abs() <= 0.03 * (1.0 + expect.abs()));
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
                    let s: f64 = (0..4)
                        .map(|j| ((q[[r, j]] - x[[i, j]]) / scale[j]).powi(2))
                        .sum();
                    (s, i as i64)
                })
                .collect();
            d.sort_by(|u, v| u.0.partial_cmp(&v.0).unwrap());
            let exact: Vec<i64> = d.iter().take(5).map(|t| t.1).collect();
            assert_eq!(ia.row(r).to_vec(), exact);
            assert_eq!(ib.row(r).to_vec(), exact);
        }
    }

    fn assert_exact_top5(
        b: &mut BpannBackend,
        x: &Array2<f64>,
        n: usize,
        scale: &[f64],
        q: &Array2<f64>,
    ) {
        let (_, ids) = b.search(&q.view(), 5, false).unwrap();
        for r in 0..q.nrows() {
            let mut d: Vec<(f64, i64)> = (0..n)
                .map(|i| {
                    let s: f64 = (0..x.ncols())
                        .map(|j| ((q[[r, j]] - x[[i, j]]) / scale[j]).powi(2))
                        .sum();
                    (s, i as i64)
                })
                .collect();
            d.sort_by(|u, v| u.0.total_cmp(&v.0));
            assert_eq!(
                ids.row(r).to_vec(),
                d.iter().take(5).map(|t| t.1).collect::<Vec<_>>()
            );
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
            b.append_rows(&x.slice(ndarray::s![n..n + 500, ..]), &y.view(), None)
                .unwrap();
            b.ensure_index_sync().unwrap();
            n += 500;
        }
        assert_exact_top5(&mut b, &x, n, &scale, &q);
    }

    #[test]
    fn rescale_replaces_loose_radii_with_exact_bounds() {
        let x = random_rows(4500, 3, 9);
        let scale = Array1::from(vec![0.05, 1.0, 8.0]);
        let dir = TempDir::new().unwrap();
        let mut backend = backend_with_rows(&dir, &x);
        let before = page_radii(&backend);
        backend.rescale_metric(&scale).unwrap();
        let after = page_radii(&backend);
        let grow = scale.iter().map(|s| 1.0 / s).fold(0.0, f64::max) as f32;
        assert_eq!(before.len(), after.len());
        assert!(leaves_fit(&backend));
        for (&old, &new) in before.iter().zip(&after) {
            assert!(
                new <= old * grow * 1.001 + 1e-3,
                "{new} exceeds loose bound {}",
                old * grow
            );
        }
        let tighter = before
            .iter()
            .zip(&after)
            .filter(|(&old, &new)| new < old * grow * 0.8)
            .count();
        assert!(
            tighter > before.len() / 2,
            "{tighter} of {} radii are below 0.8 of the max-factor bound",
            before.len()
        );
    }

    fn page_radii(backend: &BpannBackend) -> Vec<f32> {
        let tree = backend.index.tree.as_ref().expect("tree");
        (0..tree.num_pages() as u32)
            .map(|id| tree.store.radius(id))
            .collect()
    }

    fn leaves_fit(backend: &BpannBackend) -> bool {
        let tree = backend.index.tree.as_ref().expect("tree");
        let dim = tree.store.num_dim();
        for page in 0..tree.num_pages() as u32 {
            if !tree.store.is_leaf(page) {
                continue;
            }
            let Some(parent) = tree.store.parent(page) else {
                continue;
            };
            let Some(slot) = tree
                .store
                .ids(parent)
                .iter()
                .position(|&child| child == page)
            else {
                continue;
            };
            let centroid = &tree.store.block(parent)[slot * dim..(slot + 1) * dim];
            let radius = tree.store.radius(page);
            let slack = tree.store.leaf_err(page);
            let outside = tree.store.leaf_rows(page).chunks_exact(dim).any(|row| {
                crate::index::tree_counts::dist(row, centroid) > radius * 1.0001 + 1e-4 + slack
            });
            if outside {
                return false;
            }
        }
        true
    }

    #[test]
    fn rejects_bad_scale_and_restores_on_reopen() {
        let x = random_rows(50, 2, 3);
        let dir = TempDir::new().unwrap();
        let mut b = backend_with_rows(&dir, &x);
        assert!(b.rescale_metric(&Array1::from(vec![1.0])).is_err());
        assert!(b.rescale_metric(&Array1::from(vec![1.0, 0.0])).is_err());
        let scale = Array1::from(vec![1.0, 2.0]);
        assert_eq!(b.metric_scale(), None);
        b.rescale_metric(&scale).unwrap();
        drop(b);
        let mut reopened = BpannBackend::reopen(dir.path().to_path_buf()).unwrap();
        assert!(reopened.restore_metric_on_reopen().unwrap());
        assert_eq!(reopened.indexed_rows(), 0);
        assert_eq!(reopened.metric_scale(), Some(&scale));
    }

    #[test]
    fn reopen_without_a_learned_metric_restores_nothing() {
        let dir = TempDir::new().unwrap();
        drop(backend_with_rows(&dir, &random_rows(50, 2, 3)));
        let mut reopened = BpannBackend::reopen(dir.path().to_path_buf()).unwrap();
        assert!(!reopened.restore_metric_on_reopen().unwrap());
        assert_eq!(reopened.metric_scale(), None);
    }

    #[test]
    fn invalid_persisted_metric_scale_is_rejected_on_reopen() {
        let dir = TempDir::new().unwrap();
        drop(backend_with_rows(&dir, &random_rows(50, 2, 3)));
        bpann_write_json(&dir.path().join(METRIC_MARKER_FILE), &vec![1.0]).unwrap();
        let mut reopened = BpannBackend::reopen(dir.path().to_path_buf()).unwrap();
        assert!(reopened.restore_metric_on_reopen().is_err());
    }
}
