//! Caller-set diagonal metric for disk BPANN models (metric learning mode).

use ndarray::Array1;

use super::EpistemicNearestNeighbors;
use crate::error::ENNError;

/// Disk `scale_x`: largest per-dimension `|log(new / applied)|` left unapplied after an `add`.
pub const SCALE_X_RESCALE_TOL: f64 = 0.01;
/// Disk `scale_x`: largest per-dimension `|log(new / built)|` served by an in-place rescale;
/// beyond it the index is re-partitioned (same rule as `MBPANNMetric`'s default).
pub const SCALE_X_REBUILD_DRIFT: f64 = std::f64::consts::LN_2;

fn max_log_ratio(a: &Array1<f64>, b: &Array1<f64>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(u, v)| (u / v).ln().abs())
        .fold(0.0, f64::max)
}

impl EpistemicNearestNeighbors {
    /// Incremental `scale_x` for disk BPANN: move the index to the data-moment scale
    /// `x_scale` without re-reading rows, unless it drifted far from the partition's scale.
    pub(crate) fn apply_incremental_x_scale(&mut self, x_scale: Array1<f64>) -> Result<(), ENNError> {
        if max_log_ratio(&x_scale, &self.x_scale) <= SCALE_X_RESCALE_TOL {
            return Ok(());
        }
        let rebuild = max_log_ratio(&x_scale, &self.built_x_scale) > SCALE_X_REBUILD_DRIFT;
        self.backend.set_metric_scale(&x_scale, rebuild)?;
        if rebuild {
            self.built_x_scale = x_scale.clone();
        }
        self.x_scale = x_scale;
        Ok(())
    }

    /// Set per-dimension distance scales: distances use `x / x_scale`.
    ///
    /// `rebuild=false` rescales the stored index in place (partition kept);
    /// `rebuild=true` also re-partitions it in place, a little on each later add. Later `add`
    /// calls keep this metric instead of re-deriving `x_scale` from data moments.
    pub fn set_metric_scale(&mut self, x_scale: Array1<f64>, rebuild: bool) -> Result<(), ENNError> {
        if x_scale.len() != self.num_dim {
            return Err(ENNError::InvalidShape {
                expected: vec![self.num_dim],
                got: vec![x_scale.len()],
            });
        }
        if !x_scale.iter().all(|s| s.is_finite() && *s > 0.0) {
            return Err(ENNError::InvalidParameter(
                "metric x_scale entries must be finite and > 0".to_string(),
            ));
        }
        self.backend.set_metric_scale(&x_scale, rebuild)?;
        self.scale_x = true;
        self.metric_fixed = true;
        self.x_scale = x_scale;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::backend::EnnStorage;
    use crate::{EpistemicNearestNeighbors, IndexDriver};
    use ndarray::{Array1, Array2};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn rows(n: usize, d: usize, seed: u64) -> Array2<f64> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        Array2::from_shape_fn((n, d), |_| rng.gen::<f64>())
    }

    fn exact_topk(x: &Array2<f64>, q: &[f64], scale: &Array1<f64>, k: usize) -> Vec<usize> {
        let mut d: Vec<(f64, usize)> = (0..x.nrows())
            .map(|i| {
                let s = (0..q.len()).map(|j| ((q[j] - x[[i, j]]) / scale[j]).powi(2)).sum();
                (s, i)
            })
            .collect();
        d.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        d.into_iter().take(k).map(|t| t.1).collect()
    }

    #[test]
    fn metric_scale_drives_neighbors_and_survives_add() {
        let dir = tempfile::TempDir::new().unwrap();
        let x = rows(400, 3, 1);
        let y = Array2::zeros((400, 1));
        let mut model = EpistemicNearestNeighbors::new_with_storage(
            x.slice(ndarray::s![..300, ..]).to_owned(),
            y.slice(ndarray::s![..300, ..]).to_owned(),
            None,
            false,
            IndexDriver::BpAnnDisk,
            EnnStorage::Disk,
            Some(dir.path().to_path_buf()),
            None,
        )
        .unwrap();
        assert!(model.set_metric_scale(Array1::from(vec![1.0]), false).is_err());
        assert!(model.set_metric_scale(Array1::from(vec![1.0, -1.0, 1.0]), false).is_err());
        let scale = Array1::from(vec![0.25, 1.0, 4.0]);
        model.set_metric_scale(scale.clone(), false).unwrap();
        model
            .add(
                &x.slice(ndarray::s![300.., ..]),
                &y.slice(ndarray::s![300.., ..]),
                None,
            )
            .unwrap();
        assert_eq!(model.x_scale_row().row(0).to_vec(), scale.to_vec());
        let q = rows(5, 3, 2);
        let got = model.neighbors(&q.view(), 4, false).unwrap();
        for r in 0..q.nrows() {
            let exact = exact_topk(&x, &q.row(r).to_vec(), &scale, 4);
            assert_eq!(got.row(r).to_vec(), exact);
        }
        model.set_metric_scale(Array1::from(vec![1.0, 1.0, 0.5]), true).unwrap();
        assert!(model.is_scale_x());
    }

    fn column_std(x: &Array2<f64>) -> Array1<f64> {
        x.std_axis(ndarray::Axis(0), 0.0)
    }

    #[test]
    fn disk_scale_x_tracks_moments_incrementally() {
        let dir = tempfile::TempDir::new().unwrap();
        let widths = [0.01, 1.0, 100.0];
        let mut x = rows(3000, 3, 5);
        for mut row in x.rows_mut() {
            for j in 0..3 {
                row[j] *= widths[j];
            }
        }
        let y = Array2::zeros((3000, 1));
        let mut model = EpistemicNearestNeighbors::new_with_storage(
            x.slice(ndarray::s![..20, ..]).to_owned(),
            y.slice(ndarray::s![..20, ..]).to_owned(),
            None,
            true,
            IndexDriver::BpAnnDisk,
            EnnStorage::Disk,
            Some(dir.path().to_path_buf()),
            None,
        )
        .unwrap();
        for lo in (20..3000).step_by(245) {
            let hi = (lo + 245).min(3000);
            model
                .add(&x.slice(ndarray::s![lo..hi, ..]), &y.slice(ndarray::s![lo..hi, ..]), None)
                .unwrap();
            model.ensure_index_sync().unwrap();
        }
        let applied = model.x_scale_row().row(0).to_owned();
        let std = column_std(&x);
        assert!(super::max_log_ratio(&applied, &std) <= super::SCALE_X_RESCALE_TOL + 1e-9);
        assert!(super::max_log_ratio(&model.built_x_scale, &std) <= super::SCALE_X_REBUILD_DRIFT);
        let q = rows(5, 3, 6) * &Array1::from(widths.to_vec());
        let got = model.neighbors(&q.view(), 4, false).unwrap();
        for r in 0..q.nrows() {
            assert_eq!(got.row(r).to_vec(), exact_topk(&x, &q.row(r).to_vec(), &applied, 4));
        }
    }

    #[test]
    fn metric_scale_rejected_for_in_memory_backend() {
        let mut model = EpistemicNearestNeighbors::new(
            rows(10, 2, 3),
            Array2::zeros((10, 1)),
            None,
            false,
            IndexDriver::Exact,
        )
        .unwrap();
        assert!(model.set_metric_scale(Array1::from(vec![1.0, 1.0]), false).is_err());
    }
}
