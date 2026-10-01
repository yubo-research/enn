//! Caller-set diagonal metric for disk BPANN models (metric learning mode).

use ndarray::Array1;

use super::{EpistemicNearestNeighbors, scale_from_moments};
use crate::error::ENNError;
use crate::index::{is_disk_index_driver, IndexDriver};

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
    /// Index used for neighbor search.
    pub fn index_driver(&self) -> IndexDriver {
        self.backend_driver()
    }

    /// `true` when AUTO metric learning has been enabled.
    pub fn metric_learning_auto(&self) -> bool {
        self.auto_metric.is_some()
    }

    /// Grouped columns that stay unscaled, or the AUTO tied groups.
    pub fn set_tied_groups(&mut self, groups: Vec<Vec<usize>>) -> Result<(), ENNError> {
        crate::metric_weights::validate_tied_dims(&groups, self.num_dim)?;
        self.tied_groups = groups;
        Ok(())
    }

    /// Tied groups stored on the model, or the groups owned by AUTO.
    pub fn tied_groups(&self) -> &[Vec<usize>] {
        if let Some(metric) = &self.auto_metric {
            return metric.tied();
        }
        &self.tied_groups
    }

    /// `scale_x` scales from the running moments of `n` rows; `unscaled_dims` get 1.
    pub(crate) fn data_x_scale(&self, n: usize) -> Array1<f64> {
        let mut x_scale = scale_from_moments(n, self.num_dim, &self.x_sum, &self.x_sumsq, 1e-12);
        for &j in &self.unscaled_dims {
            x_scale[j] = 1.0;
        }
        x_scale
    }

    /// Move `scale_x` to `data_x_scale(n)`: incrementally for disk BPANN, else by a lazy rebuild.
    pub(crate) fn refresh_data_x_scale(&mut self, n: usize) -> Result<(), ENNError> {
        let x_scale = self.data_x_scale(n);
        if is_disk_index_driver(self.backend.driver()) {
            return self.apply_incremental_x_scale(x_scale);
        }
        self.x_scale = x_scale;
        self.backend.mark_index_stale();
        Ok(())
    }

    /// Dimensions `scale_x` leaves unscaled (e.g. one-hot categories, already in `{0, 1}`):
    /// their `x_scale` stays 1 while the other dimensions follow their standard deviations.
    pub fn set_unscaled_dims(&mut self, dims: Vec<usize>) -> Result<(), ENNError> {
        if let Some(j) = dims.iter().find(|&&j| j >= self.num_dim) {
            return Err(ENNError::InvalidParameter(format!(
                "unscaled dimension {j} is out of range for {} dimensions",
                self.num_dim
            )));
        }
        self.unscaled_dims = dims;
        if !self.scale_x || self.metric_fixed {
            return Ok(());
        }
        self.refresh_data_x_scale(self.num_obs)
    }

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
        if self.auto_metric.is_none() {
            return Err(ENNError::InvalidParameter(
                "set_metric_scale requires metric_learning=AUTO".into(),
            ));
        }
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
            crate::layout::EnnLayout::disk(dir.path().to_path_buf(), false),
            None,
        )
        .unwrap();
        assert!(model.set_metric_scale(Array1::from(vec![1.0]), false).is_err());
        assert!(model.set_metric_scale(Array1::from(vec![1.0, -1.0, 1.0]), false).is_err());
        model
            .enable_auto_metric(
                vec![],
                &x.slice(ndarray::s![..300, ..]),
                &y.slice(ndarray::s![..300, ..]),
            )
            .unwrap();
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
            crate::layout::EnnLayout::disk(dir.path().to_path_buf(), true),
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
    fn unscaled_dims_keep_scale_one_through_adds() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut x = rows(600, 3, 7);
        for mut row in x.rows_mut() {
            row[0] *= 10.0;
            row[2] = (row[2] > 0.5) as i32 as f64;
        }
        let y = Array2::zeros((600, 1));
        let open = |layout| {
            EpistemicNearestNeighbors::new_with_storage(
                x.slice(ndarray::s![..300, ..]).to_owned(),
                y.slice(ndarray::s![..300, ..]).to_owned(),
                None,
                layout,
                None,
            )
            .unwrap()
        };
        let flat = open(crate::layout::EnnLayout::memory(IndexDriver::Flat, true));
        let disk = open(crate::layout::EnnLayout::disk(dir.path().to_path_buf(), true));
        for mut model in [flat, disk] {
            assert!(model.set_unscaled_dims(vec![3]).is_err());
            model.set_unscaled_dims(vec![2]).unwrap();
            assert_eq!(model.x_scale[2], 1.0);
            model
                .add(&x.slice(ndarray::s![300.., ..]), &y.slice(ndarray::s![300.., ..]), None)
                .unwrap();
            let std = column_std(&x);
            let applied = model.x_scale_row().row(0).to_owned();
            assert_eq!(applied[2], 1.0);
            assert!((applied[0] / std[0]).ln().abs() <= super::SCALE_X_RESCALE_TOL + 1e-9);
            let q = rows(5, 3, 8);
            let got = model.neighbors(&q.view(), 4, false).unwrap();
            for r in 0..q.nrows() {
                assert_eq!(got.row(r).to_vec(), exact_topk(&x, &q.row(r).to_vec(), &applied, 4));
            }
        }
    }

    #[test]
    fn metric_scale_rejected_for_in_memory_backend() {
        let mut model = EpistemicNearestNeighbors::new(
            rows(10, 2, 3),
            Array2::zeros((10, 1)),
            None,
            false,
            IndexDriver::Flat,
        )
        .unwrap();
        assert!(model.set_metric_scale(Array1::from(vec![1.0, 1.0]), false).is_err());
    }
}
