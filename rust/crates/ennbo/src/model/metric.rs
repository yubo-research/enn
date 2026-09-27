//! Caller-set diagonal metric for disk BPANN models (the MBPANN_DISK mode).

use ndarray::Array1;

use super::EpistemicNearestNeighbors;
use crate::error::ENNError;

impl EpistemicNearestNeighbors {
    /// Set per-dimension distance scales: distances use `x / x_scale`.
    ///
    /// `rebuild=false` rescales the stored index in place (partition kept);
    /// `rebuild=true` re-indexes every row under the new metric. Later `add`
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
