
use ndarray::{Array2, ArrayView2};

use crate::index::IndexDriver;
use crate::model::EpistemicNearestNeighbors;

const SMALL_ROWS: usize = 300;
const COMPONENT_LIMIT: usize = 3_000_000;
pub(super) const F64_OVERSAMPLE: usize = 32;

type BoundedNeighbor = (f64, i64, f64);

pub(super) struct SmallFlatNeighbors {
    rows: Vec<Vec<BoundedNeighbor>>,
    possible_predecessors: Vec<usize>,
    n_train: usize,
    max_k: usize,
}

fn column_distances(columns: &[f64], query: &[f64], scale: Option<&[f64]>, distances: &mut [f64]) {
    distances.fill(0.0);
    let n = distances.len();
    for (dimension, (column, &q)) in columns.chunks_exact(n).zip(query).enumerate() {
        if let Some(scale) = scale {
            let q = q / scale[dimension];
            for (distance, &point) in distances.iter_mut().zip(column) {
                let delta = q - point / scale[dimension];
                *distance += delta * delta;
            }
        } else {
            for (distance, &point) in distances.iter_mut().zip(column) {
                let delta = q - point;
                *distance += delta * delta;
            }
        }
    }
}

fn column_norms(columns: &[f64], n: usize, scale: Option<&[f64]>) -> Vec<f64> {
    let mut norms = vec![0.0; n];
    for (dimension, column) in columns.chunks_exact(n).enumerate() {
        for (norm, &point) in norms.iter_mut().zip(column) {
            let point = scale.map_or(point, |s| point / s[dimension]);
            *norm += point * point;
        }
    }
    norms
}

impl SmallFlatNeighbors {
    pub(super) fn new(
        model: &EpistemicNearestNeighbors,
        x: &ArrayView2<f64>,
        max_k: usize,
    ) -> Option<Self> {
        if model.backend_driver() != IndexDriver::Flat {
            return None;
        }
        Self::from_arrays(
            &model.train_x_view_opt()?,
            x,
            model.scale_x.then_some(model.x_scale.as_slice()?),
            max_k,
        )
    }

    pub(super) fn from_arrays(
        train: &ArrayView2<f64>,
        x: &ArrayView2<f64>,
        scale: Option<&[f64]>,
        max_k: usize,
    ) -> Option<Self> {
        let n = train.nrows();
        let max_k = max_k.min(n);
        if max_k == 0
            || n > SMALL_ROWS
            || x.nrows() > SMALL_ROWS
            || x.ncols() == 0
            || x.ncols() != train.ncols()
            || n.saturating_mul(x.len()) > COMPONENT_LIMIT
        {
            return None;
        }
        let transposed = train.t();
        let training = transposed.as_standard_layout();
        let columns = training.as_slice()?;
        let queries = x.as_standard_layout();
        let norms = column_norms(columns, n, scale);
        let mut distances = vec![0.0; n];
        let roundoff = 16.0 * f32::EPSILON as f64 * (x.ncols() + 1) as f64;
        let underflow = 16.0 * f32::MIN_POSITIVE as f64 * (x.ncols() + 1) as f64;
        let mut rows = Vec::with_capacity(x.nrows());
        let mut possible_predecessors = Vec::with_capacity(x.nrows());
        for query in queries.as_slice()?.chunks_exact(x.ncols()) {
            let norm: f64 = query
                .iter()
                .enumerate()
                .map(|(i, &q)| {
                    let q = scale.map_or(q, |s| q / s[i]);
                    q * q
                })
                .sum();
            column_distances(columns, query, scale, &mut distances);
            let mut row = Vec::with_capacity(n);
            for (j, (&distance, &train_norm)) in distances.iter().zip(&norms).enumerate() {
                let magnitude = norm + train_norm;
                if !magnitude.is_finite() || magnitude >= f32::MAX as f64 / 16.0 {
                    return None;
                }
                row.push((
                    distance.max(0.0),
                    j as i64,
                    roundoff * magnitude + underflow,
                ));
            }
            let compare =
                |a: &BoundedNeighbor, b: &BoundedNeighbor| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1));
            if max_k < n {
                row.select_nth_unstable_by(max_k - 1, compare);
            }
            row[..max_k].sort_unstable_by(compare);
            let upper = row[..max_k].iter().map(|p| p.0 + p.2).fold(0.0, f64::max);
            possible_predecessors.push(row.iter().filter(|p| p.0 - p.2 <= upper).count());
            rows.push(row);
        }
        Some(Self {
            rows,
            possible_predecessors,
            n_train: n,
            max_k,
        })
    }

    pub(super) fn certified_row(&self, query: usize, k: usize) -> Option<&[(f64, i64, f64)]> {
        let k = k.min(self.n_train);
        if k == 0 || k > self.max_k {
            return None;
        }
        let row = self.rows.get(query)?;
        let fetched = k.saturating_add(F64_OVERSAMPLE).min(self.n_train);
        if self.possible_predecessors[query] <= fetched {
            return Some(&row[..k]);
        }
        let upper = row[..k].iter().map(|p| p.0 + p.2).fold(0.0, f64::max);
        if row.iter().filter(|p| p.0 - p.2 <= upper).count() > fetched {
            return None;
        }
        Some(&row[..k])
    }

    pub(super) fn neighbors(&self, k: usize) -> Option<(Array2<f64>, Array2<i64>)> {
        let k = k.min(self.n_train);
        if k == 0 || k > self.max_k {
            return None;
        }
        let mut distances = Array2::zeros((self.rows.len(), k));
        let mut indices = Array2::zeros((self.rows.len(), k));
        for i in 0..self.rows.len() {
            for (j, &(distance, index, _)) in self.certified_row(i, k)?.iter().enumerate() {
                distances[[i, j]] = distance;
                indices[[i, j]] = index;
            }
        }
        Some((distances, indices))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::posterior::neighbor::{dist2s_for_neighbor_indices, sort_neighbors_by_dist2};
    use ndarray::{array, Axis, Slice};

    #[test]
    fn certified_neighbors_match_faiss_candidates_with_and_without_scaling() {
        for weighted in [false, true] {
            let x = Array2::from_shape_fn((100, 3), |(i, j)| {
                ((i * 7 + j * 13) as f64 * 0.3).sin() * [0.01, 1.0, 100.0][j]
            });
            let y = Array2::ones((100, 1));
            let model =
                EpistemicNearestNeighbors::new(x.clone(), y, None, weighted, IndexDriver::Flat)
                    .unwrap();
            let query = x.slice(ndarray::s![..;3, ..]);
            let cache = SmallFlatNeighbors::new(&model, &query, 100).unwrap();
            for k in [1, 3, 30, 100] {
                let (distances, indices) = cache.neighbors(k).unwrap();
                let (_, mut expected_indices) = model
                    .backend_search(&query, (k + F64_OVERSAMPLE).min(100) as i32, false)
                    .unwrap();
                let mut expected_distances =
                    dist2s_for_neighbor_indices(&model, &query, &expected_indices);
                sort_neighbors_by_dist2(&mut expected_distances, &mut expected_indices);
                assert_eq!(
                    indices,
                    expected_indices.slice_axis(Axis(1), Slice::from(..k))
                );
                assert_eq!(
                    distances,
                    expected_distances.slice_axis(Axis(1), Slice::from(..k))
                );
            }
        }
    }

    #[test]
    fn partial_cache_matches_original_search_on_novel_queries() {
        for dimensions in [3, 16, 32] {
            let x = Array2::from_shape_fn((300, dimensions), |(i, j)| {
                ((i * 7 + j * 13) as f64 * 0.3).sin()
            });
            let query = x.slice(ndarray::s![..30, ..]).mapv(|v| v + 0.03);
            let model = EpistemicNearestNeighbors::new(
                x,
                Array2::ones((300, 1)),
                None,
                false,
                IndexDriver::Flat,
            )
            .unwrap();
            let cache = SmallFlatNeighbors::new(&model, &query.view(), 30).unwrap();
            assert!(cache.neighbors(100).is_none());
            assert!(cache.possible_predecessors.iter().any(|&count| count <= 35));
            for k in [1, 3, 10, 30] {
                let (distances, indices) = cache.neighbors(k).unwrap();
                let (_, mut expected_indices) = model
                    .backend_search(&query.view(), (k + F64_OVERSAMPLE) as i32, false)
                    .unwrap();
                let mut expected_distances =
                    dist2s_for_neighbor_indices(&model, &query.view(), &expected_indices);
                sort_neighbors_by_dist2(&mut expected_distances, &mut expected_indices);
                assert_eq!(
                    indices,
                    expected_indices.slice_axis(Axis(1), Slice::from(..k))
                );
                assert_eq!(
                    distances,
                    expected_distances.slice_axis(Axis(1), Slice::from(..k))
                );
            }
        }
    }

    #[test]
    fn float32_collision_boundary_requires_original_search() {
        let mut x = Array2::from_shape_fn((100, 1), |(i, _)| 1.0 + i as f64 * 1e-10);
        x[[0, 0]] = 0.0;
        let model = EpistemicNearestNeighbors::new(
            x,
            Array2::ones((100, 1)),
            None,
            false,
            IndexDriver::Flat,
        )
        .unwrap();
        let cache = SmallFlatNeighbors::new(&model, &array![[2.0]].view(), 100).unwrap();
        assert!(cache.neighbors(1).is_none());
        assert!(cache.neighbors(30).is_none());
        assert!(cache.neighbors(100).is_some());
        assert!(cache.neighbors(0).is_none());
    }

    #[test]
    fn oversized_invalid_and_overflow_prone_inputs_use_original_path() {
        for n in [0, 3, 301] {
            let model = EpistemicNearestNeighbors::new(
                Array2::zeros((n, 2)),
                Array2::ones((n, 1)),
                None,
                false,
                IndexDriver::Flat,
            )
            .unwrap();
            assert!(SmallFlatNeighbors::new(&model, &Array2::zeros((1, 3)).view(), 3).is_none());
            assert!(SmallFlatNeighbors::new(&model, &Array2::zeros((301, 2)).view(), 3).is_none());
            assert!(SmallFlatNeighbors::new(&model, &array![[1e30, 0.0]].view(), 3).is_none());
            if n != 3 {
                assert!(SmallFlatNeighbors::new(&model, &array![[0.0, 0.0]].view(), 3).is_none());
            }
        }
    }
}
