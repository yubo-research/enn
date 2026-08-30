//! In-memory ball-tree KNN backend for [`crate::index::IndexDriver::FastMem`].

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

use super::pad_neighbor_cols_to_search_k;
use crate::index::IndexError;

const LEAF_SIZE: usize = 16;

#[derive(Clone)]
struct BallNode {
    center: Vec<f64>,
    /// Ball radius (not squared). Stored so search can prune without per-node `sqrt`.
    radius: f64,
    left: usize,
    right: usize,
    leaf_start: usize,
    leaf_end: usize,
}

/// Exact in-memory ball tree over L2 distance.
pub(crate) struct BallTreeBackend {
    num_dim: usize,
    data: Vec<f64>,
    n: usize,
    nodes: Vec<BallNode>,
    leaf_ids: Vec<usize>,
    root: usize,
    dirty: bool,
}

impl BallTreeBackend {
    pub(crate) fn new(num_dim: usize, train_scaled: &ArrayView2<f64>) -> Result<Self, IndexError> {
        let mut backend = Self {
            num_dim,
            data: Vec::new(),
            n: 0,
            nodes: Vec::new(),
            leaf_ids: Vec::new(),
            root: 0,
            dirty: true,
        };
        backend.rebuild(train_scaled)?;
        Ok(backend)
    }

    pub(crate) fn len(&self) -> usize {
        self.n
    }

    pub(crate) fn memory_usage_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f64>()
            + self.leaf_ids.len() * std::mem::size_of::<usize>()
            + self.nodes.capacity() * std::mem::size_of::<BallNode>()
    }

    pub(crate) fn rebuild(&mut self, train_scaled: &ArrayView2<f64>) -> Result<(), IndexError> {
        if train_scaled.ncols() != self.num_dim {
            return Err(IndexError::InvalidShape {
                expected: self.num_dim,
                got: train_scaled.ncols(),
            });
        }
        self.n = train_scaled.nrows();
        self.data.clear();
        self.data.reserve(self.n * self.num_dim);
        for row in train_scaled.rows() {
            self.data.extend(row.iter().copied());
        }
        self.rebuild_tree();
        Ok(())
    }

    pub(crate) fn add(
        &mut self,
        rows_scaled: &ArrayView2<f64>,
        _start_key: u64,
    ) -> Result<(), IndexError> {
        if rows_scaled.ncols() != self.num_dim {
            return Err(IndexError::InvalidShape {
                expected: self.num_dim,
                got: rows_scaled.ncols(),
            });
        }
        if rows_scaled.nrows() == 0 {
            return Ok(());
        }
        self.data.reserve(rows_scaled.nrows() * self.num_dim);
        for row in rows_scaled.rows() {
            self.data.extend(row.iter().copied());
        }
        self.n += rows_scaled.nrows();
        self.dirty = true;
        Ok(())
    }

    pub(crate) fn search(
        &mut self,
        queries_scaled: &ArrayView2<f64>,
        k_eff: usize,
        search_k: usize,
    ) -> Result<(Array2<f64>, Array2<i64>), IndexError> {
        if queries_scaled.ncols() != self.num_dim {
            return Err(IndexError::InvalidShape {
                expected: self.num_dim,
                got: queries_scaled.ncols(),
            });
        }
        if self.dirty {
            self.rebuild_tree();
        }
        let n_query = queries_scaled.nrows();
        if self.n == 0 || k_eff == 0 {
            return Ok((
                Array2::from_elem((n_query, search_k), f64::INFINITY),
                Array2::zeros((n_query, search_k)),
            ));
        }
        let mut dist2s = Array2::from_elem((n_query, k_eff), f64::INFINITY);
        let mut indices = Array2::zeros((n_query, k_eff));
        const PARALLEL_MIN_N: usize = 128;
        if self.n >= PARALLEL_MIN_N && n_query > 1 {
            let hits: Vec<Vec<(f64, usize)>> = (0..n_query)
                .into_par_iter()
                .map(|qi| {
                    let q = queries_scaled.row(qi);
                    self.search_one(q.as_slice().unwrap(), k_eff)
                })
                .collect();
            for (qi, row_hits) in hits.into_iter().enumerate() {
                for (j, (dist2, id)) in row_hits.into_iter().enumerate() {
                    dist2s[[qi, j]] = dist2;
                    indices[[qi, j]] = id as i64;
                }
            }
        } else {
            for qi in 0..n_query {
                let q = queries_scaled.row(qi);
                let hits = self.search_one(q.as_slice().unwrap(), k_eff);
                for (j, (dist2, id)) in hits.into_iter().enumerate() {
                    dist2s[[qi, j]] = dist2;
                    indices[[qi, j]] = id as i64;
                }
            }
        }
        Ok(pad_neighbor_cols_to_search_k(dist2s, indices, search_k))
    }

    fn rebuild_tree(&mut self) {
        self.nodes.clear();
        self.leaf_ids.clear();
        self.dirty = false;
        self.root = 0;
        if self.n == 0 {
            return;
        }
        let mut ids: Vec<usize> = (0..self.n).collect();
        self.root = self.build_node(&mut ids);
    }

    fn point(&self, id: usize) -> &[f64] {
        let start = id * self.num_dim;
        &self.data[start..start + self.num_dim]
    }

    fn dist2(a: &[f64], b: &[f64]) -> f64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let d = x - y;
                d * d
            })
            .sum()
    }

    fn build_node(&mut self, ids: &mut [usize]) -> usize {
        let center = self.centroid(ids);
        let radius2 = ids
            .iter()
            .map(|&id| Self::dist2(&center, self.point(id)))
            .fold(0.0_f64, f64::max);
        let radius = radius2.sqrt();
        if ids.len() <= LEAF_SIZE {
            let leaf_start = self.leaf_ids.len();
            self.leaf_ids.extend_from_slice(ids);
            let leaf_end = self.leaf_ids.len();
            self.nodes.push(BallNode {
                center,
                radius,
                left: usize::MAX,
                right: usize::MAX,
                leaf_start,
                leaf_end,
            });
            return self.nodes.len() - 1;
        }
        let (mut left_ids, mut right_ids) = self.split_ids(ids, &center);
        let left = self.build_node(&mut left_ids);
        let right = self.build_node(&mut right_ids);
        self.nodes.push(BallNode {
            center,
            radius,
            left,
            right,
            leaf_start: 0,
            leaf_end: 0,
        });
        self.nodes.len() - 1
    }

    fn centroid(&self, ids: &[usize]) -> Vec<f64> {
        let mut c = vec![0.0_f64; self.num_dim];
        for &id in ids {
            let p = self.point(id);
            for d in 0..self.num_dim {
                c[d] += p[d];
            }
        }
        let inv = 1.0 / ids.len() as f64;
        for v in &mut c {
            *v *= inv;
        }
        c
    }

    fn split_ids(&self, ids: &[usize], center: &[f64]) -> (Vec<usize>, Vec<usize>) {
        let mut p1 = ids[0];
        let mut best = -1.0_f64;
        for &id in ids {
            let d = Self::dist2(center, self.point(id));
            if d > best {
                best = d;
                p1 = id;
            }
        }
        let mut p2 = ids[0];
        best = -1.0;
        for &id in ids {
            let d = Self::dist2(self.point(p1), self.point(id));
            if d > best {
                best = d;
                p2 = id;
            }
        }
        let mut left = Vec::new();
        let mut right = Vec::new();
        for &id in ids {
            let d1 = Self::dist2(self.point(p1), self.point(id));
            let d2 = Self::dist2(self.point(p2), self.point(id));
            if d1 <= d2 {
                left.push(id);
            } else {
                right.push(id);
            }
        }
        if left.is_empty() || right.is_empty() {
            let mid = ids.len() / 2;
            return (ids[..mid].to_vec(), ids[mid..].to_vec());
        }
        (left, right)
    }

    fn search_one(&self, query: &[f64], k: usize) -> Vec<(f64, usize)> {
        #[derive(PartialEq)]
        struct HeapItem {
            dist2: f64,
            id: usize,
        }
        impl Eq for HeapItem {}
        impl PartialOrd for HeapItem {
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                Some(self.cmp(other))
            }
        }
        impl Ord for HeapItem {
            fn cmp(&self, other: &Self) -> Ordering {
                self.dist2
                    .partial_cmp(&other.dist2)
                    .unwrap_or(Ordering::Equal)
                    .then_with(|| self.id.cmp(&other.id))
            }
        }

        let mut best: BinaryHeap<HeapItem> = BinaryHeap::with_capacity(k + 1);
        let mut tau = f64::INFINITY;
        let mut sqrt_tau = f64::INFINITY;
        let mut stack = Vec::with_capacity(64);
        stack.push(self.root);
        while let Some(ni) = stack.pop() {
            let node = &self.nodes[ni];
            let dc = Self::dist2(query, &node.center);
            if best.len() == k {
                let r = node.radius;
                let thresh = tau + r * r + 2.0 * r * sqrt_tau;
                if dc >= thresh {
                    continue;
                }
            }
            if node.left == usize::MAX {
                for &id in &self.leaf_ids[node.leaf_start..node.leaf_end] {
                    let dist2 = Self::dist2(query, self.point(id));
                    if best.len() < k {
                        best.push(HeapItem { dist2, id });
                        if best.len() == k {
                            tau = best.peek().unwrap().dist2;
                            sqrt_tau = tau.sqrt();
                        }
                    } else if dist2 < tau {
                        best.pop();
                        best.push(HeapItem { dist2, id });
                        tau = best.peek().unwrap().dist2;
                        sqrt_tau = tau.sqrt();
                    }
                }
            } else {
                let left = node.left;
                let right = node.right;
                let left_d = Self::dist2(query, &self.nodes[left].center);
                let right_d = Self::dist2(query, &self.nodes[right].center);
                if left_d <= right_d {
                    stack.push(right);
                    stack.push(left);
                } else {
                    stack.push(left);
                    stack.push(right);
                }
            }
        }
        let mut out: Vec<(f64, usize)> = best.into_iter().map(|h| (h.dist2, h.id)).collect();
        out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        while out.len() < k {
            out.push((f64::INFINITY, 0));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    #[test]
    fn ball_tree_search_matches_bruteforce_nn() {
        let train = array![
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
        ];
        let mut backend = BallTreeBackend::new(2, &train.view()).unwrap();
        assert_eq!(backend.len(), 5);
        let (d, i) = backend
            .search(&array![[0.1, 0.1]].view(), 2, 2)
            .unwrap();
        assert_eq!(i[[0, 0]], 0);
        assert!(d[[0, 0]] < d[[0, 1]]);
    }

    #[test]
    fn ball_tree_add_rebuilds_lazily() {
        let train = array![[0.0, 0.0]];
        let mut backend = BallTreeBackend::new(2, &train.view()).unwrap();
        backend.add(&array![[1.0, 0.0]].view(), 1).unwrap();
        assert_eq!(backend.len(), 2);
        let (_d, i) = backend
            .search(&array![[0.9, 0.0]].view(), 1, 1)
            .unwrap();
        assert_eq!(i[[0, 0]], 1);
        assert!(backend.memory_usage_bytes() > 0);
        backend.rebuild(&train.view()).unwrap();
        assert_eq!(backend.len(), 1);
    }

    #[test]
    fn ball_tree_empty_and_zero_k() {
        let empty = Array2::<f64>::zeros((0, 2));
        let mut backend = BallTreeBackend::new(2, &empty.view()).unwrap();
        assert_eq!(backend.len(), 0);
        let (d, i) = backend
            .search(&array![[0.0, 0.0]].view(), 0, 3)
            .unwrap();
        assert_eq!(d.ncols(), 3);
        assert_eq!(i.ncols(), 3);
        assert!(d.iter().all(|v| v.is_infinite()));
        backend.add(&Array2::<f64>::zeros((0, 2)).view(), 0).unwrap();
        assert_eq!(backend.len(), 0);
    }

    #[test]
    fn ball_tree_shape_errors() {
        let train = array![[0.0, 0.0]];
        let mut backend = BallTreeBackend::new(2, &train.view()).unwrap();
        let bad = array![[0.0, 0.0, 0.0]];
        assert!(matches!(
            backend.rebuild(&bad.view()),
            Err(IndexError::InvalidShape { expected: 2, got: 3 })
        ));
        assert!(matches!(
            backend.add(&bad.view(), 1),
            Err(IndexError::InvalidShape { expected: 2, got: 3 })
        ));
        assert!(matches!(
            backend.search(&bad.view(), 1, 1),
            Err(IndexError::InvalidShape { expected: 2, got: 3 })
        ));
    }

    #[test]
    fn ball_tree_identical_points_and_pad() {
        // Force degenerate split path via identical coordinates.
        let train = Array2::from_elem((20, 2), 1.0);
        let mut backend = BallTreeBackend::new(2, &train.view()).unwrap();
        assert_eq!(backend.len(), 20);
        let (d, i) = backend
            .search(&array![[1.0, 1.0]].view(), 3, 5)
            .unwrap();
        assert_eq!(d.ncols(), 5);
        assert_eq!(i.ncols(), 5);
        assert!(d[[0, 0]] < 1e-12);
        assert!(d[[0, 4]].is_infinite() || d[[0, 4]] >= d[[0, 2]]);
    }

    #[test]
    fn ball_tree_k_equals_n_and_multi_query() {
        let train = array![
            [0.0, 0.0],
            [2.0, 0.0],
            [0.0, 2.0],
            [2.0, 2.0],
        ];
        let mut backend = BallTreeBackend::new(2, &train.view()).unwrap();
        let queries = array![[0.1, 0.1], [1.9, 1.9]];
        let (d, i) = backend.search(&queries.view(), 4, 4).unwrap();
        assert_eq!(i[[0, 0]], 0);
        assert_eq!(i[[1, 0]], 3);
        assert!(d[[0, 0]] <= d[[0, 3]]);
        assert!(d[[1, 0]] <= d[[1, 3]]);
    }
}
