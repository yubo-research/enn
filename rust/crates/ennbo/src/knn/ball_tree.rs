//! In-memory ball-tree KNN backend for [`crate::index::IndexDriver::FastMem`].

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;
use std::cmp::Ordering;

use super::pad_neighbor_cols_to_search_k;
use crate::index::IndexError;

const LEAF_SIZE: usize = 16;
pub(super) const AABB_MODE_MIN_N: usize = 5000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SearchMode {
    Ball,
    Aabb,
}

#[derive(Clone)]
pub(crate) struct BallNode {
    pub(crate) center: Vec<f64>,
    pub(crate) radius: f64,
    pub(crate) bbox_min: Vec<f64>,
    pub(crate) bbox_max: Vec<f64>,
    pub(crate) left: usize,
    pub(crate) right: usize,
    pub(crate) leaf_start: usize,
    pub(crate) leaf_end: usize,
}

pub(crate) struct BallTreeBackend {
    num_dim: usize,
    data: Vec<f64>,
    n: usize,
    pub(crate) nodes: Vec<BallNode>,
    pub(crate) leaf_ids: Vec<usize>,
    pub(crate) root: usize,
    dirty: bool,
    pub(super) search_mode: SearchMode,
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
            search_mode: SearchMode::Ball,
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
        let use_parallel =
            self.n >= PARALLEL_MIN_N && n_query > 1 && rayon::current_num_threads() > 1;
        if use_parallel {
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
            self.search_mode = SearchMode::Ball;
            return;
        }
        self.search_mode = if self.n >= AABB_MODE_MIN_N {
            SearchMode::Aabb
        } else {
            SearchMode::Ball
        };
        let mut ids: Vec<usize> = (0..self.n).collect();
        self.root = self.build_node(&mut ids);
    }

    pub(crate) fn point(&self, id: usize) -> &[f64] {
        let start = id * self.num_dim;
        &self.data[start..start + self.num_dim]
    }

    #[inline]
    pub(crate) fn dist2(a: &[f64], b: &[f64]) -> f64 {
        let mut s = 0.0_f64;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = x - y;
            s += d * d;
        }
        s
    }

    #[inline]
    pub(crate) fn aabb_dist2(query: &[f64], bbox_min: &[f64], bbox_max: &[f64]) -> f64 {
        let mut s = 0.0_f64;
        for i in 0..query.len() {
            let q = query[i];
            let v = if q < bbox_min[i] {
                bbox_min[i] - q
            } else if q > bbox_max[i] {
                q - bbox_max[i]
            } else {
                0.0
            };
            s += v * v;
        }
        s
    }

    fn build_node(&mut self, ids: &mut [usize]) -> usize {
        let aabb_mode = self.search_mode == SearchMode::Aabb;
        let (center, radius, bbox_min, bbox_max) = if aabb_mode {
            let (bbox_min, bbox_max) = self.bbox_of(ids);
            (Vec::new(), 0.0_f64, bbox_min, bbox_max)
        } else {
            let center = self.centroid(ids);
            let radius2 = ids
                .iter()
                .map(|&id| Self::dist2(&center, self.point(id)))
                .fold(0.0_f64, f64::max);
            (center, radius2.sqrt(), Vec::new(), Vec::new())
        };
        if ids.len() <= LEAF_SIZE {
            let leaf_start = self.leaf_ids.len();
            self.leaf_ids.extend_from_slice(ids);
            let leaf_end = self.leaf_ids.len();
            self.nodes.push(BallNode {
                center,
                radius,
                bbox_min,
                bbox_max,
                left: usize::MAX,
                right: usize::MAX,
                leaf_start,
                leaf_end,
            });
            return self.nodes.len() - 1;
        }
        let (mut left_ids, mut right_ids) = if aabb_mode {
            self.split_ids_median(ids)
        } else {
            self.split_ids_farthest(ids, &center)
        };
        let left = self.build_node(&mut left_ids);
        let right = self.build_node(&mut right_ids);
        self.nodes.push(BallNode {
            center,
            radius,
            bbox_min,
            bbox_max,
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

    fn bbox_of(&self, ids: &[usize]) -> (Vec<f64>, Vec<f64>) {
        let mut bmin = vec![f64::INFINITY; self.num_dim];
        let mut bmax = vec![f64::NEG_INFINITY; self.num_dim];
        for &id in ids {
            let p = self.point(id);
            for d in 0..self.num_dim {
                if p[d] < bmin[d] {
                    bmin[d] = p[d];
                }
                if p[d] > bmax[d] {
                    bmax[d] = p[d];
                }
            }
        }
        (bmin, bmax)
    }

    fn split_ids_median(&self, ids: &[usize]) -> (Vec<usize>, Vec<usize>) {
        let mut best_dim = 0usize;
        let mut best_range = -1.0_f64;
        for d in 0..self.num_dim {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for &id in ids {
                let v = self.point(id)[d];
                if v < lo {
                    lo = v;
                }
                if v > hi {
                    hi = v;
                }
            }
            let range = hi - lo;
            if range > best_range {
                best_range = range;
                best_dim = d;
            }
        }
        let mid = ids.len() / 2;
        if mid == 0 || mid >= ids.len() {
            return (ids[..mid].to_vec(), ids[mid..].to_vec());
        }
        let mut order = ids.to_vec();
        order.select_nth_unstable_by(mid, |&a, &b| {
            self.point(a)[best_dim]
                .partial_cmp(&self.point(b)[best_dim])
                .unwrap_or(Ordering::Equal)
        });
        let (left, right) = order.split_at(mid);
        if left.is_empty() || right.is_empty() {
            return (ids[..mid].to_vec(), ids[mid..].to_vec());
        }
        (left.to_vec(), right.to_vec())
    }

    fn split_ids_farthest(&self, ids: &[usize], center: &[f64]) -> (Vec<usize>, Vec<usize>) {
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
        let mut left = Vec::with_capacity(ids.len() / 2 + 1);
        let mut right = Vec::with_capacity(ids.len() / 2 + 1);
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
        match self.search_mode {
            SearchMode::Ball => self.search_one_ball(query, k),
            SearchMode::Aabb => self.search_one_aabb(query, k),
        }
    }
}

#[cfg(test)]
#[path = "ball_tree_tests.rs"]
mod ball_tree_tests;
