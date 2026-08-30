use super::ball_tree::{BallNode, BallTreeBackend, SearchMode, AABB_MODE_MIN_N, LEAF_SIZE};
use std::cmp::Ordering;

impl BallTreeBackend {
    pub(super) fn build_tree_now(&mut self) {
        self.nodes.clear();
        self.leaf_ids.clear();
        self.root = 0;
        self.tree_pending = false;
        self.faiss_flat = None;
        if self.n == 0 {
            self.search_mode = SearchMode::Brute;
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
}
