use super::ball_tree::{BallNode, BallTreeBackend, SearchMode, AABB_MODE_MIN_N, LEAF_SIZE};
use std::cmp::Ordering;

impl BallTreeBackend {
    pub(super) fn build_tree_now(&mut self) {
        self.nodes.clear();
        self.leaves.clear();
        self.leaf_ids.clear();
        self.leaf_pack.clear();
        self.root = 0;
        self.tree_pending = false;
        self.faiss_flat = None;
        self.tree_n = 0;
        if self.n == 0 {
            self.search_mode = SearchMode::Brute;
            return;
        }
        self.search_mode = if self.n >= AABB_MODE_MIN_N {
            SearchMode::Aabb
        } else {
            SearchMode::Ball
        };
        let est_nodes = (self.n / LEAF_SIZE.max(1)).saturating_mul(2).saturating_add(64);
        self.nodes.reserve(est_nodes);
        self.leaves.reserve(est_nodes / 2 + 1);
        let mut ids: Vec<usize> = (0..self.n).collect();
        self.root = self.build_node(&mut ids);
        self.tree_n = self.n;
    }

    fn build_node(&mut self, ids: &mut [usize]) -> usize {
        if ids.len() <= LEAF_SIZE {
            return self.push_leaf_for_ids(ids);
        }
        if self.search_mode == SearchMode::Aabb {
            self.build_aabb_inner(ids)
        } else {
            self.build_ball_inner(ids)
        }
    }

    fn push_leaf_for_ids(&mut self, ids: &[usize]) -> usize {
        if self.search_mode == SearchMode::Aabb {
            let (bbox_min, bbox_max) = self.bbox_of(ids);
            self.push_leaf(Vec::new(), 0.0, bbox_min, bbox_max, ids)
        } else {
            let center = self.centroid(ids);
            let radius2 = ids
                .iter()
                .map(|&id| Self::dist2(&center, self.point(id)))
                .fold(0.0_f64, f64::max);
            self.push_leaf(center, radius2.sqrt(), Vec::new(), Vec::new(), ids)
        }
    }

    fn build_aabb_inner(&mut self, ids: &mut [usize]) -> usize {
        let (_sd, _sv, mid) = self.partition_median_with_plane(ids);
        if mid == 0 || mid >= ids.len() {
            return self.push_leaf_for_ids(ids);
        }
        let (left_ids, right_ids) = ids.split_at_mut(mid);
        let left = self.build_node(left_ids);
        let right = self.build_node(right_ids);
        let (bbox_min, bbox_max) = self.union_bbox(left, right);
        self.push_inner(Vec::new(), 0.0, bbox_min, bbox_max, left, right)
    }

    fn build_ball_inner(&mut self, ids: &mut [usize]) -> usize {
        let center = self.centroid(ids);
        let radius2 = ids
            .iter()
            .map(|&id| Self::dist2(&center, self.point(id)))
            .fold(0.0_f64, f64::max);
        let radius = radius2.sqrt();
        let mid = self.partition_farthest(ids, &center);
        if mid == 0 || mid >= ids.len() {
            return self.push_leaf(center, radius, Vec::new(), Vec::new(), ids);
        }
        let (left_ids, right_ids) = ids.split_at_mut(mid);
        let left = self.build_node(left_ids);
        let right = self.build_node(right_ids);
        self.push_inner(center, radius, Vec::new(), Vec::new(), left, right)
    }

    fn union_bbox(&self, left: usize, right: usize) -> (Vec<f64>, Vec<f64>) {
        let l = &self.nodes[left];
        let r = &self.nodes[right];
        let mut bmin = l.bbox_min.clone();
        let mut bmax = l.bbox_max.clone();
        for (d, (rmin, rmax)) in r.bbox_min.iter().zip(r.bbox_max.iter()).enumerate() {
            if *rmin < bmin[d] {
                bmin[d] = *rmin;
            }
            if *rmax > bmax[d] {
                bmax[d] = *rmax;
            }
        }
        (bmin, bmax)
    }

    fn push_inner(
        &mut self,
        center: Vec<f64>,
        radius: f64,
        bbox_min: Vec<f64>,
        bbox_max: Vec<f64>,
        left: usize,
        right: usize,
    ) -> usize {
        self.nodes.push(BallNode {
            center,
            radius,
            bbox_min,
            bbox_max,
            left,
            right,
            leaf_idx: usize::MAX,
            leaf_pack_start: 0,
            leaf_pack_len: 0,
        });
        self.nodes.len() - 1
    }

    fn push_leaf(
        &mut self,
        center: Vec<f64>,
        radius: f64,
        bbox_min: Vec<f64>,
        bbox_max: Vec<f64>,
        ids: &[usize],
    ) -> usize {
        let leaf_idx = self.leaves.len();
        self.leaves.push(ids.to_vec());
        self.leaf_ids.extend_from_slice(ids);
        let pack_start = self.leaf_pack.len() / self.num_dim.max(1);
        for &id in ids {
            let p = self.point(id).to_vec();
            self.leaf_pack.extend_from_slice(&p);
        }
        self.nodes.push(BallNode {
            center,
            radius,
            bbox_min,
            bbox_max,
            left: usize::MAX,
            right: usize::MAX,
            leaf_idx,
            leaf_pack_start: pack_start,
            leaf_pack_len: ids.len(),
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

    fn partition_median_with_plane(&self, ids: &mut [usize]) -> (usize, f64, usize) {
        let mut best_dim = 0usize;
        let mut best_range = -1.0_f64;
        let mut best_lo = 0.0_f64;
        let mut best_hi = 0.0_f64;
        for d in 0..self.num_dim {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for &id in ids.iter() {
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
                best_lo = lo;
                best_hi = hi;
            }
        }
        if ids.len() < 2 || best_range <= 0.0 {
            return (best_dim, 0.0, ids.len() / 2);
        }
        let mid_val = 0.5 * (best_lo + best_hi);
        let mut lo = 0usize;
        let mut hi = ids.len();
        while lo < hi {
            if self.point(ids[lo])[best_dim] <= mid_val {
                lo += 1;
            } else {
                hi -= 1;
                ids.swap(lo, hi);
            }
        }
        if lo == 0 || lo == ids.len() {
            let mid = ids.len() / 2;
            ids.select_nth_unstable_by(mid, |&a, &b| {
                self.point(a)[best_dim]
                    .partial_cmp(&self.point(b)[best_dim])
                    .unwrap_or(Ordering::Equal)
            });
            let split_val = self.point(ids[mid])[best_dim];
            return (best_dim, split_val, mid);
        }
        (best_dim, mid_val, lo)
    }

    fn partition_farthest(&self, ids: &mut [usize], center: &[f64]) -> usize {
        let mut p1 = ids[0];
        let mut best = -1.0_f64;
        for &id in ids.iter() {
            let d = Self::dist2(center, self.point(id));
            if d > best {
                best = d;
                p1 = id;
            }
        }
        let mut p2 = ids[0];
        best = -1.0;
        for &id in ids.iter() {
            let d = Self::dist2(self.point(p1), self.point(id));
            if d > best {
                best = d;
                p2 = id;
            }
        }
        let mut lo = 0usize;
        let mut hi = ids.len();
        while lo < hi {
            let d1 = Self::dist2(self.point(p1), self.point(ids[lo]));
            let d2 = Self::dist2(self.point(p2), self.point(ids[lo]));
            if d1 <= d2 {
                lo += 1;
            } else {
                hi -= 1;
                ids.swap(lo, hi);
            }
        }
        if lo == 0 || lo == ids.len() {
            return ids.len() / 2;
        }
        lo
    }
}
