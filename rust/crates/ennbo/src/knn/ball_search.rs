use super::ball_tree::BallTreeBackend;
use std::cmp::Ordering;

struct TopK {
    best_d: Vec<f64>,
    best_id: Vec<usize>,
    filled: usize,
    tau: f64,
    k: usize,
}

impl TopK {
    fn new(k: usize) -> Self {
        Self {
            best_d: vec![f64::INFINITY; k],
            best_id: vec![0usize; k],
            filled: 0,
            tau: f64::INFINITY,
            k,
        }
    }

    #[inline]
    fn consider(&mut self, dist2: f64, id: usize) {
        let k = self.k;
        if self.filled < k {
            self.best_d[self.filled] = dist2;
            self.best_id[self.filled] = id;
            self.filled += 1;
            if self.filled == k {
                promote_worst(&mut self.best_d, &mut self.best_id, k);
                self.tau = self.best_d[k - 1];
            }
        } else if dist2 < self.tau {
            self.best_d[k - 1] = dist2;
            self.best_id[k - 1] = id;
            promote_worst(&mut self.best_d, &mut self.best_id, k);
            self.tau = self.best_d[k - 1];
        }
    }

    fn finish(self) -> Vec<(f64, usize)> {
        let mut out: Vec<(f64, usize)> = (0..self.filled)
            .map(|i| (self.best_d[i], self.best_id[i]))
            .collect();
        out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        while out.len() < self.k {
            out.push((f64::INFINITY, 0));
        }
        out
    }
}

impl BallTreeBackend {
    /// Contiguous exact top-k fallback (mid-N normally uses Faiss Flat).
    pub(crate) fn search_one_brute(&self, query: &[f64], k: usize) -> Vec<(f64, usize)> {
        if k == 0 || self.len() == 0 {
            return Vec::new();
        }
        let mut top = TopK::new(k);
        for id in 0..self.n {
            top.consider(Self::dist2(query, self.point(id)), id);
        }
        top.finish()
    }

    pub(crate) fn search_one_ball(&self, query: &[f64], k: usize) -> Vec<(f64, usize)> {
        if k == 0 || self.n == 0 {
            return Vec::new();
        }
        let mut top = TopK::new(k);
        let mut sqrt_tau = f64::INFINITY;
        if self.tree_n > 0 && !self.nodes.is_empty() {
            let mut stack = Vec::with_capacity(64);
            stack.push(self.root);
            while let Some(ni) = stack.pop() {
                let node = &self.nodes[ni];
                if node.leaf_idx == usize::MAX && node.left == usize::MAX {
                    continue;
                }
                let dc = if node.center.is_empty() {
                    0.0
                } else {
                    Self::dist2(query, &node.center)
                };
                if top.filled == k && !node.center.is_empty() {
                    let r = node.radius;
                    let thresh = top.tau + r * r + 2.0 * r * sqrt_tau;
                    if dc >= thresh {
                        continue;
                    }
                }
                if node.left == usize::MAX {
                    let dim = self.num_dim;
                    let start = node.leaf_pack_start * dim;
                    let nleaf = node.leaf_pack_len;
                    let leaf_idx = node.leaf_idx;
                    for j in 0..nleaf {
                        let off = start + j * dim;
                        let old_tau = top.tau;
                        top.consider(
                            Self::dist2(query, &self.leaf_pack[off..off + dim]),
                            self.leaves[leaf_idx][j],
                        );
                        if top.filled == k && top.tau != old_tau {
                            sqrt_tau = top.tau.sqrt();
                        }
                    }
                } else {
                    let left = node.left;
                    let right = node.right;
                    let left_d = if self.nodes[left].center.is_empty() {
                        0.0
                    } else {
                        Self::dist2(query, &self.nodes[left].center)
                    };
                    let right_d = if self.nodes[right].center.is_empty() {
                        0.0
                    } else {
                        Self::dist2(query, &self.nodes[right].center)
                    };
                    if left_d <= right_d {
                        stack.push(right);
                        stack.push(left);
                    } else {
                        stack.push(left);
                        stack.push(right);
                    }
                }
            }
        }
        for id in self.tree_n..self.n {
            top.consider(Self::dist2(query, self.point(id)), id);
        }
        top.finish()
    }

    pub(crate) fn search_one_aabb(&self, query: &[f64], k: usize) -> Vec<(f64, usize)> {
        if k == 0 || self.n == 0 {
            return Vec::new();
        }
        let mut top = TopK::new(k);
        if self.tree_n > 0 && !self.nodes.is_empty() {
            let mut stack = Vec::with_capacity(64);
            stack.push(self.root);
            while let Some(ni) = stack.pop() {
                let node = &self.nodes[ni];
                if top.filled == k {
                    let lb = Self::aabb_dist2(query, &node.bbox_min, &node.bbox_max);
                    if lb >= top.tau {
                        continue;
                    }
                }
                if node.left == usize::MAX {
                    let dim = self.num_dim;
                    let start = node.leaf_pack_start * dim;
                    let nleaf = node.leaf_pack_len;
                    let leaf_idx = node.leaf_idx;
                    for j in 0..nleaf {
                        let off = start + j * dim;
                        top.consider(
                            Self::dist2(query, &self.leaf_pack[off..off + dim]),
                            self.leaves[leaf_idx][j],
                        );
                    }
                } else {
                    let left = node.left;
                    let right = node.right;
                    let left_lb = Self::aabb_dist2(
                        query,
                        &self.nodes[left].bbox_min,
                        &self.nodes[left].bbox_max,
                    );
                    let right_lb = Self::aabb_dist2(
                        query,
                        &self.nodes[right].bbox_min,
                        &self.nodes[right].bbox_max,
                    );
                    if left_lb <= right_lb {
                        if top.filled < k || right_lb < top.tau {
                            stack.push(right);
                        }
                        if top.filled < k || left_lb < top.tau {
                            stack.push(left);
                        }
                    } else {
                        if top.filled < k || left_lb < top.tau {
                            stack.push(left);
                        }
                        if top.filled < k || right_lb < top.tau {
                            stack.push(right);
                        }
                    }
                }
            }
        }
        for id in self.tree_n..self.n {
            top.consider(Self::dist2(query, self.point(id)), id);
        }
        top.finish()
    }
}

#[inline]
fn promote_worst(best_d: &mut [f64], best_id: &mut [usize], k: usize) {
    let mut max_i = 0usize;
    for i in 1..k {
        if best_d[i] > best_d[max_i] {
            max_i = i;
        }
    }
    if max_i != k - 1 {
        best_d.swap(max_i, k - 1);
        best_id.swap(max_i, k - 1);
    }
}
