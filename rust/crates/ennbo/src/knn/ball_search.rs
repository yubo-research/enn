use super::ball_tree::BallTreeBackend;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

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

impl BallTreeBackend {
    /// Contiguous exact top-k fallback (mid-N normally uses Faiss Flat).
    pub(crate) fn search_one_brute(&self, query: &[f64], k: usize) -> Vec<(f64, usize)> {
        if k == 0 || self.len() == 0 {
            return Vec::new();
        }
        let n = self.len();
        let mut best_d = vec![f64::INFINITY; k];
        let mut best_id = vec![0usize; k];
        let mut filled = 0usize;
        let mut tau = f64::INFINITY;
        for id in 0..n {
            let dist2 = Self::dist2(query, self.point(id));
            if filled < k {
                best_d[filled] = dist2;
                best_id[filled] = id;
                filled += 1;
                if filled == k {
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
                    tau = best_d[k - 1];
                }
            } else if dist2 < tau {
                best_d[k - 1] = dist2;
                best_id[k - 1] = id;
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
                tau = best_d[k - 1];
            }
        }
        let mut out: Vec<(f64, usize)> = (0..filled).map(|i| (best_d[i], best_id[i])).collect();
        out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
        while out.len() < k {
            out.push((f64::INFINITY, 0));
        }
        out
    }

    pub(crate) fn search_one_ball(&self, query: &[f64], k: usize) -> Vec<(f64, usize)> {
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
        finish_heap(best, k)
    }

    pub(crate) fn search_one_aabb(&self, query: &[f64], k: usize) -> Vec<(f64, usize)> {
        let mut best: BinaryHeap<HeapItem> = BinaryHeap::with_capacity(k + 1);
        let mut tau = f64::INFINITY;
        let mut stack = Vec::with_capacity(64);
        stack.push(self.root);
        while let Some(ni) = stack.pop() {
            let node = &self.nodes[ni];
            if best.len() == k {
                let lb = Self::aabb_dist2(query, &node.bbox_min, &node.bbox_max);
                if lb >= tau {
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
                        }
                    } else if dist2 < tau {
                        best.pop();
                        best.push(HeapItem { dist2, id });
                        tau = best.peek().unwrap().dist2;
                    }
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
                    if best.len() < k || right_lb < tau {
                        stack.push(right);
                    }
                    if best.len() < k || left_lb < tau {
                        stack.push(left);
                    }
                } else {
                    if best.len() < k || left_lb < tau {
                        stack.push(left);
                    }
                    if best.len() < k || right_lb < tau {
                        stack.push(right);
                    }
                }
            }
        }
        finish_heap(best, k)
    }
}

fn finish_heap(best: BinaryHeap<HeapItem>, k: usize) -> Vec<(f64, usize)> {
    let mut out: Vec<(f64, usize)> = best.into_iter().map(|h| (h.dist2, h.id)).collect();
    out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(Ordering::Equal));
    while out.len() < k {
        out.push((f64::INFINITY, 0));
    }
    out
}
