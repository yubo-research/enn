
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::distance::{bpann_row_to_f32, l2_sq_f32};
use crate::error::BpannError;
use crate::index::search::MmapSearchStore;
use crate::index::tree::Tree;
use crate::small_n_search::OrderedF32;

pub const TREE_SEARCH_LEAVES_PER_LN_N: f64 = 128.0;

pub fn leaf_budget(indexed_rows: usize) -> usize {
    let ln_n = (indexed_rows.max(1) as f64).ln();
    ((TREE_SEARCH_LEAVES_PER_LN_N * ln_n).ceil() as usize).max(1)
}

pub const TREE_SEARCH_RADIUS_WEIGHT: f32 = 0.1;

fn discounted(d2: f32, radius: f32, w: f32) -> f32 {
    let gap = (d2.sqrt() - w * radius).max(0.0);
    gap * gap
}

fn priority(d2: f32, radius: f32) -> f32 {
    discounted(d2, radius, TREE_SEARCH_RADIUS_WEIGHT)
}

fn lower_bound(d2: f32, radius: f32) -> f32 {
    discounted(d2, radius, 1.0)
}

struct TopK {
    k: usize,
    heap: BinaryHeap<(OrderedF32, u32)>,
}

impl TopK {
    fn offer(&mut self, row: u32, dist: f32) {
        if self.heap.len() < self.k {
            self.heap.push((OrderedF32(dist), row));
        } else if let Some(&(worst, worst_row)) = self.heap.peek() {
            if (OrderedF32(dist), row) < (worst, worst_row) {
                self.heap.pop();
                self.heap.push((OrderedF32(dist), row));
            }
        }
    }

    fn cannot_improve(&self, bound: f32) -> bool {
        self.heap.len() >= self.k && self.heap.peek().is_some_and(|&(worst, _)| bound * (1.0 - 1e-4) > worst.0)
    }

    fn into_sorted(self) -> Vec<(u32, f32)> {
        let mut out: Vec<(u32, f32)> = self.heap.into_iter().map(|(d, r)| (r, d.0)).collect();
        out.sort_by(|a, b| OrderedF32(a.1).cmp(&OrderedF32(b.1)).then(a.0.cmp(&b.0)));
        out
    }
}

pub fn search_tree(
    tree: &Tree,
    query: &[f32],
    k: usize,
    leaf_budget: usize,
    store: &MmapSearchStore<'_>,
) -> Result<Vec<(u32, f32)>, BpannError> {
    let mut top = TopK {
        k,
        heap: BinaryHeap::with_capacity(k + 1),
    };
    if k == 0 || tree.num_pages() == 0 {
        return Ok(Vec::new());
    }
    let s = &tree.store;
    let d = query.len().max(1);
    let mut frontier: BinaryHeap<Reverse<(OrderedF32, OrderedF32, u32)>> = BinaryHeap::new();
    frontier.push(Reverse((OrderedF32(0.0), OrderedF32(0.0), tree.header.root_page_id)));
    let mut leaves = 0usize;
    let mut buf = Vec::with_capacity(query.len());
    while let Some(Reverse((_, bound, page_id))) = frontier.pop() {
        if leaves >= leaf_budget && top.heap.len() >= k {
            break;
        }
        if top.cannot_improve(bound.0) {
            continue;
        }
        if !s.is_leaf(page_id) {
            for (&child, c) in s.ids(page_id).iter().zip(s.block(page_id).chunks_exact(d)) {
                let d2 = l2_sq_f32(query, c);
                let radius = s.radius(child);
                let bound = lower_bound(d2, radius);
                if !top.cannot_improve(bound) {
                    frontier.push(Reverse((OrderedF32(priority(d2, radius)), OrderedF32(bound), child)));
                }
            }
            continue;
        }
        if tree.rows_cached() {
            for (&row, x) in s.ids(page_id).iter().zip(s.block(page_id).chunks_exact(d)) {
                top.offer(row, l2_sq_f32(query, x));
            }
        } else {
            for &row in s.ids(page_id) {
                bpann_row_to_f32(store.train_x.mmap_row_slice(row as usize)?, store.scale_x, store.x_scale, &mut buf);
                top.offer(row, l2_sq_f32(query, &buf));
            }
        }
        leaves += 1;
    }
    Ok(top.into_sorted())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn leaf_budget_grows_like_ln_n() {
        assert_eq!(leaf_budget(0), 1);
        assert_eq!(leaf_budget(1_000_000), 1769);
        let ratio = leaf_budget(10_000_000) as f64 / leaf_budget(1_000_000) as f64;
        assert!((ratio - 10_000_000f64.ln() / 1_000_000f64.ln()).abs() < 1e-3);
    }

    #[test]
    fn priority_discounts_radius_and_never_goes_negative() {
        assert_eq!(priority(4.0, 0.0), 4.0);
        assert!((priority(4.0, 10.0) - 1.0).abs() < 1e-6);
        assert_eq!(priority(1.0, 100.0), 0.0);
        assert_eq!(lower_bound(4.0, 1.0), 1.0);
        assert_eq!(lower_bound(4.0, 3.0), 0.0);
    }
}
