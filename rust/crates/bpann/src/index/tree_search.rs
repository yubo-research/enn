//! Best-first search of the incremental tree.
//!
//! Pages are expanded in order of their centroid's distance to the query, less a
//! fraction of their radius ([`TREE_SEARCH_RADIUS_WEIGHT`]), and the
//! search stops after scoring [`leaf_budget`] leaves (once it holds `k` candidates).
//! The budget is proportional to `ln N`, so the work per query grows like `ln N`.
//! A page whose ball (centroid and radius) lies entirely farther than the current
//! `k`-th best row is skipped without using budget, since it cannot hold a better row.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::distance::{bpann_row_to_f32, l2_sq_f32};
use crate::error::BpannError;
use crate::index::build::BpannIndex;
use crate::index::page::Page;
use crate::index::search::MmapSearchStore;
use crate::index::tree::TreeCounts;
use crate::small_n_search::OrderedF32;

/// Leaves scored per query per unit of `ln N`.
pub const TREE_SEARCH_LEAVES_PER_LN_N: f64 = 128.0;

/// Leaves a query scores in a tree of `indexed_rows` rows: `ceil(128 ln N)`, at least 1.
pub fn leaf_budget(indexed_rows: usize) -> usize {
    let ln_n = (indexed_rows.max(1) as f64).ln();
    ((TREE_SEARCH_LEAVES_PER_LN_N * ln_n).ceil() as usize).max(1)
}

/// Pages are expanded in order of `(d - 0.1 r)^2`, with `d` the distance to the page's
/// centroid and `r` its radius. Pure centroid order (`0`) loses recall as the tree
/// grows (uniform 12-d, 3M rows: 0.87 of the true 10 neighbors) and ball-bound order
/// (`1`) spends the budget on large, far pages; `0.1` finds 0.998 at the same budget.
pub const TREE_SEARCH_RADIUS_WEIGHT: f32 = 0.1;

/// Squared `max(0, d - w r)` for a page whose centroid is at squared distance `d2`.
fn discounted(d2: f32, radius: f32, w: f32) -> f32 {
    let gap = (d2.sqrt() - w * radius).max(0.0);
    gap * gap
}

fn priority(d2: f32, radius: f32) -> f32 {
    discounted(d2, radius, TREE_SEARCH_RADIUS_WEIGHT)
}

/// Squared lower bound on the distance from a query to any row in a page whose
/// centroid is at squared distance `d2` and whose radius is `radius`.
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

    /// Whether no row at squared distance >= `bound` can enter the top `k`. The
    /// small slack keeps `f32` rounding in radii from pruning a true neighbor.
    fn cannot_improve(&self, bound: f32) -> bool {
        self.heap.len() >= self.k && self.heap.peek().is_some_and(|&(worst, _)| bound * (1.0 - 1e-4) > worst.0)
    }

    fn into_sorted(self) -> Vec<(u32, f32)> {
        let mut out: Vec<(u32, f32)> = self.heap.into_iter().map(|(d, r)| (r, d.0)).collect();
        out.sort_by(|a, b| OrderedF32(a.1).cmp(&OrderedF32(b.1)).then(a.0.cmp(&b.0)));
        out
    }
}

/// Approximate `k` nearest rows to `query` (scaled coordinates), ascending by distance.
pub fn search_tree(
    index: &BpannIndex,
    counts: &TreeCounts,
    query: &[f32],
    k: usize,
    leaf_budget: usize,
    store: &MmapSearchStore<'_>,
) -> Result<Vec<(u32, f32)>, BpannError> {
    let mut top = TopK {
        k,
        heap: BinaryHeap::with_capacity(k + 1),
    };
    if k == 0 || index.pages.is_empty() {
        return Ok(Vec::new());
    }
    let mut frontier: BinaryHeap<Reverse<(OrderedF32, OrderedF32, u32)>> = BinaryHeap::new();
    frontier.push(Reverse((OrderedF32(0.0), OrderedF32(0.0), index.header.root_page_id)));
    let mut leaves = 0usize;
    let mut buf = Vec::with_capacity(query.len());
    while let Some(Reverse((_, bound, page_id))) = frontier.pop() {
        if leaves >= leaf_budget && top.heap.len() >= k {
            break;
        }
        if top.cannot_improve(bound.0) {
            continue;
        }
        match index.page_by_id(page_id) {
            Some(Page::Internal {
                centroids,
                child_page_ids,
                ..
            }) => {
                let block = counts.block(page_id);
                let flat = block.len() == child_page_ids.len() * query.len();
                for (slot, &child) in child_page_ids.iter().enumerate() {
                    let c = if flat {
                        &block[slot * query.len()..(slot + 1) * query.len()]
                    } else {
                        centroids[slot].as_slice()
                    };
                    let d2 = l2_sq_f32(query, c);
                    let radius = counts.radius(child);
                    let bound = lower_bound(d2, radius);
                    if !top.cannot_improve(bound) {
                        frontier.push(Reverse((OrderedF32(priority(d2, radius)), OrderedF32(bound), child)));
                    }
                }
            }
            Some(Page::Leaf { row_ids, .. }) => {
                let block = counts.block(page_id);
                if block.len() == row_ids.len() * query.len() {
                    for (&row, x) in row_ids.iter().zip(block.chunks_exact(query.len().max(1))) {
                        top.offer(row, l2_sq_f32(query, x));
                    }
                } else {
                    for &row in row_ids {
                        bpann_row_to_f32(store.train_x.mmap_row_slice(row as usize)?, store.scale_x, store.x_scale, &mut buf);
                        top.offer(row, l2_sq_f32(query, &buf));
                    }
                }
                leaves += 1;
            }
            None => {}
        }
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
