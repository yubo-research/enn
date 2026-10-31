
use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::distance::{bpann_row_to_f32, l2_sq_f32};
use crate::error::BpannError;
use crate::index::search::MmapSearchStore;
use crate::index::tree::Tree;
use crate::small_n_search::{OrderedF32, OrderedFloat};

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
    heap: BinaryHeap<(OrderedFloat<f64>, u32)>,
}

impl TopK {
    fn offer(&mut self, row: u32, dist: f64) {
        if self.heap.len() < self.k {
            self.heap.push((OrderedFloat(dist), row));
        } else if let Some(&(worst, worst_row)) = self.heap.peek() {
            if (OrderedFloat(dist), row) < (worst, worst_row) {
                self.heap.pop();
                self.heap.push((OrderedFloat(dist), row));
            }
        }
    }

    fn cannot_improve(&self, bound: f32) -> bool {
        let lower = (f64::from(bound) * (1.0 - 1e-4)
            - 64.0 * f32::MIN_POSITIVE as f64).max(0.0);
        self.heap.len() >= self.k && self.heap.peek().is_some_and(|&(worst, _)| lower > worst.0)
    }

    fn limit(&self) -> f64 {
        if self.heap.len() < self.k { f64::INFINITY }
        else { self.heap.peek().map_or(f64::INFINITY, |p| p.0.0) }
    }

    fn into_sorted(self) -> Vec<(u32, f64)> {
        let mut out: Vec<(u32, f64)> = self.heap.into_iter().map(|(d, r)| (r, d.0)).collect();
        out.sort_by(|a, b| OrderedFloat(a.1).cmp(&OrderedFloat(b.1)).then(a.0.cmp(&b.0)));
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
    let rows = search_tree_scored(tree, query, k, leaf_budget, |row, cached, _| {
        if let Some(x) = cached {
            return Ok(Some(f64::from(l2_sq_f32(query, x))));
        }
        let mut buf = Vec::with_capacity(query.len());
        bpann_row_to_f32(store.train_x.mmap_row_slice(row as usize)?, store.scale_x, store.x_scale, &mut buf);
        Ok(Some(f64::from(l2_sq_f32(query, &buf))))
    })?;
    Ok(rows.into_iter().map(|(row, dist)| (row, dist as f32)).collect())
}

pub fn search_tree_refined(
    tree: &Tree, query: &[f64], k: usize, store: &MmapSearchStore<'_>,
) -> Result<Vec<(u32, f64)>, BpannError> {
    search_tree_refined_with_extra(tree, query, k, store, 10)
}

pub fn search_tree_refined_with_extra(
    tree: &Tree, query: &[f64], k: usize, store: &MmapSearchStore<'_>, extra: usize,
) -> Result<Vec<(u32, f64)>, BpannError> {
    let pool = k.saturating_add(extra).min(tree.header.indexed_rows);
    let mut scaled = Vec::with_capacity(query.len());
    bpann_row_to_f32(query, store.scale_x, store.x_scale, &mut scaled);
    let candidates = search_tree(tree, &scaled, pool, leaf_budget(tree.header.indexed_rows), store)?;
    let boundary = candidates.last().map_or(0.0, |p| f64::from(p.1));
    let mut refined: Vec<(u32, f64)> = candidates.iter().map(|&(row, _)| {
        Ok((row, crate::distance::row_sq_l2(
            ndarray::ArrayView1::from(query),
            ndarray::ArrayView1::from(store.train_x.mmap_row_slice(row as usize)?),
            store.scale_x, ndarray::ArrayView1::from(store.x_scale),
        )))
    }).collect::<Result<_, BpannError>>()?;
    refined.sort_by(|a,b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    let all_rows = refined.len() == tree.header.indexed_rows;
    let certified = refined.get(k.saturating_sub(1)).is_some_and(|p| {
        refinement_lower_bound(tree, query, store, boundary) > p.1
    });
    if !all_rows && (!tree.rows_cached() || !certified) {
        return search_tree_f64(tree, query, k, store);
    }
    refined.truncate(k);
    Ok(refined)
}

fn refinement_lower_bound(tree: &Tree, query: &[f64], store: &MmapSearchStore<'_>, boundary: f64) -> f64 {
    let eta = 4.0 * tree.coordinate_relative_error
        + 16.0 * (query.len() + 1) as f64 * f32::EPSILON as f64;
    let norm: f64 = query.iter().enumerate().map(|(j, &q)| {
        let q = if store.scale_x { q / store.x_scale[j] } else { q };
        q * q
    }).sum();
    if eta >= 0.1 || !norm.is_finite() || !boundary.is_finite() {
        return 0.0;
    }
    let absolute = 64.0 * (query.len() as f64).sqrt()
        * (tree.coordinate_absolute_error + (f32::MIN_POSITIVE as f64).sqrt());
    let lower = boundary.sqrt() * (1.0 - eta) - eta * norm.sqrt() - absolute;
    lower.max(0.0).powi(2)
}

pub fn search_tree_f64(
    tree: &Tree,
    query: &[f64],
    k: usize,
    store: &MmapSearchStore<'_>,
) -> Result<Vec<(u32, f64)>, BpannError> {
    let mut scaled = Vec::with_capacity(query.len());
    bpann_row_to_f32(query, store.scale_x, store.x_scale, &mut scaled);
    let query_scaled: Vec<f64> = query.iter().enumerate().map(|(j, &q)| {
        if store.scale_x { q / store.x_scale[j] } else { q }
    }).collect();
    search_tree_scored(tree, &scaled, k, leaf_budget(tree.header.indexed_rows), |row, _, limit| {
        let point = store.train_x.mmap_row_slice(row as usize)?;
        let mut distance = 0.0;
        for (j, (&q, &x)) in query_scaled.iter().zip(point).enumerate() {
            let x = if store.scale_x { x / store.x_scale[j] } else { x };
            let delta = q - x;
            distance += delta * delta;
            if distance > limit {
                return Ok(None);
            }
        }
        Ok(Some(distance.max(0.0)))
    })
}

fn search_tree_scored(
    tree: &Tree,
    query: &[f32],
    k: usize,
    leaf_budget: usize,
    mut score: impl FnMut(u32, Option<&[f32]>, f64) -> Result<Option<f64>, BpannError>,
) -> Result<Vec<(u32, f64)>, BpannError> {
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
                if let Some(distance) = score(row, Some(x), top.limit())? {
                    top.offer(row, distance);
                }
            }
        } else {
            for &row in s.ids(page_id) {
                if let Some(distance) = score(row, None, top.limit())? {
                    top.offer(row, distance);
                }
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
