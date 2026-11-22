use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::distance::{bpann_row_to_f32, l2_sq_f32};
use crate::error::BpannError;
use crate::index::search::MmapSearchStore;
use crate::index::tree::Tree;
use crate::ordered::{OrderedF32, OrderedFloat};

pub const TREE_SEARCH_LEAVES_PER_LN_N: f64 = 10.0;

pub const TREE_SEARCH_TIE_LEAVES_PER_LN_N: f64 = 100.0;

static LEAF_COEFFICIENT_BITS: AtomicU64 = AtomicU64::new(0);

pub fn leaf_coefficient() -> f64 {
    let bits = LEAF_COEFFICIENT_BITS.load(Ordering::Relaxed);
    if bits == 0 {
        TREE_SEARCH_LEAVES_PER_LN_N
    } else {
        f64::from_bits(bits)
    }
}

pub fn set_leaf_coefficient(value: f64) {
    let bits = if value == TREE_SEARCH_LEAVES_PER_LN_N {
        0
    } else {
        value.to_bits()
    };
    LEAF_COEFFICIENT_BITS.store(bits, Ordering::Relaxed);
}

static EXACT_FALLBACK: AtomicU64 = AtomicU64::new(0);

pub fn exact_fallback() -> bool {
    EXACT_FALLBACK.load(Ordering::Relaxed) != 0
}

pub fn set_exact_fallback(value: bool) {
    EXACT_FALLBACK.store(u64::from(value), Ordering::Relaxed);
}

struct WalkMeter {
    leaves: AtomicU64,
    walks: AtomicU64,
    budget_stops: AtomicU64,
}

impl WalkMeter {
    const fn new() -> Self {
        Self {
            leaves: AtomicU64::new(0),
            walks: AtomicU64::new(0),
            budget_stops: AtomicU64::new(0),
        }
    }

    fn record(&self, leaves: usize, stopped_on_budget: bool) {
        self.walks.fetch_add(1, Ordering::Relaxed);
        self.leaves.fetch_add(leaves as u64, Ordering::Relaxed);
        if stopped_on_budget {
            self.budget_stops.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn take(&self) -> (u64, u64, u64) {
        (
            self.leaves.swap(0, Ordering::Relaxed),
            self.walks.swap(0, Ordering::Relaxed),
            self.budget_stops.swap(0, Ordering::Relaxed),
        )
    }
}

static F32_WALKS: WalkMeter = WalkMeter::new();
static F64_WALKS: WalkMeter = WalkMeter::new();
static CERTIFIED_RETURNS: AtomicU64 = AtomicU64::new(0);
static FALLBACK_RETURNS: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WalkTotals {
    pub f32_leaves: u64,
    pub f32_walks: u64,
    pub f32_budget_stops: u64,
    pub f64_leaves: u64,
    pub f64_walks: u64,
    pub f64_budget_stops: u64,
    pub certified: u64,
    pub fallbacks: u64,
}

pub fn take_walk_totals() -> WalkTotals {
    let (f32_leaves, f32_walks, f32_budget_stops) = F32_WALKS.take();
    let (f64_leaves, f64_walks, f64_budget_stops) = F64_WALKS.take();
    WalkTotals {
        f32_leaves,
        f32_walks,
        f32_budget_stops,
        f64_leaves,
        f64_walks,
        f64_budget_stops,
        certified: CERTIFIED_RETURNS.swap(0, Ordering::Relaxed),
        fallbacks: FALLBACK_RETURNS.swap(0, Ordering::Relaxed),
    }
}

pub fn leaf_budget(indexed_rows: usize) -> usize {
    ln_budget(leaf_coefficient(), indexed_rows)
}

pub fn tie_leaf_budget(indexed_rows: usize) -> usize {
    ln_budget(TREE_SEARCH_TIE_LEAVES_PER_LN_N, indexed_rows)
}

fn ln_budget(coefficient: f64, indexed_rows: usize) -> usize {
    let ln_n = (indexed_rows.max(1) as f64).ln();
    ((coefficient * ln_n).ceil() as usize).max(1)
}

const MAX_FRONTIER_PAGES: usize = 4096;

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
        let lower = (f64::from(bound) * (1.0 - 1e-4) - 64.0 * f32::MIN_POSITIVE as f64).max(0.0);
        self.heap.len() >= self.k && self.heap.peek().is_some_and(|&(worst, _)| lower > worst.0)
    }

    fn limit(&self) -> f64 {
        if self.heap.len() < self.k {
            f64::INFINITY
        } else {
            self.heap.peek().map_or(f64::INFINITY, |p| p.0 .0)
        }
    }

    fn into_sorted(self) -> Vec<(u32, f64)> {
        let mut out: Vec<(u32, f64)> = self.heap.into_iter().map(|(d, r)| (r, d.0)).collect();
        out.sort_by(|a, b| {
            OrderedFloat(a.1)
                .cmp(&OrderedFloat(b.1))
                .then(a.0.cmp(&b.0))
        });
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
    let rows = search_tree_scored(tree, query, k, leaf_budget, &F32_WALKS, |row, cached, _| {
        if let Some(x) = cached {
            return Ok(Some(f64::from(l2_sq_f32(query, x))));
        }
        let mut buf = Vec::with_capacity(query.len());
        bpann_row_to_f32(
            store.train_x.mmap_row_slice(row as usize)?,
            store.scale_x,
            store.x_scale,
            &mut buf,
        );
        Ok(Some(f64::from(l2_sq_f32(query, &buf))))
    })?;
    Ok(rows
        .into_iter()
        .map(|(row, dist)| (row, dist as f32))
        .collect())
}

pub fn search_tree_refined(
    tree: &Tree,
    query: &[f64],
    k: usize,
    store: &MmapSearchStore<'_>,
) -> Result<Vec<(u32, f64)>, BpannError> {
    search_tree_refined_with_extra(tree, query, k, store, 10)
}

pub fn search_tree_refined_with_extra(
    tree: &Tree,
    query: &[f64],
    k: usize,
    store: &MmapSearchStore<'_>,
    extra: usize,
) -> Result<Vec<(u32, f64)>, BpannError> {
    if exact_fallback() && !tree.rows_cached() {
        FALLBACK_RETURNS.fetch_add(1, Ordering::Relaxed);
        return search_tree_f64(tree, query, k, store);
    }
    let pool = k.saturating_add(extra).min(tree.header.indexed_rows);
    let mut scaled = Vec::with_capacity(query.len());
    bpann_row_to_f32(query, store.scale_x, store.x_scale, &mut scaled);
    let budget = if exact_fallback() {
        usize::MAX
    } else {
        leaf_budget(tree.header.indexed_rows)
    };
    let candidates = search_tree(tree, &scaled, pool, budget, store)?;
    let boundary = candidates.last().map_or(0.0, |p| f64::from(p.1));
    let pool_rows: Vec<usize> = candidates.iter().map(|&(row, _)| row as usize).collect();
    store.train_x.prefetch_rows(&pool_rows);
    let mut refined: Vec<(u32, f64)> = candidates
        .iter()
        .map(|&(row, _)| {
            Ok((
                row,
                crate::distance::row_sq_l2(
                    ndarray::ArrayView1::from(query),
                    ndarray::ArrayView1::from(store.train_x.mmap_row_slice(row as usize)?),
                    store.scale_x,
                    ndarray::ArrayView1::from(store.x_scale),
                ),
            ))
        })
        .collect::<Result<_, BpannError>>()?;
    refined.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    let all_rows = refined.len() == tree.header.indexed_rows;
    let certified = refined
        .get(k.saturating_sub(1))
        .is_some_and(|p| refinement_lower_bound(tree, query, store, boundary) > p.1);
    if !all_rows && (!tree.rows_cached() || (exact_fallback() && !certified)) {
        FALLBACK_RETURNS.fetch_add(1, Ordering::Relaxed);
        return search_tree_f64(tree, query, k, store);
    }
    if !all_rows && !certified && !f32_separates_top_k(&candidates, &refined, k) {
        FALLBACK_RETURNS.fetch_add(1, Ordering::Relaxed);
        let budget = tie_leaf_budget(tree.header.indexed_rows);
        return search_tree_f64_with_budget(tree, query, k, store, budget);
    }
    CERTIFIED_RETURNS.fetch_add(1, Ordering::Relaxed);
    refined.truncate(k);
    Ok(refined)
}

fn f32_separates_top_k(candidates: &[(u32, f32)], refined: &[(u32, f64)], k: usize) -> bool {
    let Some(&(_, boundary)) = candidates.last() else {
        return true;
    };
    refined.iter().take(k).all(|&(row, _)| {
        candidates
            .iter()
            .find(|&&(r, _)| r == row)
            .is_some_and(|&(_, d)| d < boundary)
    })
}

fn refinement_lower_bound(
    tree: &Tree,
    query: &[f64],
    store: &MmapSearchStore<'_>,
    boundary: f64,
) -> f64 {
    let eta = 4.0 * tree.coordinate_relative_error
        + 16.0 * (query.len() + 1) as f64 * f32::EPSILON as f64;
    let norm: f64 = query
        .iter()
        .enumerate()
        .map(|(j, &q)| {
            let q = if store.scale_x {
                q / store.x_scale[j]
            } else {
                q
            };
            q * q
        })
        .sum();
    if eta >= 0.1 || !norm.is_finite() || !boundary.is_finite() {
        return 0.0;
    }
    let absolute = 64.0
        * (query.len() as f64).sqrt()
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
    let budget = if exact_fallback() {
        usize::MAX
    } else {
        leaf_budget(tree.header.indexed_rows)
    };
    search_tree_f64_with_budget(tree, query, k, store, budget)
}

fn search_tree_f64_with_budget(
    tree: &Tree,
    query: &[f64],
    k: usize,
    store: &MmapSearchStore<'_>,
    budget: usize,
) -> Result<Vec<(u32, f64)>, BpannError> {
    let mut scaled = Vec::with_capacity(query.len());
    bpann_row_to_f32(query, store.scale_x, store.x_scale, &mut scaled);
    let query_scaled: Vec<f64> = query
        .iter()
        .enumerate()
        .map(|(j, &q)| {
            if store.scale_x {
                q / store.x_scale[j]
            } else {
                q
            }
        })
        .collect();
    search_tree_scored(tree, &scaled, k, budget, &F64_WALKS, |row, _, limit| {
        let point = store.train_x.mmap_row_slice(row as usize)?;
        let mut distance = 0.0;
        for (j, (&q, &x)) in query_scaled.iter().zip(point).enumerate() {
            let x = if store.scale_x {
                x / store.x_scale[j]
            } else {
                x
            };
            let delta = q - x;
            distance += delta * delta;
            if distance > limit {
                return Ok(None);
            }
        }
        Ok(Some(distance.max(0.0)))
    })
}

type FrontierEntry = (OrderedF32, OrderedF32, u32);

struct Walk<'a, F> {
    tree: &'a Tree,
    query: &'a [f32],
    top: TopK,
    score: F,
    leaves: usize,
}

impl<F> Walk<'_, F>
where
    F: FnMut(u32, Option<&[f32]>, f64) -> Result<Option<f64>, BpannError>,
{
    fn score_leaf(&mut self, page: u32) -> Result<(), BpannError> {
        let s = &self.tree.store;
        let d = self.query.len().max(1);
        if self.tree.rows_cached() {
            s.prefetch(page);
        }
        let block = self.tree.rows_cached().then(|| s.block(page));
        for (i, &row) in s.ids(page).iter().enumerate() {
            let x = block.map(|v| &v[i * d..(i + 1) * d]);
            if let Some(distance) = (self.score)(row, x, self.top.limit())? {
                self.top.offer(row, distance);
            }
        }
        self.leaves += 1;
        Ok(())
    }

    fn children(&self, page: u32) -> Vec<FrontierEntry> {
        let s = &self.tree.store;
        let d = self.query.len().max(1);
        s.prefetch(page);
        s.ids(page)
            .iter()
            .zip(s.block(page).chunks_exact(d))
            .filter_map(|(&child, c)| {
                let d2 = l2_sq_f32(self.query, c);
                let radius = s.radius(child);
                let bound = lower_bound(d2, radius);
                (!self.top.cannot_improve(bound))
                    .then_some((OrderedF32(priority(d2, radius)), OrderedF32(bound), child))
            })
            .collect()
    }

    fn best_first(&mut self, leaf_budget: usize) -> Result<bool, BpannError> {
        let root = (OrderedF32(0.0), OrderedF32(0.0), self.tree.header.root_page_id);
        let mut frontier = BinaryHeap::from([Reverse(root)]);
        while let Some(Reverse((_, bound, page_id))) = frontier.pop() {
            if self.leaves >= leaf_budget && self.top.heap.len() >= self.top.k {
                return Ok(true);
            }
            if self.top.cannot_improve(bound.0) {
                continue;
            }
            if self.tree.store.is_leaf(page_id) {
                self.score_leaf(page_id)?;
                continue;
            }
            frontier.extend(self.children(page_id).into_iter().map(Reverse));
            if frontier.len() > MAX_FRONTIER_PAGES {
                frontier = keep_most_promising(frontier);
            }
        }
        Ok(false)
    }

    fn depth_first(&mut self) -> Result<(), BpannError> {
        let mut stack = vec![(OrderedF32(0.0), self.tree.header.root_page_id)];
        while let Some((bound, page_id)) = stack.pop() {
            if self.top.cannot_improve(bound.0) {
                continue;
            }
            if self.tree.store.is_leaf(page_id) {
                self.score_leaf(page_id)?;
                continue;
            }
            let mut kids = self.children(page_id);
            kids.sort_unstable_by(|a, b| b.cmp(a));
            stack.extend(kids.into_iter().map(|(_, bound, child)| (bound, child)));
        }
        Ok(())
    }
}

fn keep_most_promising(
    frontier: BinaryHeap<Reverse<FrontierEntry>>,
) -> BinaryHeap<Reverse<FrontierEntry>> {
    let keep = MAX_FRONTIER_PAGES / 2;
    let mut entries = frontier.into_vec();
    entries.select_nth_unstable_by(keep, |a, b| a.0.cmp(&b.0));
    entries.truncate(keep);
    BinaryHeap::from(entries)
}

fn search_tree_scored<F>(
    tree: &Tree,
    query: &[f32],
    k: usize,
    leaf_budget: usize,
    meter: &WalkMeter,
    score: F,
) -> Result<Vec<(u32, f64)>, BpannError>
where
    F: FnMut(u32, Option<&[f32]>, f64) -> Result<Option<f64>, BpannError>,
{
    if k == 0 || tree.num_pages() == 0 {
        return Ok(Vec::new());
    }
    let mut walk = Walk {
        tree,
        query,
        top: TopK {
            k,
            heap: BinaryHeap::with_capacity(k + 1),
        },
        score,
        leaves: 0,
    };
    let stopped_on_budget = if leaf_budget == usize::MAX {
        walk.depth_first()?;
        false
    } else {
        walk.best_first(leaf_budget)?
    };
    meter.record(walk.leaves, stopped_on_budget);
    Ok(walk.top.into_sorted())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_fallback_defaults_off() {
        assert!(!exact_fallback());
    }

    #[test]
    fn leaf_budget_grows_like_ln_n() {
        assert_eq!(leaf_budget(0), 1);
        assert_eq!(leaf_budget(1_000_000), 139);
        assert_eq!(leaf_budget(10_000_000), 162);
        assert_eq!(tie_leaf_budget(1_000_000), 1382);
        let ratio = leaf_budget(10_000_000) as f64 / leaf_budget(1_000_000) as f64;
        let ln_ratio = 10_000_000f64.ln() / 1_000_000f64.ln();
        assert!((ratio - ln_ratio).abs() < 1.0 / leaf_budget(1_000_000) as f64);
    }

    #[test]
    fn priority_discounts_radius_and_never_goes_negative() {
        assert_eq!(priority(4.0, 0.0), 4.0);
        assert!((priority(4.0, 10.0) - 1.0).abs() < 1e-6);
        assert_eq!(priority(1.0, 100.0), 0.0);
        assert_eq!(lower_bound(4.0, 1.0), 1.0);
        assert_eq!(lower_bound(4.0, 3.0), 0.0);
    }

    #[test]
    fn keep_most_promising_halves_the_frontier_and_keeps_the_best() {
        let entries = (0..=MAX_FRONTIER_PAGES as u32)
            .rev()
            .map(|i| Reverse((OrderedF32(i as f32), OrderedF32(0.0), i)));
        let kept = keep_most_promising(entries.collect());
        assert_eq!(kept.len(), MAX_FRONTIER_PAGES / 2);
        let mut ids: Vec<u32> = kept.into_iter().map(|Reverse((_, _, id))| id).collect();
        ids.sort_unstable();
        assert_eq!(ids, (0..(MAX_FRONTIER_PAGES / 2) as u32).collect::<Vec<_>>());
    }
}
