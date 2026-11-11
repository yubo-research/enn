use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::distance::{bpann_row_to_f32, l2_sq_f32, sq_l2_f64_abandon};
use crate::error::BpannError;
use crate::index::search::MmapSearchStore;
use crate::index::tree::Tree;
use crate::small_n_search::{OrderedF32, OrderedFloat};

pub const TREE_SEARCH_LEAVES_PER_LN_N: f64 = 10.0;

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

pub const TREE_SEARCH_STOP_EPSILON: f64 = 1.0;

static STOP_EPSILON_BITS: AtomicU64 = AtomicU64::new(TREE_SEARCH_STOP_EPSILON.to_bits());

pub fn stop_epsilon() -> f64 {
    f64::from_bits(STOP_EPSILON_BITS.load(Ordering::Relaxed))
}

pub fn set_stop_epsilon(value: f64) {
    STOP_EPSILON_BITS.store(value.max(0.0).to_bits(), Ordering::Relaxed);
}

static EXACT_FALLBACK: AtomicU64 = AtomicU64::new(0);

pub fn exact_fallback() -> bool {
    EXACT_FALLBACK.load(Ordering::Relaxed) != 0
}

pub fn set_exact_fallback(value: bool) {
    EXACT_FALLBACK.store(u64::from(value), Ordering::Relaxed);
}

static SKIP_BUDGET_FALLBACK: AtomicU64 = AtomicU64::new(1);

pub fn set_skip_budget_fallback(value: bool) {
    SKIP_BUDGET_FALLBACK.store(u64::from(value), Ordering::Relaxed);
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

pub static XP_INTERNAL_PAGES: AtomicU64 = AtomicU64::new(0);
pub static XP_INTERNAL_ROWS: AtomicU64 = AtomicU64::new(0);
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
    let ln_n = (indexed_rows.max(1) as f64).ln();
    ((leaf_coefficient() * ln_n).ceil() as usize).max(1)
}

pub fn internal_page_cap(indexed_rows: usize) -> usize {
    let ln_n = (indexed_rows.max(1) as f64).ln();
    ((TREE_SEARCH_INTERNAL_PAGES_PER_LN_N * ln_n).ceil() as usize).max(1)
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

pub const TREE_SEARCH_INTERNAL_PAGES_PER_LN_N: f64 = 30.0;

type FrontierItem = (u16, OrderedF32, OrderedF32, u32, u16);

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
        let scale = (1.0 + stop_epsilon()).powi(2);
        let lower =
            (f64::from(bound) * (1.0 - 1e-4) - 64.0 * f32::MIN_POSITIVE as f64).max(0.0) * scale;
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
    let stops_before = F32_WALKS.budget_stops.load(Ordering::Relaxed);
    let candidates = search_tree(tree, &scaled, pool, budget, store)?;
    let stopped = F32_WALKS.budget_stops.load(Ordering::Relaxed) > stops_before;
    let boundary = candidates.last().map_or(0.0, |p| f64::from(p.1));
    let ids: Vec<u32> = candidates.iter().map(|p| p.0).collect();
    store.train_x.prefetch_rows(&ids);
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
    let skip = stopped && SKIP_BUDGET_FALLBACK.load(Ordering::Relaxed) != 0 && refined.len() >= k;
    if !all_rows && !skip && (!tree.rows_cached() || !certified) {
        FALLBACK_RETURNS.fetch_add(1, Ordering::Relaxed);
        return search_tree_f64(tree, query, k, store);
    }
    CERTIFIED_RETURNS.fetch_add(1, Ordering::Relaxed);
    refined.truncate(k);
    Ok(refined)
}

struct F32Slack {
    eta: f64,
    norm_sqrt: f64,
    absolute: f64,
    usable: bool,
}

impl F32Slack {
    fn new(tree: &Tree, query: &[f64], store: &MmapSearchStore<'_>) -> Self {
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
        let absolute = 64.0
            * (query.len() as f64).sqrt()
            * (tree.coordinate_absolute_error + (f32::MIN_POSITIVE as f64).sqrt());
        Self {
            eta,
            norm_sqrt: norm.sqrt(),
            absolute,
            usable: eta < 0.1 && norm.is_finite(),
        }
    }
    fn lower(&self, boundary: f64) -> f64 {
        if !self.usable || !boundary.is_finite() {
            return 0.0;
        }
        let lower = boundary.sqrt() * (1.0 - self.eta) - self.eta * self.norm_sqrt - self.absolute;
        lower.max(0.0).powi(2)
    }
}

fn refinement_lower_bound(
    tree: &Tree,
    query: &[f64],
    store: &MmapSearchStore<'_>,
    boundary: f64,
) -> f64 {
    F32Slack::new(tree, query, store).lower(boundary)
}

pub fn search_tree_f64(
    tree: &Tree,
    query: &[f64],
    k: usize,
    store: &MmapSearchStore<'_>,
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
    let budget = if exact_fallback() {
        usize::MAX
    } else {
        leaf_budget(tree.header.indexed_rows)
    };
    let slack = F32Slack::new(tree, query, store);
    search_tree_scored(
        tree,
        &scaled,
        k,
        budget,
        &F64_WALKS,
        |row, cached, limit| {
            if let Some(x) = cached {
                if limit.is_finite() && slack.lower(f64::from(l2_sq_f32(&scaled, x))) > limit {
                    return Ok(None);
                }
            }
            let point = store.train_x.mmap_row_slice(row as usize)?;
            Ok(sq_l2_f64_abandon(
                &query_scaled,
                point,
                store.scale_x,
                store.x_scale,
                limit,
            ))
        },
    )
}

fn search_tree_scored(
    tree: &Tree,
    query: &[f32],
    k: usize,
    leaf_budget: usize,
    meter: &WalkMeter,
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
    let mut frontier: BinaryHeap<Reverse<FrontierItem>> = BinaryHeap::new();
    frontier.push(Reverse((
        0,
        OrderedF32(0.0),
        OrderedF32(0.0),
        tree.header.root_page_id,
        0,
    )));
    let mut deepest_first = false;
    let mut leaves = 0usize;
    let mut expanded = 0usize;
    let internal_cap = internal_page_cap(tree.header.indexed_rows);
    let mut stopped_on_budget = false;
    while let Some(Reverse((phase, key, bound, page_id, depth))) = frontier.pop() {
        if !deepest_first && expanded >= internal_cap {
            deepest_first = true;
            let mut all = std::mem::take(&mut frontier).into_vec();
            all.push(Reverse((phase, key, bound, page_id, depth)));
            frontier = all
                .into_iter()
                .map(|Reverse((_, k2, b2, p2, d2))| Reverse((u16::MAX - d2, k2, b2, p2, d2)))
                .collect();
            continue;
        }
        if leaves >= leaf_budget && top.heap.len() >= k {
            stopped_on_budget = true;
            break;
        }
        if top.cannot_improve(bound.0) {
            continue;
        }
        if !s.is_leaf(page_id) {
            expanded += 1;
            XP_INTERNAL_PAGES.fetch_add(1, Ordering::Relaxed);
            XP_INTERNAL_ROWS.fetch_add(s.ids(page_id).len() as u64, Ordering::Relaxed);
            for (&child, c) in s.ids(page_id).iter().zip(s.block(page_id).chunks_exact(d)) {
                let d2 = l2_sq_f32(query, c);
                let radius = s.radius(child);
                let bound = lower_bound(d2, radius);
                if !top.cannot_improve(bound) {
                    let key = priority(d2, radius);
                    let phase = if deepest_first {
                        u16::MAX - (depth + 1)
                    } else {
                        0
                    };
                    frontier.push(Reverse((
                        phase,
                        OrderedF32(key),
                        OrderedF32(bound),
                        child,
                        depth + 1,
                    )));
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
    meter.record(leaves, stopped_on_budget);
    Ok(top.into_sorted())
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
        let ratio = leaf_budget(10_000_000) as f64 / leaf_budget(1_000_000) as f64;
        assert!((ratio - 10_000_000f64.ln() / 1_000_000f64.ln()).abs() < 1e-2);
    }

    #[test]
    fn internal_page_cap_grows_like_ln_n() {
        assert_eq!(internal_page_cap(0), 1);
        let ratio = internal_page_cap(10_000_000) as f64 / internal_page_cap(1_000_000) as f64;
        assert!((ratio - 10_000_000f64.ln() / 1_000_000f64.ln()).abs() < 1e-2);
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
    fn epsilon_prunes_more_and_zero_is_exact() {
        let mut top = TopK {
            k: 1,
            heap: BinaryHeap::new(),
        };
        top.offer(0, 4.0);
        set_stop_epsilon(1.0);
        assert!(top.cannot_improve(1.1));
        assert!(!top.cannot_improve(0.9));
        set_stop_epsilon(0.0);
        assert!(!top.cannot_improve(3.9));
        assert!(top.cannot_improve(4.1));
        set_stop_epsilon(TREE_SEARCH_STOP_EPSILON);
    }
}
