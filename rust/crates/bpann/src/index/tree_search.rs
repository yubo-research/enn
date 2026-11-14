use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::distance::{bpann_row_to_f32, l2_sq_f32, sq_l2_f64_abandon};
use crate::error::BpannError;
use crate::index::search::MmapSearchStore;
use crate::index::tree::Tree;
use crate::index::tree_codes::{code_prefetch, leaf_codes};
pub use crate::index::tree_codes_search::take_coded_retries;
use crate::index::tree_codes_search::{candidate_walk, note_coded_retry, refine_rows, CodedScan};
use crate::index::tree_topk::TopK;
use crate::index::tree_walk::{pad_query, ColdProbe, Lookahead};
use crate::small_n_search::OrderedF32;

pub use super::tree_search_knobs::*;
use super::tree_search_knobs::{F32_RERANK_SKIP, SKIP_BUDGET_FALLBACK};

pub(super) struct WalkMeter {
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
pub static XP_LEAF_ROWS: AtomicU64 = AtomicU64::new(0);
static PHASE_WALK_NS: AtomicU64 = AtomicU64::new(0);
static PHASE_REFINE_NS: AtomicU64 = AtomicU64::new(0);
pub(super) static F32_WALKS: WalkMeter = WalkMeter::new();
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

pub fn take_phase_ns() -> (u64, u64) {
    (
        PHASE_WALK_NS.swap(0, Ordering::Relaxed),
        PHASE_REFINE_NS.swap(0, Ordering::Relaxed),
    )
}

pub fn take_leaf_rows() -> u64 {
    XP_LEAF_ROWS.swap(0, Ordering::Relaxed)
}

pub fn take_internal_totals() -> (u64, u64) {
    (
        XP_INTERNAL_PAGES.swap(0, Ordering::Relaxed),
        XP_INTERNAL_ROWS.swap(0, Ordering::Relaxed),
    )
}

thread_local! {
    static CASCADE_LEAVES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    static LAST_WALK_LEAVES: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn cascade_spent(indexed_rows: usize) -> bool {
    cascade_remaining(indexed_rows) < LAST_WALK_LEAVES.with(std::cell::Cell::get).max(1)
}

fn cascade_reset() {
    CASCADE_LEAVES.with(|c| c.set(0));
    LAST_WALK_LEAVES.with(|c| c.set(0));
}

fn cascade_remaining(indexed_rows: usize) -> usize {
    if exact_fallback() {
        usize::MAX
    } else {
        leaf_budget(indexed_rows).saturating_sub(CASCADE_LEAVES.with(std::cell::Cell::get))
    }
}

pub fn leaf_budget(indexed_rows: usize) -> usize {
    let ln_n = (indexed_rows.max(1) as f64).ln();
    ((leaf_coefficient() * ln_n).ceil() as usize).max(1)
}

static INTERNAL_COEFFICIENT_BITS: AtomicU64 = AtomicU64::new(0);

pub fn internal_coefficient() -> f64 {
    let bits = INTERNAL_COEFFICIENT_BITS.load(Ordering::Relaxed);
    if bits == 0 {
        TREE_SEARCH_INTERNAL_PAGES_PER_LN_N
    } else {
        f64::from_bits(bits)
    }
}

pub fn set_internal_coefficient(value: f64) {
    let bits = if value == TREE_SEARCH_INTERNAL_PAGES_PER_LN_N {
        0
    } else {
        value.to_bits()
    };
    INTERNAL_COEFFICIENT_BITS.store(bits, Ordering::Relaxed);
}

pub fn internal_page_cap(indexed_rows: usize) -> usize {
    let ln_n = (indexed_rows.max(1) as f64).ln();
    ((internal_coefficient() * ln_n).ceil() as usize).max(1)
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

pub const TREE_SEARCH_INTERNAL_PAGES_PER_LN_N: f64 = 10.0;

pub(super) type FrontierItem = (u16, OrderedF32, OrderedF32, u32, u16);

pub fn search_tree(
    tree: &Tree,
    query: &[f32],
    k: usize,
    leaf_budget: usize,
    store: &MmapSearchStore<'_>,
) -> Result<Vec<(u32, f32)>, BpannError> {
    let walk = search_tree_scored(
        tree,
        query,
        k,
        leaf_budget,
        &F32_WALKS,
        None,
        |row, cached, _| {
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
        },
    );
    let rows = walk?;
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
    cascade_reset();
    refined_walk(tree, query, k, store, extra, leaf_codes())
}

fn refined_walk(
    tree: &Tree,
    query: &[f64],
    k: usize,
    store: &MmapSearchStore<'_>,
    extra: usize,
    codes: bool,
) -> Result<Vec<(u32, f64)>, BpannError> {
    if exact_fallback() && !tree.rows_cached() {
        FALLBACK_RETURNS.fetch_add(1, Ordering::Relaxed);
        return search_tree_f64(tree, query, k, store);
    }
    let coded = codes && tree.rows_cached();
    let mut scaled = Vec::with_capacity(query.len());
    bpann_row_to_f32(query, store.scale_x, store.x_scale, &mut scaled);
    let budget = cascade_remaining(tree.header.indexed_rows).max(1);
    let stops_before = F32_WALKS.budget_stops.load(Ordering::Relaxed);
    let t_walk = std::time::Instant::now();
    let (candidates, boundary) = candidate_walk(tree, &scaled, (k, extra, budget), coded, store)?;
    PHASE_WALK_NS.fetch_add(t_walk.elapsed().as_nanos() as u64, Ordering::Relaxed);
    let t_refine = std::time::Instant::now();
    let stopped = F32_WALKS.budget_stops.load(Ordering::Relaxed) > stops_before;
    if stopped
        && SKIP_BUDGET_FALLBACK.load(Ordering::Relaxed) != 0
        && tree.rows_cached()
        && candidates.len() >= k
        && F32_RERANK_SKIP.load(Ordering::Relaxed) != 0
    {
        PHASE_REFINE_NS.fetch_add(t_refine.elapsed().as_nanos() as u64, Ordering::Relaxed);
        CERTIFIED_RETURNS.fetch_add(1, Ordering::Relaxed);
        return Ok(candidates
            .iter()
            .take(k)
            .map(|&(row, d)| (row, f64::from(d)))
            .collect());
    }
    let mut refined = refine_rows(query, &candidates, store)?;
    PHASE_REFINE_NS.fetch_add(t_refine.elapsed().as_nanos() as u64, Ordering::Relaxed);
    let all_rows = refined.len() == tree.header.indexed_rows;
    let certified = refined
        .get(k.saturating_sub(1))
        .is_some_and(|p| refinement_lower_bound(tree, query, store, boundary) > p.1);
    let spent = cascade_spent(tree.header.indexed_rows);
    let skip = (stopped || spent)
        && SKIP_BUDGET_FALLBACK.load(Ordering::Relaxed) != 0
        && refined.len() >= k;
    if !all_rows && !skip && (!tree.rows_cached() || !certified) {
        if coded {
            note_coded_retry();
            return refined_walk(tree, query, k, store, extra, false);
        }
        FALLBACK_RETURNS.fetch_add(1, Ordering::Relaxed);
        return f64_walk(tree, query, k, store);
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
    cascade_reset();
    f64_walk(tree, query, k, store)
}

fn f64_walk(
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
    let budget = cascade_remaining(tree.header.indexed_rows).max(1);
    let slack = F32Slack::new(tree, query, store);
    search_tree_scored(
        tree,
        &scaled,
        k,
        budget,
        &F64_WALKS,
        None,
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

pub(super) fn search_tree_scored(
    tree: &Tree,
    query: &[f32],
    k: usize,
    leaf_budget: usize,
    meter: &WalkMeter,
    mut coded: Option<&mut CodedScan>,
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
    let orig = query.len();
    let query = &*pad_query(query, s.num_dim());
    let qn2: f32 = query.iter().map(|x| x * x).sum();
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
    let probe = ColdProbe::start();
    let mut ahead = Lookahead::new(if coded.is_some() { code_prefetch() } else { 0 });
    while let Some(Reverse((phase, key, bound, page_id, depth))) = ahead.pop(&mut frontier) {
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
            let mut fresh: Vec<(f32, u32)> = Vec::new();
            let collect_fresh = prefetch_pages() > 0 || probe.ahead() > 0;
            s.internal_scores(
                (page_id, depth),
                (query, qn2),
                coded.is_some(),
                |child, d2, e, radius| {
                    let low = if e > 0.0 {
                        (d2.sqrt() - e).max(0.0).powi(2)
                    } else {
                        d2
                    };
                    let bound = lower_bound(low, radius);
                    if !top.cannot_improve(bound) {
                        let key = priority(d2, radius);
                        let phase = if deepest_first {
                            u16::MAX - (depth + 1)
                        } else {
                            0
                        };
                        if collect_fresh {
                            fresh.push((key, child));
                        }
                        frontier.push(Reverse((
                            phase,
                            OrderedF32(key),
                            OrderedF32(bound),
                            child,
                            depth + 1,
                        )));
                    }
                },
            );
            probe.advise(s, &mut fresh, coded.is_some());
            continue;
        }
        ahead.before_leaf(s, &mut frontier, coded.as_deref_mut());
        XP_LEAF_ROWS.fetch_add(s.ids(page_id).len() as u64, Ordering::Relaxed);
        score_leaf(
            tree,
            page_id,
            (query, qn2, orig),
            coded.as_deref_mut(),
            &mut top,
            &mut score,
        )?;
        leaves += 1;
    }
    meter.record(leaves, stopped_on_budget);
    CASCADE_LEAVES.with(|c| c.set(c.get() + leaves));
    LAST_WALK_LEAVES.with(|c| c.set(leaves));
    Ok(top.into_sorted())
}

fn score_leaf(
    tree: &Tree,
    page_id: u32,
    (query, qn2, orig): (&[f32], f32, usize),
    coded: Option<&mut CodedScan>,
    top: &mut TopK,
    score: &mut impl FnMut(u32, Option<&[f32]>, f64) -> Result<Option<f64>, BpannError>,
) -> Result<(), BpannError> {
    let s = &tree.store;
    if let Some(scan) = coded {
        scan.scan_leaf(s, page_id, query, qn2, &mut |i, d2| top.offer(i, d2));
        return Ok(());
    }
    let _ = (orig, query);
    for &row in s.ids(page_id) {
        if let Some(distance) = score(row, None, top.limit())? {
            top.offer(row, distance);
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "tree_search_tests.rs"]
mod tests;
