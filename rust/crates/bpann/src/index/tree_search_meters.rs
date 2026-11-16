use std::sync::atomic::{AtomicU64, Ordering};

pub(super) struct WalkMeter {
    leaves: AtomicU64,
    walks: AtomicU64,
    pub(super) budget_stops: AtomicU64,
}

impl WalkMeter {
    pub(super) const fn new() -> Self {
        Self {
            leaves: AtomicU64::new(0),
            walks: AtomicU64::new(0),
            budget_stops: AtomicU64::new(0),
        }
    }

    pub(super) fn record(&self, leaves: usize, stopped_on_budget: bool) {
        self.walks.fetch_add(1, Ordering::Relaxed);
        self.leaves.fetch_add(leaves as u64, Ordering::Relaxed);
        if stopped_on_budget {
            self.budget_stops.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub(super) fn take(&self) -> (u64, u64, u64) {
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
pub(super) static PHASE_WALK_NS: AtomicU64 = AtomicU64::new(0);
pub(super) static PHASE_REFINE_NS: AtomicU64 = AtomicU64::new(0);
pub(super) static F32_WALKS: WalkMeter = WalkMeter::new();
pub(super) static F64_WALKS: WalkMeter = WalkMeter::new();
pub(super) static CERTIFIED_RETURNS: AtomicU64 = AtomicU64::new(0);
pub(super) static FALLBACK_RETURNS: AtomicU64 = AtomicU64::new(0);

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
