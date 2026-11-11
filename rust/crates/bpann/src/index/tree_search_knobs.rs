use std::sync::atomic::{AtomicU64, Ordering};

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

pub const TREE_SEARCH_PREFETCH_PAGES: u64 = 10;

static PREFETCH_PAGES: AtomicU64 = AtomicU64::new(TREE_SEARCH_PREFETCH_PAGES);

pub fn prefetch_pages() -> usize {
    PREFETCH_PAGES.load(Ordering::Relaxed) as usize
}

pub fn set_prefetch_pages(value: usize) {
    PREFETCH_PAGES.store(value as u64, Ordering::Relaxed);
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

pub(super) static SKIP_BUDGET_FALLBACK: AtomicU64 = AtomicU64::new(1);
pub(super) static F32_RERANK_SKIP: AtomicU64 = AtomicU64::new(1);

pub fn f32_rerank_skip() -> bool {
    F32_RERANK_SKIP.load(Ordering::Relaxed) != 0
}

pub fn set_f32_rerank_skip(value: bool) {
    F32_RERANK_SKIP.store(u64::from(value), Ordering::Relaxed);
}

pub fn set_skip_budget_fallback(value: bool) {
    SKIP_BUDGET_FALLBACK.store(u64::from(value), Ordering::Relaxed);
}
