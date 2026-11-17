use std::sync::atomic::{AtomicU64, Ordering};

pub const TREE_SEARCH_LEAVES_PER_LN_N: f64 =
    100.0 * 64.0 / crate::index::tree::TREE_LEAF_CAPACITY as f64;

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

pub const TREE_SEARCH_PREFETCH_PAGES: u64 = 0;

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

pub const TREE_SEARCH_META_PREFETCH: u64 = 4;

static META_PREFETCH: AtomicU64 = AtomicU64::new(TREE_SEARCH_META_PREFETCH);

pub fn meta_prefetch() -> usize {
    META_PREFETCH.load(Ordering::Relaxed) as usize
}

pub fn set_meta_prefetch(value: usize) {
    META_PREFETCH.store(value as u64, Ordering::Relaxed);
}

pub const TREE_SEARCH_RADIUS_WEIGHT: f32 = 0.1;

static RADIUS_WEIGHT_BITS: AtomicU64 = AtomicU64::new(TREE_SEARCH_RADIUS_WEIGHT.to_bits() as u64);

pub fn radius_weight() -> f32 {
    f32::from_bits(RADIUS_WEIGHT_BITS.load(Ordering::Relaxed) as u32)
}

pub fn set_radius_weight(value: f32) {
    RADIUS_WEIGHT_BITS.store(u64::from(value.max(0.0).to_bits()), Ordering::Relaxed);
}

static PATIENCE_BITS: AtomicU64 = AtomicU64::new(0);

pub fn patience_coefficient() -> f64 {
    f64::from_bits(PATIENCE_BITS.load(Ordering::Relaxed))
}

pub fn set_patience_coefficient(value: f64) {
    PATIENCE_BITS.store(value.max(0.0).to_bits(), Ordering::Relaxed);
}

pub fn patience_leaves(indexed_rows: usize) -> usize {
    patience_for(patience_coefficient(), indexed_rows)
}

pub fn patience_for(p: f64, indexed_rows: usize) -> usize {
    if p <= 0.0 {
        return usize::MAX;
    }
    ((p * (indexed_rows.max(1) as f64).ln()).ceil() as usize).max(1)
}

pub(crate) const fn parse_cap(v: Option<&str>, default: usize) -> usize {
    let Some(v) = v else { return default };
    let b = v.as_bytes();
    let (mut i, mut n) = (0, 0usize);
    while i < b.len() {
        assert!(
            b[i].is_ascii_digit(),
            "ENN_LEAF_CAP must be a decimal integer"
        );
        n = n * 10 + (b[i] - b'0') as usize;
        i += 1;
    }
    assert!(n >= 8, "ENN_LEAF_CAP must be >= 8");
    n
}
