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
fn internal_page_cap_default_coefficient() {
    assert_eq!(internal_coefficient(), TREE_SEARCH_INTERNAL_PAGES_PER_LN_N);
    assert_eq!(internal_page_cap(1_000_000), 139);
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
fn prefetch_pages_default_and_setter() {
    assert_eq!(prefetch_pages(), TREE_SEARCH_PREFETCH_PAGES as usize);
    set_prefetch_pages(0);
    assert_eq!(prefetch_pages(), 0);
    set_prefetch_pages(TREE_SEARCH_PREFETCH_PAGES as usize);
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
