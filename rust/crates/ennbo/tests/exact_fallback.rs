use ennbo::set_exact_fallback;
use ennbo_bpann::index::tree_search;

#[test]
fn public_switch_controls_bpann_exact_fallback() {
    set_exact_fallback(true);
    assert!(tree_search::exact_fallback());
    set_exact_fallback(false);
    assert!(!tree_search::exact_fallback());
}
