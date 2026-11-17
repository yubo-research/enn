use super::*;
use crate::index::tree_pbuf::tests::missed_neighbours;
use crate::index::tree_pbuf_tag::set_pbuf_tag;

#[test]
fn diag_reports_one_location_per_target_for_one_walk() {
    set_pbuf_tag(true);
    set_pbuf_diag_targets(vec![1, 2, 3]);
    let _ = missed_neighbours(200);
    let out = take_pbuf_diag();
    assert_eq!(out.len(), 3);
    assert!(out.iter().all(|r| matches!(r.1, 0 | 1 | 3)), "{out:?}");
    assert!(
        out.iter().filter(|r| r.1 != 1).all(|r| r.6 != u32::MAX),
        "{out:?}"
    );
    assert!(take_pbuf_diag().is_empty());
}
