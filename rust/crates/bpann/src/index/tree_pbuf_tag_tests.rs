use super::*;
use crate::index::tree::Tree;
use crate::index::tree_pbuf::set_pbuf_rows;
use crate::index::tree_pbuf::tests::{assert_buffers_covered, missed_neighbours};
use crate::index::tree_pbuf_all::set_pbuf_all;

fn row(r: usize, d: usize) -> Vec<f32> {
    (0..d)
        .map(|j| ((r * 7919 + j * 104_729) % 1000) as f32 / 1000.0 + (r % 7) as f32)
        .collect()
}

fn assert_tags_covered(t: &Tree) {
    let s = &t.store;
    let half = 0.5 * (s.num_dim() as f32).sqrt();
    let dd = s.num_dim();
    let mut checked = 0usize;
    for p in 0..t.num_pages() as u32 {
        if s.is_free(p) || s.is_leaf(p) || s.pbuf_len(p) == 0 {
            continue;
        }
        let block = s.block(p);
        for (_, code, tag) in s.pbuf_tagged_records(p) {
            let slot = tag as usize;
            assert!(slot < s.len(p), "page {p} stale tag {tag}");
            let (scale, _, q) = decode_header(code);
            let v: Vec<f32> = q.iter().map(|&x| scale * f32::from(x)).collect();
            let slack = half * scale;
            let c = &block[slot * dd..(slot + 1) * dd];
            assert!(
                dist(&v, c) - slack <= s.route_radius(p, slot) * 1.0001 + 1e-5,
                "page {p} slot {slot}"
            );
            checked += 1;
        }
    }
    assert!(checked > 0, "no buffered rows checked");
}

#[test]
fn tagged_buffers_keep_every_row_and_routes_cover_tagged_rows() {
    set_pbuf_tag(true);
    set_pbuf_rows(50);
    set_pbuf_all(true);
    let (n, d) = (8000usize, 4usize);
    let dir = tempfile::tempdir().unwrap();
    let mut t = Tree::new_tree(0, &row(0, d), d, dir.path().join("i")).unwrap();
    for r in 1..n {
        t.insert_row(r as u32, &row(r, d)).unwrap();
    }
    assert!(t.store.pbuf_tagged());
    assert_eq!(t.leaf_row_ids(), (0..n as u32).collect::<Vec<u32>>());
    assert_buffers_covered(&t);
    assert_tags_covered(&t);
    t.pbuf_flush_all().unwrap();
    assert!(t.pbuf_row_ids().is_empty());
    assert_eq!(t.leaf_row_ids(), (0..n as u32).collect::<Vec<u32>>());
}

#[test]
fn tagged_every_level_search_finds_true_neighbours() {
    set_pbuf_tag(true);
    set_pbuf_all(true);
    let missed = missed_neighbours(200);
    assert!(missed <= 2, "{missed} of 140 true neighbours missed");
    let (scanned, visited, _, _) = take_pbuf_tag_stats();
    assert!(scanned < visited, "scanned {scanned} of {visited}");
}

#[test]
fn tagged_leaf_parent_search_finds_true_neighbours() {
    set_pbuf_tag(true);
    let missed = missed_neighbours(200);
    assert!(missed <= 2, "{missed} of 140 true neighbours missed");
}

#[test]
fn tag_pick_selects_slot_stale_or_all() {
    assert!(TagPick::All.keeps(3));
    assert!(TagPick::Rows(&[1, 2]).keeps(0));
    assert!(TagPick::Stale(4).keeps(STALE_TAG) && !TagPick::Stale(4).keeps(3));
}
