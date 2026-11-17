use super::*;
use crate::index::tree_lloyd::{set_lloyd, set_lloyd_flush};
use crate::index::tree_pbuf::set_pbuf_rows;
use crate::index::tree_pbuf::tests::{assert_buffers_covered, missed_neighbours};
use crate::index::tree_pbuf_all::{set_pbuf_all, set_pbuf_leaf_rows};
use crate::index::tree_pbuf_tag::set_pbuf_tag;

fn row(r: usize, d: usize) -> Vec<f32> {
    (0..d)
        .map(|j| ((r * 7919 + j * 104_729) % 1000) as f32 / 1000.0 + (r % 7) as f32)
        .collect()
}

fn grown(n: usize, d: usize, dir: &std::path::Path) -> Tree {
    let mut t = Tree::new_tree(0, &row(0, d), d, dir.join("i")).unwrap();
    for r in 1..n {
        t.insert_row(r as u32, &row(r, d)).unwrap();
    }
    t
}

fn arm(rows: usize) {
    set_pbuf_tag(true);
    set_pbuf_rows(rows);
    set_pbuf_all(true);
    set_lloyd(4, 16, 8, true);
    set_lloyd_flush(true);
    set_pbuf_rehome(true);
}

#[test]
fn rehomed_rows_stay_indexed_and_covered_by_ancestor_balls() {
    arm(50);
    let (n, d) = (8000usize, 4usize);
    let dir = tempfile::tempdir().unwrap();
    let mut t = grown(n, d, dir.path());
    let (checked, moved) = take_pbuf_rehome_stats();
    assert!(checked > 0 && moved > 0, "checked {checked} moved {moved}");
    assert_eq!(t.leaf_row_ids(), (0..n as u32).collect::<Vec<u32>>());
    assert_buffers_covered(&t);
    for p in 0..t.num_pages() as u32 {
        let s = &t.store;
        if !s.is_free(p) && !s.is_leaf(p) && s.pbuf_len(p) > 0 {
            assert!(s.pbuf_tagged_records(p).all(|r| (r.2 as usize) < s.len(p)));
        }
    }
    t.pbuf_flush_all().unwrap();
    assert!(t.pbuf_row_ids().is_empty());
    assert_eq!(t.leaf_row_ids(), (0..n as u32).collect::<Vec<u32>>());
}

#[test]
fn rehomed_search_finds_true_neighbours() {
    arm(200);
    let missed = missed_neighbours(200);
    assert!(missed <= 2, "{missed} of 140 true neighbours missed");
}

#[test]
fn leaf_parent_cap_bounds_leaf_parent_buffers_only() {
    set_pbuf_all(true);
    set_pbuf_rows(200);
    set_pbuf_leaf_rows(20);
    let (n, d) = (6000usize, 4usize);
    let dir = tempfile::tempdir().unwrap();
    let t = grown(n, d, dir.path());
    set_pbuf_leaf_rows(0);
    let s = &t.store;
    let mut leaf_parents = 0usize;
    for p in 0..t.num_pages() as u32 {
        if s.is_free(p) || s.is_leaf(p) || s.len(p) == 0 {
            continue;
        }
        if s.is_leaf(s.ids(p)[0]) {
            leaf_parents += 1;
            assert!(s.pbuf_len(p) < 20, "page {p} holds {}", s.pbuf_len(p));
        }
    }
    assert!(leaf_parents > 1);
    assert_eq!(t.leaf_row_ids(), (0..n as u32).collect::<Vec<u32>>());
}
