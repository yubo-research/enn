use super::*;
use crate::index::tree_lloyd::{set_lloyd, set_lloyd_flush};
use crate::index::tree_pbuf::set_pbuf_rows;
use crate::index::tree_pbuf::tests::{assert_buffers_covered, missed_neighbours};
use crate::index::tree_pbuf_all::set_pbuf_all;
use crate::index::tree_pbuf_tag::set_pbuf_tag;

fn row(r: usize, d: usize) -> Vec<f32> {
    (0..d)
        .map(|j| ((r * 7919 + j * 104_729) % 1000) as f32 / 1000.0 + (r % 7) as f32)
        .collect()
}

fn arm(rows: usize) {
    set_pbuf_tag(true);
    set_pbuf_rows(rows);
    set_pbuf_all(true);
    set_lloyd(4, 16, 8, true);
    set_lloyd_flush(true);
    set_reparent(true);
    set_reparent_every(10);
    set_lloyd_retag(false);
}

fn depths(t: &Tree, page: u32, depth: usize, out: &mut Vec<usize>) {
    let s = &t.store;
    if s.is_leaf(page) {
        out.push(depth);
        return;
    }
    for &c in s.ids(page) {
        assert_eq!(s.parent(c), Some(page), "child {c} of {page}");
        depths(t, c, depth + 1, out);
    }
}

#[test]
fn reparented_tree_keeps_rows_parents_and_ball_cover() {
    arm(50);
    let (n, d) = (8000usize, 4usize);
    let dir = tempfile::tempdir().unwrap();
    let mut t = Tree::new_tree(0, &row(0, d), d, dir.path().join("i")).unwrap();
    for r in 1..n {
        t.insert_row(r as u32, &row(r, d)).unwrap();
    }
    let (tried, moved, _, _) = take_reparent_stats();
    assert!(tried > 0 && moved > 0, "tried {tried} moved {moved}");
    assert_eq!(t.leaf_row_ids(), (0..n as u32).collect::<Vec<u32>>());
    assert_buffers_covered(&t);
    let mut ds = Vec::new();
    depths(&t, t.header.root_page_id, 0, &mut ds);
    assert!(ds.iter().all(|&x| x == ds[0]), "unbalanced leaves");
    t.pbuf_flush_all().unwrap();
    assert_eq!(t.leaf_row_ids(), (0..n as u32).collect::<Vec<u32>>());
}

#[test]
fn reparented_search_finds_true_neighbours() {
    arm(200);
    let missed = missed_neighbours(200);
    assert!(missed <= 2, "{missed} of 140 true neighbours missed");
}
