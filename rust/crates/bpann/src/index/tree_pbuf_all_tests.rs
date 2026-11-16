use super::*;
use crate::index::tree_pbuf::set_pbuf_rows;
use crate::index::tree_pbuf::tests::{assert_buffers_covered, missed_neighbours};

fn row(r: usize, d: usize) -> Vec<f32> {
    (0..d)
        .map(|j| ((r * 7919 + j * 104_729) % 1000) as f32 / 1000.0 + (r % 7) as f32)
        .collect()
}

fn upper_buffered(t: &Tree) -> usize {
    let s = &t.store;
    (0..t.num_pages() as u32)
        .filter(|&p| !s.is_free(p) && !s.is_leaf(p) && s.pbuf_len(p) > 0)
        .filter(|&p| s.ids(p).iter().any(|&c| !s.is_leaf(c)))
        .count()
}

#[test]
fn every_level_buffers_hold_every_row_and_stay_covered() {
    set_pbuf_rows(50);
    set_pbuf_all(true);
    set_pbuf_stagger(true);
    let (n, d) = (8000usize, 4usize);
    let dir = tempfile::tempdir().unwrap();
    let mut t = Tree::new_tree(0, &row(0, d), d, dir.path().join("i")).unwrap();
    for r in 1..n {
        t.insert_row(r as u32, &row(r, d)).unwrap();
    }
    assert_eq!(t.header.indexed_rows, n);
    assert_eq!(t.leaf_row_ids(), (0..n as u32).collect::<Vec<u32>>());
    assert!(upper_buffered(&t) > 0, "no buffer above the leaf parents");
    assert_buffers_covered(&t);
    t.assert_counts_consistent();
    t.pbuf_flush_all().unwrap();
    assert!(t.pbuf_row_ids().is_empty());
    assert_eq!(t.leaf_row_ids(), (0..n as u32).collect::<Vec<u32>>());
    t.assert_counts_consistent();
}

#[test]
fn staggered_capacity_spans_half_to_full() {
    set_pbuf_rows(1000);
    set_pbuf_stagger(true);
    let dir = tempfile::tempdir().unwrap();
    let t = Tree::new_tree(0, &row(0, 3), 3, dir.path().join("i")).unwrap();
    let caps: Vec<usize> = (0..2000).map(|p| t.store.pbuf_cap(p)).collect();
    assert!(caps.iter().all(|&c| (500..=1000).contains(&c)));
    let mean = caps.iter().sum::<usize>() as f64 / caps.len() as f64;
    assert!((700.0..800.0).contains(&mean), "mean cap {mean}");
}

#[test]
fn every_level_search_finds_true_neighbours() {
    set_pbuf_all(true);
    let missed = missed_neighbours(200);
    assert!(missed <= 2, "{missed} of 140 true neighbours missed");
}

#[test]
fn punched_buffers_keep_every_row_and_free_taken_pages() {
    set_pbuf_rows(50);
    set_pbuf_all(true);
    set_pbuf_punch(true);
    let (n, d) = (6000usize, 4usize);
    let dir = tempfile::tempdir().unwrap();
    let mut t = Tree::new_tree(0, &row(0, d), d, dir.path().join("i")).unwrap();
    for r in 1..n {
        t.insert_row(r as u32, &row(r, d)).unwrap();
    }
    assert_buffers_covered(&t);
    t.pbuf_flush_all().unwrap();
    assert!(t.pbuf_row_ids().is_empty());
    assert_eq!(t.leaf_row_ids(), (0..n as u32).collect::<Vec<u32>>());
    t.assert_counts_consistent();
    let root = t.header.root_page_id;
    let _ = t.store.pbuf_push(root, 7, &row(7, d));
    let _ = t.store.pbuf_take(root);
    let first = t.store.pbuf_region_bytes(root, 1);
    assert!(first.iter().all(|&b| b == 0), "taken page not punched");
}
