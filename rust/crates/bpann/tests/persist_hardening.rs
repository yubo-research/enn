//! Persist-on-close hardening tests: in-session parity and idempotency.

use bpann::BpannBackend;
use tempfile::TempDir;

#[test]
fn multi_fragment_persist_preserves_in_session_neighbors() {
    use std::fs;
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_path_buf();
    let rows = 2500usize;
    let dim = 4usize;
    let query = ndarray::Array2::from_shape_fn((1, dim), |(_, j)| j as f64 * 0.01);
    let mut b = BpannBackend::new_empty(path.clone(), dim, 1)
        .unwrap()
        .with_pending_flush_threshold(1000)
        .with_defer_append_indexing(true);
    for (start, count) in [(0, 1000usize), (1000, 1000usize), (2000, 500usize)] {
        let x = ndarray::Array2::from_shape_fn((count, dim), |(i, j)| (start + i + j) as f64);
        let y = ndarray::Array2::from_shape_fn((count, 1), |(i, _)| (start + i) as f64);
        b.append_rows(&x.view(), &y.view(), None).unwrap();
        b.ensure_index_sync().unwrap();
    }
    assert_eq!(b.indexed_rows(), rows);
    let (pre_dist, pre_idx) = b.search(&query.view(), 5, false).unwrap();
    b.persist_index_to_disk().unwrap();
    let (post_dist, post_idx) = b.search(&query.view(), 5, false).unwrap();
    assert_eq!(pre_idx, post_idx);
    assert_eq!(pre_dist, post_dist);
    let header_text = fs::read_to_string(path.join("index/header.json")).expect("header.json");
    assert!(
        header_text.contains("\"indexed_rows\": 2500"),
        "header: {header_text}"
    );
}

#[test]
fn persist_rewrites_corrupt_pages_with_matching_header() {
    use std::fs;
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_path_buf();
    let mut b = BpannBackend::new_empty(path.clone(), 4, 1)
        .unwrap()
        .with_pending_flush_threshold(1000)
        .with_defer_append_indexing(true);
    let x = ndarray::Array2::from_shape_fn((500, 4), |(i, j)| (i + j) as f64);
    let y = ndarray::Array2::from_shape_fn((500, 1), |(i, _)| i as f64);
    b.append_rows(&x.view(), &y.view(), None).unwrap();
    b.ensure_index_sync().unwrap();
    b.persist_index_to_disk().unwrap();
    let pages_valid = fs::read(path.join("index/pages.bin")).unwrap();

    let pages_path = path.join("index/pages.bin");
    let mut pages_corrupt = pages_valid.clone();
    for byte in pages_corrupt.iter_mut().skip(100).take(100) {
        *byte ^= 0xFF;
    }
    fs::write(&pages_path, &pages_corrupt).unwrap();
    assert_ne!(pages_valid, pages_corrupt);

    b.persist_index_to_disk().unwrap();
    let pages_healed = fs::read(&pages_path).unwrap();
    assert_eq!(pages_healed, pages_valid);

    let b2 = BpannBackend::reopen(path).unwrap();
    let query = ndarray::Array2::from_shape_fn((1, 4), |(_, j)| j as f64 * 0.01);
    b2.search(&query.view(), 5, false).unwrap();
}

#[test]
fn persist_idempotent_skips_pages_rewrite() {
    use std::fs;
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_path_buf();
    let mut b = BpannBackend::new_empty(path.clone(), 4, 1)
        .unwrap()
        .with_pending_flush_threshold(1000)
        .with_defer_append_indexing(true);
    let x = ndarray::Array2::from_shape_fn((500, 4), |(i, j)| (i + j) as f64);
    let y = ndarray::Array2::from_shape_fn((500, 1), |(i, _)| i as f64);
    b.append_rows(&x.view(), &y.view(), None).unwrap();
    b.ensure_index_sync().unwrap();
    b.persist_index_to_disk().unwrap();
    let pages_first = fs::read(path.join("index/pages.bin")).unwrap();
    b.persist_index_to_disk().unwrap();
    let pages_second = fs::read(path.join("index/pages.bin")).unwrap();
    assert_eq!(pages_first, pages_second);
}
