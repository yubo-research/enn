//! Persist-on-close hardening tests: in-session parity and idempotency.

use ennbo_bpann::BpannBackend;
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
fn soft_sync_search_metamorphic_matches_hard_persist() {
    // Soft-synced RAM index must agree with post-persist search (same rows).
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_path_buf();
    let mut soft = BpannBackend::new_empty(path.clone(), 3, 1)
        .unwrap()
        .with_pending_flush_threshold(10)
        .with_defer_append_indexing(true);
    let mut hard = BpannBackend::new_empty(path.join("hard"), 3, 1)
        .unwrap()
        .with_pending_flush_threshold(10)
        .with_defer_append_indexing(true);
    let x = ndarray::Array2::from_shape_fn((40, 3), |(i, j)| (i * 3 + j) as f64 * 0.1);
    let y = ndarray::Array2::from_shape_fn((40, 1), |(i, _)| i as f64);
    soft.append_rows(&x.view(), &y.view(), None).unwrap();
    hard.append_rows(&x.view(), &y.view(), None).unwrap();
    soft.ensure_index_sync().unwrap();
    hard.ensure_index_sync().unwrap();
    hard.persist_index_to_disk().unwrap();
    let query = ndarray::Array2::from_shape_fn((2, 3), |(i, j)| (i + j) as f64 * 0.07);
    let (d_soft, i_soft) = soft.search(&query.view(), 5, false).unwrap();
    let (d_hard, i_hard) = hard.search(&query.view(), 5, false).unwrap();
    assert_eq!(i_soft, i_hard);
    for (a, b) in d_soft.iter().zip(d_hard.iter()) {
        assert!((a - b).abs() < 1e-9, "{a} vs {b}");
    }
}

#[test]
fn soft_sync_fuzz_pending_drain_all_seeds() {
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    let seed: u64 = 0x5EED_F1A5;
    println!("soft_sync_fuzz seed={seed}");
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for trial in 0..8 {
        let dir = TempDir::new().unwrap();
        let threshold = rng.gen_range(1usize..=7);
        let batches = rng.gen_range(1usize..=5);
        let mut b = BpannBackend::new_empty(dir.path().to_path_buf(), 2, 1)
            .unwrap()
            .with_pending_flush_threshold(threshold)
            .with_defer_append_indexing(true);
        let mut total = 0usize;
        for _ in 0..batches {
            let n = rng.gen_range(1usize..=threshold.saturating_mul(2).max(2));
            let x = ndarray::Array2::from_shape_fn((n, 2), |(i, j)| {
                rng.gen::<f64>() + (total + i + j) as f64 * 0.01
            });
            let y = ndarray::Array2::from_shape_fn((n, 1), |(i, _)| (total + i) as f64);
            b.append_rows(&x.view(), &y.view(), None).unwrap();
            total += n;
            if b.pending_rows() >= threshold {
                b.ensure_index_sync().unwrap();
                assert_eq!(b.pending_rows(), 0, "trial={trial}");
            }
        }
        b.ensure_index_sync().unwrap();
        assert_eq!(b.indexed_rows(), total, "trial={trial}");
        assert!(!dir.path().join("index/pages.bin").exists());
        b.persist_index_to_disk().unwrap();
        assert!(dir.path().join("index/pages.bin").exists());
    }
}

#[test]
fn soft_sync_advances_index_without_rewriting_pages() {
    use std::fs;
    use std::hash::{Hash, Hasher};
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_path_buf();
    let mut b = BpannBackend::new_empty(path.clone(), 4, 1)
        .unwrap()
        .with_pending_flush_threshold(100)
        .with_defer_append_indexing(true);
    let pages_path = path.join("index/pages.bin");
    assert!(!pages_path.exists());

    for (start, count) in [(0, 150usize), (150, 150usize)] {
        let x = ndarray::Array2::from_shape_fn((count, 4), |(i, j)| (start + i + j) as f64);
        let y = ndarray::Array2::from_shape_fn((count, 1), |(i, _)| (start + i) as f64);
        b.append_rows(&x.view(), &y.view(), None).unwrap();
        b.ensure_index_sync().unwrap();
    }
    assert_eq!(b.indexed_rows(), 300);
    assert!(!pages_path.exists(), "soft_sync must not create pages.bin");

    let query = ndarray::Array2::from_shape_fn((1, 4), |(_, j)| j as f64 * 0.01);
    let (pre_dist, pre_idx) = b.search(&query.view(), 5, false).unwrap();
    assert!(pre_idx[[0, 0]] >= 0);

    b.persist_index_to_disk().unwrap();
    assert!(pages_path.exists());
    let pages_first = fs::read(&pages_path).unwrap();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    pages_first.hash(&mut hasher);
    let checksum = hasher.finish();

    b.ensure_index_sync().unwrap();
    let pages_after_soft = fs::read(&pages_path).unwrap();
    let mut hasher2 = std::collections::hash_map::DefaultHasher::new();
    pages_after_soft.hash(&mut hasher2);
    assert_eq!(
        checksum,
        hasher2.finish(),
        "soft_sync must not rewrite pages.bin after hard persist"
    );
    let (post_dist, post_idx) = b.search(&query.view(), 5, false).unwrap();
    assert_eq!(pre_idx, post_idx);
    assert_eq!(pre_dist, post_dist);
}

#[test]
fn soft_sync_keeps_disk_dirty_until_hard_persist() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_path_buf();
    let mut b = BpannBackend::new_empty(path.clone(), 2, 1)
        .unwrap()
        .with_pending_flush_threshold(10)
        .with_defer_append_indexing(true);
    let x = ndarray::Array2::from_shape_fn((20, 2), |(i, j)| (i + j) as f64);
    let y = ndarray::Array2::from_shape_fn((20, 1), |(i, _)| i as f64);
    b.append_rows(&x.view(), &y.view(), None).unwrap();
    b.ensure_index_sync().unwrap();
    assert_eq!(b.indexed_rows(), 20);
    // Soft-only catch-up must still require a hard rewrite (pages lag).
    b.persist_index_to_disk().unwrap();
    assert!(path.join("index/pages.bin").exists());
    // Idempotent second persist after hard write.
    let pages = std::fs::read(path.join("index/pages.bin")).unwrap();
    b.persist_index_to_disk().unwrap();
    assert_eq!(pages, std::fs::read(path.join("index/pages.bin")).unwrap());
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
