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
fn persist_drops_pages_and_reopens_from_stamp() {
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
    assert!(!path.join("index/pages.bin").exists());

    let query = ndarray::Array2::from_shape_fn((3, 4), |(i, j)| (i * 7 + j) as f64 * 0.37);
    let before = b.search(&query.view(), 5, false).unwrap();
    drop(b);
    let b2 = BpannBackend::reopen(path).unwrap();
    assert_eq!(b2.indexed_rows(), 500);
    assert_eq!(b2.search(&query.view(), 5, false).unwrap(), before);
}

/// The stamp is the commit record of a persist. A crash after the tree files are
/// written but before the stamp lands leaves no stamp and no pages.bin. Reopening must
/// then rebuild from the stored rows, never adopt a stale index, and give the same
/// neighbors as before the crash.
#[test]
fn reopen_without_stamp_rebuilds_from_rows() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_path_buf();
    let mut b = BpannBackend::new_empty(path.clone(), 4, 1)
        .unwrap()
        .with_pending_flush_threshold(1000)
        .with_defer_append_indexing(true);
    let x = ndarray::Array2::from_shape_fn((600, 4), |(i, j)| ((i * 31 + j * 17) % 97) as f64);
    let y = ndarray::Array2::from_shape_fn((600, 1), |(i, _)| i as f64);
    b.append_rows(&x.view(), &y.view(), None).unwrap();
    b.ensure_index_sync().unwrap();
    b.persist_index_to_disk().unwrap();
    let query = ndarray::Array2::from_shape_fn((4, 4), |(i, j)| (i * 13 + j * 5) as f64);
    let (_, before_idx) = b.search(&query.view(), 5, false).unwrap();
    drop(b);
    std::fs::remove_file(path.join("index/tree_state.json")).unwrap();
    assert!(!path.join("index/pages.bin").exists());

    let b2 = BpannBackend::reopen(path.clone()).unwrap();
    assert_eq!(b2.indexed_rows(), 600);
    let (_, after_idx) = b2.search(&query.view(), 5, false).unwrap();
    let as_sets = |m: &ndarray::Array2<i64>| -> Vec<Vec<i64>> {
        m.rows()
            .into_iter()
            .map(|r| {
                let mut v = r.to_vec();
                v.sort_unstable();
                v
            })
            .collect()
    };
    assert_eq!(as_sets(&before_idx), as_sets(&after_idx));
}

#[test]
fn soft_sync_search_metamorphic_matches_hard_persist() {
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
        assert!(!dir.path().join("index/pages.bin").exists());
        assert!(dir.path().join("index/tree_state.json").is_file());
    }
}

#[test]
fn soft_sync_advances_index_without_rewriting_pages() {
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
    assert!(!pages_path.exists(), "a stamped persist writes no pages.bin");

    b.ensure_index_sync().unwrap();
    assert!(!pages_path.exists(), "soft_sync must not write pages.bin");
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

    b.persist_index_to_disk().unwrap();
    assert!(!path.join("index/pages.bin").exists());

    let stamp = std::fs::read(path.join("index/tree_state.json")).unwrap();
    b.persist_index_to_disk().unwrap();
    assert_eq!(stamp, std::fs::read(path.join("index/tree_state.json")).unwrap());
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
    let snapshot = || -> Vec<Vec<u8>> {
        ["header.json", "tree_state.json", "tree_meta.bin", "tree_slots.bin", "tree_codes.bin"]
            .iter()
            .map(|f| fs::read(path.join("index").join(f)).unwrap())
            .collect()
    };
    let first = snapshot();
    b.persist_index_to_disk().unwrap();
    assert_eq!(first, snapshot(), "a second persist must not change any index file");
}
