//! Integration tests for searchable pending buffer (dual-leg search, deferred sync).

use ennbo::disk_hnsw::DiskHnswEnnBackend;
use ennbo::disk_hnsw::brute_force_topk;
use ennbo::index::IndexDriver;
use ndarray::{Array1, Array2, array};
use std::fs;
use tempfile::TempDir;

fn exact_brute_topk_ids(
    backend: &DiskHnswEnnBackend,
    query: &[f64],
    k: usize,
    exclude_nearest: bool,
) -> Vec<i64> {
    let n = backend.len();
    let mut scored: Vec<(i64, f64)> = (0..n)
        .map(|i| {
            let row = backend.row_x(i).unwrap();
            let mut acc = 0.0;
            for (&q, &r) in query.iter().zip(row.iter()) {
                let d = q - r;
                acc += d * d;
            }
            (i as i64, acc)
        })
        .collect();
    scored.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap()
            .then_with(|| a.0.cmp(&b.0))
    });
    if exclude_nearest && scored.len() > 1 {
        scored.remove(0);
    }
    scored.truncate(k);
    scored.into_iter().map(|(id, _)| id).collect()
}

#[test]
fn disk_hnsw_search_includes_pending_without_sync() {
    let dir = TempDir::new().expect("tempdir");
    let mut backend = DiskHnswEnnBackend::new(
        dir.path().to_path_buf(),
        array![[0.0, 0.0], [1.0, 0.0]],
        array![[0.0], [1.0]],
        None,
        false,
        Array1::ones(2),
        IndexDriver::HNSWDisk,
    )
    .unwrap();
    backend.ensure_index_sync(false, &Array1::ones(2)).unwrap();
    assert_eq!(backend.indexed_rows(), 2);
    backend
        .append_rows(
            &array![[100.0, 100.0]].view(),
            &array![[2.0]].view(),
            None,
        )
        .unwrap();
    let (_, idx) = backend
        .search(&array![[100.0, 100.0]].view(), 1, false)
        .unwrap();
    assert_eq!(idx[[0, 0]], 2);
    assert_eq!(backend.indexed_rows(), 2);
}

#[test]
fn disk_hnsw_search_no_index_build_on_query() {
    let dir = TempDir::new().expect("tempdir");
    let mut backend = DiskHnswEnnBackend::new(
        dir.path().to_path_buf(),
        array![[0.0, 0.0], [1.0, 0.0]],
        array![[0.0], [1.0]],
        None,
        false,
        Array1::ones(2),
        IndexDriver::HNSWDisk,
    )
    .unwrap();
    backend.ensure_index_sync(false, &Array1::ones(2)).unwrap();
    backend
        .append_rows(
            &array![[2.0, 2.0]].view(),
            &array![[2.0]].view(),
            None,
        )
        .unwrap();
    let nodes_path = dir.path().join("graph/nodes.bin");
    let size_before = fs::metadata(&nodes_path).map(|m| m.len()).unwrap_or(0);
    let indexed_before = backend.indexed_rows();
    backend
        .search(&array![[0.5, 0.5]].view(), 1, false)
        .unwrap();
    assert_eq!(backend.indexed_rows(), indexed_before);
    let size_after = fs::metadata(&nodes_path).map(|m| m.len()).unwrap_or(0);
    assert_eq!(size_before, size_after);
}

#[test]
fn disk_hnsw_search_mixed_matches_brute_force() {
    let dir = TempDir::new().expect("tempdir");
    let mut backend = DiskHnswEnnBackend::new(
        dir.path().to_path_buf(),
        array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        array![[0.0], [1.0], [2.0]],
        None,
        false,
        Array1::ones(2),
        IndexDriver::HNSWDisk,
    )
    .unwrap();
    backend.ensure_index_sync(false, &Array1::ones(2)).unwrap();
    backend
        .append_rows(
            &array![[2.0, 2.0], [3.0, 0.0]].view(),
            &array![[3.0], [4.0]].view(),
            None,
        )
        .unwrap();
    let query = [0.9, 0.9];
    let (_, idx) = backend.search(&array![query].view(), 2, false).unwrap();
    let expected = exact_brute_topk_ids(&backend, &query, 2, false);
    assert_eq!(idx.row(0).to_vec(), expected);
}

#[test]
fn disk_hnsw_search_all_pending_brute_only() {
    let dir = TempDir::new().expect("tempdir");
    let mut backend = DiskHnswEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1).unwrap();
    backend
        .append_rows(
            &array![[0.0, 0.0], [1.0, 0.0]].view(),
            &array![[0.0], [1.0]].view(),
            None,
        )
        .unwrap();
    assert_eq!(backend.indexed_rows(), 0);
    let (_, idx) = backend
        .search(&array![[0.1, 0.0]].view(), 1, false)
        .unwrap();
    assert_eq!(idx[[0, 0]], 0);
}

#[test]
fn disk_hnsw_search_exclude_nearest_merged() {
    let dir = TempDir::new().expect("tempdir");
    let mut backend = DiskHnswEnnBackend::new(
        dir.path().to_path_buf(),
        array![[0.0, 0.0], [1.0, 0.0]],
        array![[0.0], [1.0]],
        None,
        false,
        Array1::ones(2),
        IndexDriver::HNSWDisk,
    )
    .unwrap();
    backend.ensure_index_sync(false, &Array1::ones(2)).unwrap();
    backend
        .append_rows(
            &array![[0.01, 0.01]].view(),
            &array![[2.0]].view(),
            None,
        )
        .unwrap();
    let query = [0.0, 0.0];
    let (_, idx) = backend.search(&array![query].view(), 1, true).unwrap();
    let expected = exact_brute_topk_ids(&backend, &query, 1, true);
    assert_eq!(idx.row(0).to_vec(), expected);
    assert_ne!(idx[[0, 0]], 0);
}

#[test]
fn disk_hnsw_reopen_searches_pending_without_sync() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().to_path_buf();
    let mut backend = DiskHnswEnnBackend::new(
        path.clone(),
        array![[0.0, 0.0], [1.0, 0.0]],
        array![[0.0], [1.0]],
        None,
        false,
        Array1::ones(2),
        IndexDriver::HNSWDisk,
    )
    .unwrap();
    backend.ensure_index_sync(false, &Array1::ones(2)).unwrap();
    backend
        .append_rows(
            &array![[5.0, 5.0]].view(),
            &array![[2.0]].view(),
            None,
        )
        .unwrap();
    drop(backend);
    let reopened = DiskHnswEnnBackend::new(
        path,
        Array2::zeros((0, 2)),
        Array2::zeros((0, 1)),
        None,
        false,
        Array1::ones(2),
        IndexDriver::HNSWDisk,
    )
    .unwrap();
    assert_eq!(reopened.indexed_rows(), 2);
    assert_eq!(reopened.len(), 3);
    let (_, idx) = reopened
        .search(&array![[5.0, 5.0]].view(), 1, false)
        .unwrap();
    assert_eq!(idx[[0, 0]], 2);
}

#[test]
fn disk_hnsw_new_empty_default_threshold_is_1000() {
    let dir = TempDir::new().expect("tempdir");
    let backend = DiskHnswEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1).unwrap();
    assert_eq!(backend.pending_flush_threshold(), 1000);
}

#[test]
fn disk_hnsw_append_past_threshold_does_not_sync() {
    let dir = TempDir::new().expect("tempdir");
    let mut backend = DiskHnswEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1)
        .unwrap()
        .with_pending_flush_threshold(3)
        .with_defer_append_indexing(true);
    let nodes_path = dir.path().join("graph/nodes.bin");
    let size_before = fs::metadata(&nodes_path).map(|m| m.len()).unwrap_or(0);
    for i in 0..3 {
        backend
            .append_rows(
                &array![[i as f64, 0.0]].view(),
                &array![[i as f64]].view(),
                None,
            )
            .unwrap();
    }
    assert_eq!(backend.len(), 3);
    assert!(backend.indexed_rows() < backend.len());
    let size_after = fs::metadata(&nodes_path).map(|m| m.len()).unwrap_or(0);
    assert_eq!(size_before, size_after);
}

#[test]
fn disk_hnsw_defer_sync_when_pending_at_or_above_threshold() {
    let dir = TempDir::new().expect("tempdir");
    let mut backend = DiskHnswEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1)
        .unwrap()
        .with_pending_flush_threshold(3)
        .with_defer_append_indexing(true);
    for i in 0..3 {
        backend
            .append_rows(
                &array![[i as f64, 0.0]].view(),
                &array![[i as f64]].view(),
                None,
            )
            .unwrap();
    }
    assert!(backend.defer_index_sync_for_search());
    let nodes_path = dir.path().join("graph/nodes.bin");
    let size_before = fs::metadata(&nodes_path).map(|m| m.len()).unwrap_or(0);
    let (_, idx) = backend
        .search(&array![[1.0, 0.0]].view(), 1, false)
        .unwrap();
    assert_eq!(idx[[0, 0]], 1);
    let size_after = fs::metadata(&nodes_path).map(|m| m.len()).unwrap_or(0);
    assert_eq!(size_before, size_after);
}

#[test]
fn disk_hnsw_explicit_sync_flushes_below_threshold() {
    let dir = TempDir::new().expect("tempdir");
    let mut backend = DiskHnswEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1)
        .unwrap()
        .with_pending_flush_threshold(100);
    backend
        .append_rows(
            &array![[0.0, 0.0], [1.0, 0.0]].view(),
            &array![[0.0], [1.0]].view(),
            None,
        )
        .unwrap();
    assert_eq!(backend.indexed_rows(), 0);
    backend.ensure_index_sync(false, &Array1::ones(2)).unwrap();
    assert_eq!(backend.indexed_rows(), backend.len());
}

#[test]
fn disk_hnsw_stale_forces_sync_on_search() {
    let dir = TempDir::new().expect("tempdir");
    let mut backend = DiskHnswEnnBackend::new(
        dir.path().to_path_buf(),
        array![[0.0, 0.0], [1.0, 0.0]],
        array![[0.0], [1.0]],
        None,
        false,
        Array1::ones(2),
        IndexDriver::HNSWDisk,
    )
    .unwrap();
    backend.ensure_index_sync(false, &Array1::ones(2)).unwrap();
    backend
        .append_rows(
            &array![[2.0, 2.0]].view(),
            &array![[2.0]].view(),
            None,
        )
        .unwrap();
    backend.mark_index_stale();
    assert!(!backend.defer_index_sync_for_search());
    assert!(backend.indexed_rows() < backend.len());

    // Model search path: sync when defer is false, then search.
    if !backend.defer_index_sync_for_search() {
        backend
            .ensure_index_sync(false, &Array1::ones(2))
            .unwrap();
    }
    backend
        .search(&array![[0.5, 0.5]].view(), 1, false)
        .unwrap();
    assert_eq!(backend.indexed_rows(), backend.len());
}

#[test]
fn disk_hnsw_scale_x_pending_leg_live_scale() {
    let dir = TempDir::new().expect("tempdir");
    let mut backend = DiskHnswEnnBackend::new(
        dir.path().to_path_buf(),
        array![[2.0, 4.0]],
        array![[0.0]],
        None,
        true,
        Array1::from_elem(2, 2.0),
        IndexDriver::HNSWDisk,
    )
    .unwrap();
    backend
        .append_rows(
            &array![[4.0, 8.0]].view(),
            &array![[1.0]].view(),
            None,
        )
        .unwrap();
    let query = [2.0, 2.0];
    let (_, idx) = backend.search(&array![query].view(), 1, false).unwrap();
    let vecs: Vec<Vec<f32>> = (0..2)
        .map(|i| {
            backend
                .row_x(i)
                .unwrap()
                .iter()
                .map(|&v| (v / 2.0) as f32)
                .collect()
        })
        .collect();
    let q: Vec<f32> = query.iter().map(|&v| (v / 2.0) as f32).collect();
    let bf = brute_force_topk(&vecs, &q, 1);
    assert_eq!(idx[[0, 0]] as u32, bf[0].0);
}

#[test]
fn disk_hnsw_append_syncs_at_threshold_flag() {
    let dir = TempDir::new().expect("tempdir");
    let defer_backend = DiskHnswEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1).unwrap();
    assert!(!defer_backend.append_syncs_at_threshold());
    let sync_backend = DiskHnswEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1)
        .unwrap()
        .with_defer_append_indexing(false);
    assert!(sync_backend.append_syncs_at_threshold());
}

#[test]
fn disk_hnsw_ensure_sync_persists_when_already_indexed() {
    let dir = TempDir::new().expect("tempdir");
    let mut backend = DiskHnswEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1)
        .unwrap()
        .with_pending_flush_threshold(2)
        .with_defer_append_indexing(false);
    backend
        .append_rows(
            &array![[0.0, 0.0], [1.0, 0.0]].view(),
            &array![[0.0], [1.0]].view(),
            None,
        )
        .unwrap();
    assert_eq!(backend.indexed_rows(), 2);
    backend.ensure_index_sync(false, &Array1::ones(2)).unwrap();
    assert!(dir.path().join("graph/header.json").exists());
}

#[test]
fn disk_hnsw_search_fast_path_fully_indexed() {
    let dir = TempDir::new().expect("tempdir");
    let mut backend = DiskHnswEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1)
        .unwrap()
        .with_pending_flush_threshold(2)
        .with_defer_append_indexing(false);
    backend
        .append_rows(
            &array![[0.0, 0.0], [1.0, 0.0], [2.0, 2.0]].view(),
            &array![[0.0], [1.0], [2.0]].view(),
            None,
        )
        .unwrap();
    let (dist2s, idx) = backend
        .search(&array![[0.1, 0.1]].view(), 1, false)
        .unwrap();
    assert_eq!(idx[[0, 0]], 0);
    assert!(dist2s[[0, 0]] >= 0.0);
}

#[test]
fn disk_hnsw_sync_append_indexes_with_scale_x() {
    let dir = TempDir::new().expect("tempdir");
    let mut backend = DiskHnswEnnBackend::new(
        dir.path().to_path_buf(),
        array![[2.0, 4.0]],
        array![[0.0]],
        None,
        true,
        Array1::from_elem(2, 2.0),
        IndexDriver::HNSWDisk,
    )
    .unwrap();
    backend.set_defer_append_indexing(false);
    backend.set_pending_flush_threshold(2);
    backend
        .append_rows(
            &array![[4.0, 8.0], [6.0, 12.0]].view(),
            &array![[1.0], [2.0]].view(),
            None,
        )
        .unwrap();
    assert_eq!(backend.indexed_rows(), 3);
}

#[test]
fn disk_hnsw_open_graph_rejects_dim_mismatch() {
    let dir = TempDir::new().expect("tempdir");
    let _ = DiskHnswEnnBackend::new(
        dir.path().to_path_buf(),
        array![[0.0, 0.0]],
        array![[0.0]],
        None,
        false,
        Array1::ones(2),
        IndexDriver::HNSWDisk,
    )
    .unwrap();
    let err = DiskHnswEnnBackend::new(
        dir.path().to_path_buf(),
        array![[0.0, 0.0, 0.0]],
        array![[0.0]],
        None,
        false,
        Array1::ones(3),
        IndexDriver::HNSWDisk,
    );
    assert!(err.is_err());
}
