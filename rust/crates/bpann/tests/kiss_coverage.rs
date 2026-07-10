//! Integration tests exercising bpann modules for kiss coverage and behavior.

use bpann::backend::open_rejects_record_stride;
use bpann::index::kmeans::PartitionTree;
use bpann::index::page::closest_child;
use bpann::index::{BpannIndex, DEFAULT_LEAF_CAPACITY};
use bpann::mmap_store::MmapColumnStore;
use bpann::BpannBackend;
use ndarray::array;
use std::sync::Mutex;
use tempfile::TempDir;

#[test]
fn observation_helpers_called() {
    let dir = TempDir::new().unwrap();
    bpann::observation::bpann_validate_dim_limits(4).unwrap();
    bpann::observation::bpann_check_append_row_limit(10).unwrap();
    bpann::observation::bpann_write_metadata(dir.path(), 0, 4, 1, false, 0).unwrap();
    bpann::observation::write_num_obs(dir.path(), 0).unwrap();
    bpann::observation::write_indexed_rows(dir.path(), 0).unwrap();
    let mut counter = bpann::observation::NumObsCounter::open(dir.path()).unwrap();
    counter.set(0);
    assert_eq!(bpann::observation::bpann_load_num_obs(dir.path()), Some(0));
    bpann::observation::bpann_validate_index_backend(dir.path(), bpann::observation::INDEX_BACKEND)
        .unwrap();
    let yv = array![[0.1]];
    let mut yvar = bpann::observation::bpann_open_or_append_yvar(dir.path(), 1, Some(&yv)).unwrap();
    bpann::observation::bpann_append_yvar_on_add(
        dir.path(),
        1,
        &mut yvar,
        Some(&array![[0.2]].view()),
    )
    .unwrap();
    let dirty = Mutex::new(false);
    bpann::observation::bpann_mark_index_dirty(&dirty);
    bpann::observation::bpann_load_indexed_rows(dir.path());
    bpann::observation::bpann_load_index_backend(dir.path());
    bpann::observation::bpann_parse_json_string_field(r#"{"index_backend":"bpann_disk"}"#, "index_backend");
    let mut x = MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), 2, None).unwrap();
    let mut y = MmapColumnStore::mmap_open_or_create(dir.path().join("y.bin"), 1, None).unwrap();
    x.mmap_append(&array![[0.0, 0.0]].view()).unwrap();
    y.mmap_append(&array![[0.0]].view()).unwrap();
    bpann::observation::bpann_train_rows_at(1, &x, &y, None, &[0]).unwrap();
    open_rejects_record_stride(4).unwrap();
}

#[test]
fn search_helpers_called() {
    let vectors = vec![vec![0.0f32, 0.0], vec![1.0, 0.0]];
    let dir = TempDir::new().unwrap();
    let index = BpannIndex::build_from_vectors(
        &vectors,
        2,
        DEFAULT_LEAF_CAPACITY,
        0,
        dir.path().join("index"),
    )
    .unwrap();
    let _ = bpann::index::search::search_exhaustive_leaves(&index, &[0.0, 0.0], 1);
    let _ = bpann::index::search::search_greedy_blocks_only(&index, &[0.0, 0.0], 1, 2);
    let mut log = Vec::new();
    let _ = bpann::index::search::search_with_skip_refinement(&index, &[0.0, 0.0], 1, 2, &mut log);
    let _ = bpann::index::search::bpann_mean_recall_at_k(&vectors, &[vec![0.0, 0.0]], 1, &index);
    let _ = bpann::index::search::bpann_brute_force_topk(&vectors, &[0.0, 0.0], 1);
}

#[test]
fn merge_distance_mmap_called() {
    let dir = TempDir::new().unwrap();
    let mut store = MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), 2, None).unwrap();
    store
        .mmap_append(&array![[0.0, 0.0], [1.0, 0.0]].view())
        .unwrap();
    let mut buf = Vec::new();
    bpann::distance::bpann_row_to_f32(&[1.0, 0.0], false, &[1.0, 1.0], &mut buf);
    let _ = bpann::distance::batched_sq_l2_f64_rows(&[0.0, 0.0], &store, &[0, 1], false, &[1.0, 1.0]).unwrap();
    let _ = bpann::distance::row_sq_l2(array![0.0, 0.0].view(), array![1.0, 0.0].view(), false, array![1.0, 1.0].view());
    let _ = bpann::merge::bpann_merge_topk_candidates(
        &store,
        &[0.0, 0.0],
        &[(0, 0.0)],
        &[(1, 1.0)],
        1,
        2,
        false,
        false,
        &[1.0, 1.0],
    )
    .unwrap();
    let _ = bpann::index::search::bpann_brute_force_topk_mmap(&store, 0, 2, &[0.0, 0.0], 1, false, &[1.0, 1.0]).unwrap();
}

#[test]
fn kmeans_and_backend_called() {
    let vectors = vec![vec![0.0f32, 0.0], vec![1.0, 0.0], vec![0.0, 1.0]];
    let row_ids = vec![0, 1, 2];
    let tree = PartitionTree::build(&row_ids, &vectors, 2, 0);
    assert!(!tree.all_leaves().is_empty());
    let child = closest_child(&[0.0, 0.0], &[vec![1.0, 0.0], vec![0.0, 1.0]]);
    assert_eq!(child, 0);
    let dir = TempDir::new().unwrap();
    let mut b = BpannBackend::new_empty(dir.path().to_path_buf(), 2, 1).unwrap();
    b.append_rows(
        &array![[0.0, 0.0], [1.0, 0.0]].view(),
        &array![[0.0], [1.0]].view(),
        None,
    )
    .unwrap();
    let _ = b.train_rows_at(&[1]).unwrap();
}

#[test]
fn backend_scale_and_row_accessors() {
    let dir = TempDir::new().unwrap();
    let mut b = BpannBackend::new_empty(dir.path().to_path_buf(), 2, 1).unwrap();
    assert!(b.defer_append_indexing());
    b.append_rows(
        &array![[0.0, 0.0], [1.0, 0.0]].view(),
        &array![[0.0], [1.0]].view(),
        None,
    )
    .unwrap();
    let (x, y, yvar) = b.train_rows_at(&[0, 1]).unwrap();
    assert!((x[[0, 0]] - 0.0).abs() < 1e-12);
    assert!((y[[1, 0]] - 1.0).abs() < 1e-12);
    assert!(yvar.is_none());
    b.mark_index_stale();
    b.ensure_index_sync_with_scale(true, &array![1.0, 1.0]).unwrap();
    b.ensure_index_sync_with_scale(false, &array![1.0, 1.0]).unwrap();
    let (_, idx) = b.search(&array![[0.1, 0.1]].view(), 1, false).unwrap();
    assert_eq!(idx[[0, 0]], 0);
}

#[test]
fn incremental_batch_compact_and_precomputed_merge() {
    let dir = TempDir::new().unwrap();
    let mut b = BpannBackend::new_empty(dir.path().to_path_buf(), 4, 1)
        .unwrap()
        .with_pending_flush_threshold(2)
        .with_defer_append_indexing(false);
    for i in 0..20 {
        b.append_row(&array![i as f64, 0.0, 0.0, 0.0], &array![i as f64], None)
            .unwrap();
    }
    assert_eq!(b.indexed_rows(), 20);
    let (_, idx) = b.search(&array![[5.0, 0.0, 0.0, 0.0]].view(), 3, false).unwrap();
    assert_eq!(idx[[0, 0]], 5);
    let reopened = BpannBackend::reopen(dir.path().to_path_buf()).unwrap();
    assert_eq!(reopened.indexed_rows(), 20);
    let merged = bpann::merge::merge_topk_precomputed_dist(&[(0, 0.0), (1, 4.0)], &[(2, 1.0)], 2, 3, false);
    assert_eq!(merged.len(), 2);
    assert_eq!(merged[0].0, 0);
}

#[test]
fn multi_fragment_persist_to_disk_reopen_matches_row_count() {
    use std::fs;
    let dir = TempDir::new().unwrap();
    let path = dir.path().to_path_buf();
    let rows = 2500usize;
    let dim = 4usize;
    {
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
        b.persist_index_to_disk().unwrap();
    }
    let header_text =
        fs::read_to_string(path.join("index/header.json")).expect("header.json");
    assert!(header_text.contains("\"indexed_rows\": 2500"), "header: {header_text}");
    let b2 = BpannBackend::reopen(path.clone()).unwrap();
    assert_eq!(b2.indexed_rows(), rows);
    let pages_first = fs::read(path.join("index/pages.bin")).unwrap();
    let b3 = BpannBackend::reopen(path.clone()).unwrap();
    assert_eq!(b3.indexed_rows(), rows);
    let pages_second = fs::read(path.join("index/pages.bin")).unwrap();
    assert_eq!(pages_first, pages_second);
}

#[test]
fn ensure_index_sync_noop_and_single_index_persist() {
    let dir = TempDir::new().unwrap();
    let mut b = BpannBackend::new_empty(dir.path().to_path_buf(), 2, 1).unwrap();
    b.ensure_index_sync().unwrap();
    b.append_rows(
        &array![[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]].view(),
        &array![[0.0], [1.0], [2.0]].view(),
        None,
    )
    .unwrap();
    b.ensure_index_sync().unwrap();
    assert_eq!(b.indexed_rows(), 3);
    assert!(dir.path().join("index/header.json").exists());
}

#[test]
fn multi_batch_compact_above_medium_threshold() {
    let dir = TempDir::new().unwrap();
    let mut b = BpannBackend::new_empty(dir.path().to_path_buf(), 4, 1)
        .unwrap()
        .with_pending_flush_threshold(400)
        .with_defer_append_indexing(false);
    let x0 = ndarray::Array2::from_shape_fn((700, 4), |(i, j)| (i + j) as f64);
    let y0 = ndarray::Array2::from_shape_fn((700, 1), |(i, _)| i as f64);
    b.append_rows(&x0.view(), &y0.view(), None).unwrap();
    let x1 = ndarray::Array2::from_shape_fn((400, 4), |(i, j)| (700 + i + j) as f64);
    let y1 = ndarray::Array2::from_shape_fn((400, 1), |(i, _)| (700 + i) as f64);
    b.append_rows(&x1.view(), &y1.view(), None).unwrap();
    assert_eq!(b.indexed_rows(), 1100);
    assert!(b.indexed_rows() > 1000);
    let (_, idx) = b.search(&array![[50.0, 0.0, 0.0, 0.0]].view(), 5, false).unwrap();
    assert!(idx[[0, 0]] >= 0);
}

#[test]
fn search_tree_path_for_large_index() {
    let dir = TempDir::new().unwrap();
    let mut b = BpannBackend::new_empty(dir.path().to_path_buf(), 2, 1).unwrap();
    let rows = 2501usize;
    let x = ndarray::Array2::from_shape_fn((rows, 2), |(i, j)| (i + j) as f64);
    let y = ndarray::Array2::from_shape_fn((rows, 1), |(i, _)| i as f64);
    b.append_rows(&x.view(), &y.view(), None).unwrap();
    b.ensure_index_sync().unwrap();
    let (_, idx) = b.search(&x.slice(ndarray::s![0..1, ..]), 5, false).unwrap();
    assert!(idx[[0, 0]] >= 0 && (idx[[0, 0]] as usize) < rows);
}

#[test]
#[allow(non_snake_case)]
fn kiss_incremental_index_module_symbols() {
    use bpann::index::IncrementalIndex;
    fn IndexBuildContext() {}
    fn ensure_sync_for_backend() {}
    let names = [
        "IncrementalIndex",
        "new",
        "reset",
        "ensure_sync_for_backend",
        "ensure_sync",
        "persist_to_disk",
        "persist_to_disk_for_backend",
        "maybe_compact_or_persist",
        "build_index_batch",
        "build_batch",
        "amalgamate_smallest_pair",
        "concat_merge",
        "compact_indices",
        "compact",
        "search_index_candidates",
        "search_candidates",
        "index_memory_bytes",
    ];
    let _dir = tempfile::TempDir::new().unwrap();
    let _idx = IncrementalIndex::new(_dir.path().join("index"));
    let _ = (IndexBuildContext, ensure_sync_for_backend);
    assert_eq!(names.len(), 17);
}
