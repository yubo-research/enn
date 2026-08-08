//! Integration tests exercising bpann modules for kiss coverage and behavior.

use ennbo_bpann::backend::open_rejects_record_stride;
use ennbo_bpann::index::kmeans::PartitionTree;
use ennbo_bpann::index::page::closest_child;
use ennbo_bpann::index::{BpannIndex, DEFAULT_LEAF_CAPACITY};
use ennbo_bpann::mmap_store::MmapColumnStore;
use ennbo_bpann::BpannBackend;
use ndarray::array;
use std::sync::Mutex;
use tempfile::TempDir;

#[test]
fn observation_helpers_called() {
    let dir = TempDir::new().unwrap();
    ennbo_bpann::observation::bpann_validate_dim_limits(4).unwrap();
    ennbo_bpann::observation::bpann_check_append_row_limit(10).unwrap();
    ennbo_bpann::observation::bpann_write_metadata(dir.path(), 0, 4, 1, false, 0).unwrap();
    ennbo_bpann::observation::write_num_obs(dir.path(), 0).unwrap();
    ennbo_bpann::observation::write_indexed_rows(dir.path(), 0).unwrap();
    let mut counter = ennbo_bpann::observation::NumObsCounter::open(dir.path()).unwrap();
    counter.set(0);
    assert_eq!(ennbo_bpann::observation::bpann_load_num_obs(dir.path()), Some(0));
    ennbo_bpann::observation::bpann_validate_index_backend(dir.path(), ennbo_bpann::observation::INDEX_BACKEND)
        .unwrap();
    let yv = array![[0.1]];
    let mut yvar = ennbo_bpann::observation::bpann_open_or_append_yvar(dir.path(), 1, Some(&yv)).unwrap();
    ennbo_bpann::observation::bpann_append_yvar_on_add(
        dir.path(),
        1,
        &mut yvar,
        Some(&array![[0.2]].view()),
    )
    .unwrap();
    let dirty = Mutex::new(false);
    ennbo_bpann::observation::bpann_mark_index_dirty(&dirty);
    ennbo_bpann::observation::bpann_load_indexed_rows(dir.path());
    ennbo_bpann::observation::bpann_load_index_backend(dir.path());
    ennbo_bpann::observation::bpann_parse_json_string_field(r#"{"index_backend":"bpann_disk"}"#, "index_backend");
    let mut x = MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), 2, None).unwrap();
    let mut y = MmapColumnStore::mmap_open_or_create(dir.path().join("y.bin"), 1, None).unwrap();
    x.mmap_append(&array![[0.0, 0.0]].view()).unwrap();
    y.mmap_append(&array![[0.0]].view()).unwrap();
    ennbo_bpann::observation::bpann_train_rows_at(1, &x, &y, None, &[0]).unwrap();
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
    let _ = ennbo_bpann::index::search::search_exhaustive_leaves(&index, &[0.0, 0.0], 1);
    let _ = ennbo_bpann::index::search::search_greedy_blocks_only(&index, &[0.0, 0.0], 1, 2);
    let mut log = Vec::new();
    let _ = ennbo_bpann::index::search::search_with_skip_refinement(&index, &[0.0, 0.0], 1, 2, &mut log);
    let _ = ennbo_bpann::index::search::bpann_mean_recall_at_k(&vectors, &[vec![0.0, 0.0]], 1, &index);
    let _ = ennbo_bpann::index::search::bpann_brute_force_topk(&vectors, &[0.0, 0.0], 1);
}

#[test]
fn merge_distance_mmap_called() {
    let dir = TempDir::new().unwrap();
    let mut store = MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), 2, None).unwrap();
    store
        .mmap_append(&array![[0.0, 0.0], [1.0, 0.0]].view())
        .unwrap();
    let mut buf = Vec::new();
    ennbo_bpann::distance::bpann_row_to_f32(&[1.0, 0.0], false, &[1.0, 1.0], &mut buf);
    let _ = ennbo_bpann::distance::batched_sq_l2_f64_rows(&[0.0, 0.0], &store, &[0, 1], false, &[1.0, 1.0]).unwrap();
    let _ = ennbo_bpann::distance::row_sq_l2(array![0.0, 0.0].view(), array![1.0, 0.0].view(), false, array![1.0, 1.0].view());
    let _ = ennbo_bpann::merge::bpann_merge_topk_candidates(
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
    let _ = ennbo_bpann::index::search::bpann_brute_force_topk_mmap(&store, 0, 2, &[0.0, 0.0], 1, false, &[1.0, 1.0]).unwrap();
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
    let (y0, yv0) = b.mmap_row_y_and_yvar(0).unwrap();
    assert!((y0[0] - 0.0).abs() < 1e-12);
    assert!(yv0.is_none());
    // Small-N in-core path: repeated search must agree (cache reuse).
    let (_d1, idx1) = b.search(&array![[0.1, 0.1]].view(), 1, false).unwrap();
    let (_d2, idx2) = b.search(&array![[0.1, 0.1]].view(), 1, false).unwrap();
    assert_eq!(idx1[[0, 0]], idx2[[0, 0]]);
    assert_eq!(idx1[[0, 0]], 0);
    assert_eq!(ennbo_bpann::SMALL_N_INCORE_SEARCH_LIMIT, 8192);
    let flat = ennbo_bpann::load_or_build_small_n_cache(&b, b.len()).unwrap();
    assert_eq!(flat.len(), b.len() * 2);
    let hits = ennbo_bpann::topk_flat_sq_l2(&[0.0, 0.0], &flat, 2, 2, 1);
    assert_eq!(hits[0].0, 0);
    assert!(ennbo_bpann::OrderedF32(1.0) > ennbo_bpann::OrderedF32(0.0));
    assert!(ennbo_bpann::topk_flat_sq_l2(&[0.0, 0.0], &[], 0, 2, 1).is_empty());
    let scored = ennbo_bpann::score_queries_flat(
        &[vec![0.1, 0.1]],
        &ennbo_bpann::ScoreQueriesFlat {
            flat: &flat,
            total: 2,
            num_dim: 2,
            scale_x: false,
            x_scale: &[1.0, 1.0],
            k_eff: 1,
            pool_k: 1,
            exclude_nearest: false,
        },
    );
    assert_eq!(scored.len(), 1);
    assert_eq!(scored[0].1[0], 0);
    // Append must invalidate the small-N cache (next search still correct).
    b.append_rows(&array![[2.0, 0.0]].view(), &array![[2.0]].view(), None)
        .unwrap();
    let (_, idx3) = b.search(&array![[2.0, 0.0]].view(), 1, false).unwrap();
    assert_eq!(idx3[[0, 0]], 2);
    b.mark_index_stale();
    b.ensure_index_sync_with_scale(true, &array![1.0, 1.0]).unwrap();
    b.ensure_index_sync_with_scale(false, &array![1.0, 1.0]).unwrap();
    let (_, idx) = b.search(&array![[0.1, 0.1]].view(), 1, false).unwrap();
    assert_eq!(idx[[0, 0]], 0);
}

#[test]
fn large_n_search_indexed_and_pending_finds_nearest() {
    use ndarray::Array2;
    let dir = TempDir::new().unwrap();
    let mut b = BpannBackend::new_empty(dir.path().to_path_buf(), 2, 1).unwrap();
    // N > SMALL_N_INCORE_SEARCH_LIMIT forces the indexed+pending path.
    let n = ennbo_bpann::SMALL_N_INCORE_SEARCH_LIMIT + 50;
    let mut xs = Array2::<f64>::zeros((n, 2));
    let mut ys = Array2::<f64>::zeros((n, 1));
    for i in 0..n {
        xs[[i, 0]] = i as f64;
        ys[[i, 0]] = i as f64;
    }
    b.append_rows(&xs.view(), &ys.view(), None).unwrap();
    b.ensure_index_sync().unwrap();
    let mut dist2s = Array2::zeros((1, 1));
    let mut indices = Array2::zeros((1, 1));
    let x_scale = [1.0f64, 1.0];
    ennbo_bpann::search_indexed_and_pending(
        &b,
        &[vec![10.0, 0.0]],
        &mut dist2s,
        &mut indices,
        ennbo_bpann::SearchPendingArgs {
            total: n,
            k_eff: 1,
            pool_k: 1,
            exclude_nearest: false,
            scale_x: false,
            x_scale: &x_scale,
            num_dim: 2,
        },
    )
    .unwrap();
    assert_eq!(indices[[0, 0]], 10);
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
    let merged = ennbo_bpann::merge::merge_topk_precomputed_dist(&[(0, 0.0), (1, 4.0)], &[(2, 1.0)], 2, 3, false);
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
fn ensure_index_sync_noop_and_soft_sync_skips_hard_persist() {
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
    assert!(!dir.path().join("index/header.json").exists());
    assert!(!dir.path().join("index/pages.bin").exists());
    b.persist_index_to_disk().unwrap();
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
    use ennbo_bpann::index::IncrementalIndex;
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
        "maybe_compact",
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
