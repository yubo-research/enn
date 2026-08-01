use bpann::merge::{
    bpann_apply_exclude_nearest, bpann_merge_topk_candidates, find_query_train_id,
    find_query_train_id_flat, merge_topk_precomputed_dist, merge_topk_precomputed_dist_with_self,
};
use bpann::mmap_store::MmapColumnStore;
use ndarray::array;
use tempfile::TempDir;

#[test]
fn test_bpann_merge_topk_candidates_excludes_self() {
    let dir = TempDir::new().unwrap();
    let mut store = MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), 2, None).unwrap();
    store
        .mmap_append(&array![[0.0, 0.0], [1.0, 0.0]].view())
        .unwrap();
    let merged = bpann_merge_topk_candidates(
        &store,
        &[0.0, 0.0],
        &[(0, 0.0), (1, 1.0)],
        &[],
        1,
        2,
        true,
        false,
        &[1.0, 1.0],
    )
    .unwrap();
    assert_eq!(merged[0].0, 1);
}

#[test]
fn test_merge_topk_precomputed_dist_excludes_nearest() {
    let merged = merge_topk_precomputed_dist(
        &[(0, 0.0), (1, 1.0), (2, 4.0)],
        &[],
        1,
        3,
        true,
    );
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].0, 1);
}

#[test]
fn test_bpann_apply_exclude_nearest_identity_and_find_query() {
    let mut ranked = vec![(3u32, 0.5), (0u32, 0.0), (4u32, 1.0)];
    bpann_apply_exclude_nearest(&mut ranked, true, Some(0));
    assert!(!ranked.iter().any(|(id, _)| *id == 0));

    let mut missed = vec![(3u32, 0.5), (4u32, 1.0)];
    bpann_apply_exclude_nearest(&mut missed, true, Some(0));
    assert_eq!(missed.len(), 2);

    let dir = TempDir::new().unwrap();
    let mut store = MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), 2, None).unwrap();
    store
        .mmap_append(&array![[0.0, 0.0], [1.0, 0.0]].view())
        .unwrap();
    assert_eq!(find_query_train_id(&store, &[1.0, 0.0]), Some(1));
    assert_eq!(
        find_query_train_id_flat(&[0.0, 0.0, 1.0, 0.0], 2, 2, &[1.0, 0.0]),
        Some(1)
    );

    let out = merge_topk_precomputed_dist_with_self(
        &[(0, 0.0), (1, 1.0)],
        &[],
        1,
        2,
        true,
        Some(0),
    );
    assert_eq!(out[0].0, 1);
}
