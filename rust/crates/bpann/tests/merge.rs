use bpann::mmap_store::MmapColumnStore;
use ndarray::array;
use tempfile::TempDir;

#[test]
fn bpann_merge_topk_candidates() {
    let dir = TempDir::new().unwrap();
    let mut store = MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), 2, None).unwrap();
    store
        .mmap_append(&array![[0.0, 0.0], [1.0, 0.0]].view())
        .unwrap();
    let merged = bpann::merge::bpann_merge_topk_candidates(
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
fn merge_topk_precomputed_dist_excludes_nearest() {
    let merged = bpann::merge::merge_topk_precomputed_dist(
        &[(0, 0.0), (1, 1.0), (2, 4.0)],
        &[],
        1,
        3,
        true,
    );
    assert_eq!(merged.len(), 1);
    assert_eq!(merged[0].0, 1);
}
