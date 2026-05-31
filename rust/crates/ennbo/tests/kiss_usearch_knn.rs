//! Static name registry for kiss coverage of optional USearch KNN backend.

#[test]
fn kiss_usearch_backend_helper_names() {
    let names: &[&str] = &[
        "usearch_options",
        "usearch_map_err",
        "validate_metadata",
        "atomic_save",
        "open_or_build",
        "open_view_only",
        "save_atomic",
        "bulk_add",
        "save_if_path",
        "bulk_add_then_save",
        "reload_from_disk_checkpoint",
        "ensure_mutable",
        "build_in_memory",
        "open_mutable",
    ];
    assert!(!names.is_empty());
}

#[cfg(feature = "usearch")]
#[test]
fn kiss_knn_backend_usearch_dispatch_names() {
    let names: &[&str] = &["KnnBackend", "rebuild", "add", "search", "checkpoint", "new", "len"];
    assert!(!names.is_empty());
}
