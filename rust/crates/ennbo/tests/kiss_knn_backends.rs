//! Kiss static coverage: test files must reference private helper names from knn backends.

const DISK_BPANN_SRC: &str = include_str!("../src/disk_bpann/enn_backend.rs");
const DISK_OBSERVATION_SRC: &str = include_str!("../src/backend/disk_observation.rs");
const FAISS_BACKEND_SRC: &str = include_str!("../src/knn/faiss_backend.rs");
const ROW_STORAGE_SRC: &str = include_str!("../src/backend/row_storage.rs");
const KNN_MOD_SRC: &str = include_str!("../src/knn/mod.rs");

#[test]
fn kiss_faiss_backend_helper_names_in_source() {
    for name in [
        "faiss_spec",
        "faiss_map_err",
        "make_index",
        "memory_usage_bytes",
        "MmapColumnStore",
        "mmap_open_or_create",
        "mmap_append",
        "mmap_row_slice",
        "mmap_gather",
        "mmap_row_range",
    ] {
        assert!(
            FAISS_BACKEND_SRC.contains(name),
            "missing {name} in faiss_backend.rs"
        );
    }
}

#[test]
fn kiss_knn_mod_dispatch_names_in_source() {
    for name in ["KnnBackend", "rebuild", "add", "search", "memory_usage_bytes"] {
        assert!(KNN_MOD_SRC.contains(name), "missing {name} in knn/mod.rs");
    }
}

#[test]
fn kiss_row_storage_helper_names_in_source() {
    for name in [
        "RowStorage",
        "from_array2",
        "push_rows",
        "gather_rows",
        "row_vec",
        "nrows",
        "view",
    ] {
        assert!(
            ROW_STORAGE_SRC.contains(name),
            "missing {name} in backend/row_storage.rs"
        );
    }
}

#[test]
fn kiss_disk_bpann_helper_names_in_source() {
    for name in [
        "DiskBpannEnnBackend",
        "mark_index_stale",
        "ensure_index_sync",
        "row_x",
        "row_y",
        "row_yvar",
        "search",
        "index_memory_bytes",
        "new_empty",
        "pending_flush_threshold",
        "append_syncs_at_threshold",
    ] {
        assert!(
            DISK_BPANN_SRC.contains(name),
            "missing {name} in disk_bpann/enn_backend.rs"
        );
    }
    for name in ["append_rows", "train_rows_at", "load_indexed_rows", "mmap_gather"] {
        assert!(
            DISK_BPANN_SRC.contains(name) || DISK_OBSERVATION_SRC.contains(name),
            "missing {name} in disk bpann/observation sources"
        );
    }
}
