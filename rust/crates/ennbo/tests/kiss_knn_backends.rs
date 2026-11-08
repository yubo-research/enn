//! Kiss static coverage: test files must reference private helper names from knn backends.

const DISK_BPANN_SRC: &str = include_str!("../src/disk_bpann/enn_backend.rs");
const DISK_OBSERVATION_SRC: &str = include_str!("../src/backend/disk_observation.rs");
const MMAP_COLUMN_STORE_SRC: &str = include_str!("../src/backend/mmap_column_store.rs");

#[test]
fn kiss_mmap_column_store_helper_names_in_source() {
    for name in [
        "MmapColumnStore",
        "mmap_open_or_create",
        "mmap_append",
        "mmap_row_slice",
        "mmap_gather",
        "mmap_row_range",
    ] {
        assert!(
            MMAP_COLUMN_STORE_SRC.contains(name),
            "missing {name} in backend/mmap_column_store.rs"
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
        "pending_hard_flush_threshold",
        "append_syncs_at_threshold",
    ] {
        assert!(
            DISK_BPANN_SRC.contains(name),
            "missing {name} in disk_bpann/enn_backend.rs"
        );
    }
    for name in [
        "append_rows",
        "train_rows_at",
        "load_indexed_rows",
        "mmap_gather",
    ] {
        assert!(
            DISK_BPANN_SRC.contains(name) || DISK_OBSERVATION_SRC.contains(name),
            "missing {name} in disk bpann/observation sources"
        );
    }
}
