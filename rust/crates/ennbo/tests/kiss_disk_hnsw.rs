//! Kiss static coverage: disk HNSW backend and graph module symbols.

const DISK_HNSW_SRC: &str = include_str!("../src/disk_hnsw/enn_backend.rs");
const DISK_FLUSH_SRC: &str = include_str!("../src/disk_hnsw/flush.rs");
const DISK_OBSERVATION_SRC: &str = include_str!("../src/backend/disk_observation.rs");
const HNSW_GRAPH_SRC: &str = include_str!("../src/disk_hnsw/hnsw.rs");
const HNSW_STORE_SRC: &str = include_str!("../src/disk_hnsw/store.rs");

#[test]
fn kiss_disk_hnsw_backend_names_in_source() {
    for name in [
        "DiskHnswEnnBackend",
        "open_or_create_graph",
        "disk_obs::load_num_obs",
        "INDEX_BACKEND",
    ] {
        assert!(DISK_HNSW_SRC.contains(name), "missing {name}");
    }
}

#[test]
fn kiss_disk_flush_names_in_source() {
    for name in [
        "BackgroundFlushState",
        "wait_for_background_flush",
        "try_schedule_background_flush",
        "finish_flush_thread",
        "FlushTestBarrier",
        "lock_flush_state",
    ] {
        assert!(DISK_FLUSH_SRC.contains(name), "missing {name}");
    }
}

#[test]
fn kiss_disk_flush_runtime_reference() {
    use std::sync::{Arc, Mutex};
    use ennbo::disk_hnsw::flush::{wait_for_background_flush, BackgroundFlushState};
    let st = Arc::new(Mutex::new(BackgroundFlushState::default()));
    wait_for_background_flush(&st).unwrap();
}

#[test]
fn kiss_disk_observation_and_graph_names_in_source() {
    for name in [
        "load_indexed_rows",
        "write_metadata",
        "search_layer",
        "check_append_row_limit",
        "truncate_nodes",
        "NodeLayout",
    ] {
        assert!(
            DISK_OBSERVATION_SRC.contains(name)
                || HNSW_GRAPH_SRC.contains(name)
                || HNSW_STORE_SRC.contains(name),
            "missing {name}"
        );
    }
}
