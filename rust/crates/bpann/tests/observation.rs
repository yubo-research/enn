use bpann::mmap_store::MmapColumnStore;
use ndarray::array;
use std::sync::Mutex;
use tempfile::TempDir;

#[test]
fn bpann_validate_dim_limits() {
    bpann::observation::bpann_validate_dim_limits(4).unwrap();
    assert!(bpann::observation::bpann_validate_dim_limits(bpann::observation::MAX_NUM_DIM + 1).is_err());
}

#[test]
fn bpann_check_append_row_limit() {
    bpann::observation::bpann_check_append_row_limit(10).unwrap();
}

#[test]
fn bpann_validate_index_backend() {
    let dir = TempDir::new().unwrap();
    bpann::observation::bpann_write_metadata(dir.path(), 0, 4, 1, false, 0).unwrap();
    bpann::observation::bpann_validate_index_backend(dir.path(), bpann::observation::INDEX_BACKEND)
        .unwrap();
}

#[test]
fn bpann_load_indexed_rows() {
    let dir = TempDir::new().unwrap();
    bpann::observation::bpann_write_metadata(dir.path(), 0, 4, 1, false, 2).unwrap();
    assert_eq!(bpann::observation::bpann_load_indexed_rows(dir.path()), Some(2));
}

#[test]
fn bpann_load_index_backend() {
    let dir = TempDir::new().unwrap();
    bpann::observation::bpann_write_metadata(dir.path(), 0, 4, 1, false, 0).unwrap();
    assert_eq!(
        bpann::observation::bpann_load_index_backend(dir.path()).as_deref(),
        Some(bpann::observation::INDEX_BACKEND)
    );
}

#[test]
fn bpann_write_metadata() {
    let dir = TempDir::new().unwrap();
    bpann::observation::bpann_write_metadata(dir.path(), 1, 2, 1, false, 0).unwrap();
}

#[test]
fn bpann_open_or_append_yvar() {
    let dir = TempDir::new().unwrap();
    assert!(bpann::observation::bpann_open_or_append_yvar(dir.path(), 1, Some(&array![[0.1]]))
        .unwrap()
        .is_some());
}

#[test]
fn bpann_append_yvar_on_add() {
    let dir = TempDir::new().unwrap();
    let mut slot = None;
    bpann::observation::bpann_append_yvar_on_add(dir.path(), 1, &mut slot, Some(&array![[0.2]].view()))
        .unwrap();
    assert!(slot.is_some());
}

#[test]
fn bpann_train_rows_at() {
    let dir = TempDir::new().unwrap();
    let mut x = MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), 2, None).unwrap();
    let mut y = MmapColumnStore::mmap_open_or_create(dir.path().join("y.bin"), 1, None).unwrap();
    x.mmap_append(&array![[0.0, 0.0]].view()).unwrap();
    y.mmap_append(&array![[0.0]].view()).unwrap();
    bpann::observation::bpann_train_rows_at(1, &x, &y, None, &[0]).unwrap();
}

#[test]
fn bpann_mark_index_dirty() {
    let dirty = Mutex::new(false);
    bpann::observation::bpann_mark_index_dirty(&dirty);
    assert!(*dirty.lock().unwrap());
}

#[test]
fn num_obs_counter_and_sidecars() {
    let dir = TempDir::new().unwrap();
    let mut counter = bpann::observation::NumObsCounter::open(dir.path()).unwrap();
    counter.set(42);
    assert_eq!(bpann::observation::bpann_load_num_obs(dir.path()), Some(42));
    bpann::observation::write_num_obs(dir.path(), 7).unwrap();
    assert_eq!(bpann::observation::bpann_load_num_obs(dir.path()), Some(7));
    bpann::observation::write_indexed_rows(dir.path(), 5).unwrap();
    assert_eq!(bpann::observation::bpann_load_indexed_rows(dir.path()), Some(5));
}

#[test]
fn bpann_parse_json_string_field() {
    assert_eq!(
        bpann::observation::bpann_parse_json_string_field(
            "{\"index_backend\":\"bpann_disk\"}",
            "index_backend"
        )
        .as_deref(),
        Some("bpann_disk")
    );
}
