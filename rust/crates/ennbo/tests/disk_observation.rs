use ndarray::array;
use std::sync::Mutex;
use tempfile::TempDir;

#[test]
fn check_append_row_limit() {
    ennbo::backend::disk_observation::check_append_row_limit(10).unwrap();
}

#[test]
fn read_index_stale() {
    let stale = Mutex::new(false);
    assert!(!ennbo::backend::disk_observation::read_index_stale(&stale));
    ennbo::backend::disk_observation::set_index_stale(&stale);
    assert!(ennbo::backend::disk_observation::read_index_stale(&stale));
}

#[test]
fn open_or_append_yvar() {
    let dir = TempDir::new().unwrap();
    let yv = array![[0.1]];
    ennbo::backend::disk_observation::open_or_append_yvar(dir.path(), 1, Some(&yv)).unwrap();
}

#[test]
fn validate_index_backend() {
    let dir = TempDir::new().unwrap();
    ennbo::backend::disk_observation::write_metadata(dir.path(), 0, 4, 1, false, 0, "hnsw_disk")
        .unwrap();
    ennbo::backend::disk_observation::validate_index_backend(dir.path(), "hnsw_disk").unwrap();
}

#[test]
fn append_yvar_on_add() {
    let dir = TempDir::new().unwrap();
    let mut yvar = ennbo::backend::disk_observation::open_or_append_yvar(dir.path(), 1, None).unwrap();
    ennbo::backend::disk_observation::append_yvar_on_add(
        dir.path(),
        1,
        &mut yvar,
        Some(&array![[0.2]].view()),
    )
    .unwrap();
}

#[test]
fn mark_index_dirty() {
    let dirty = Mutex::new(false);
    ennbo::backend::disk_observation::mark_index_dirty(&dirty);
    assert!(*dirty.lock().unwrap());
}

#[test]
fn load_indexed_rows() {
    let dir = TempDir::new().unwrap();
    ennbo::backend::disk_observation::write_metadata(dir.path(), 0, 4, 1, false, 2, "hnsw_disk")
        .unwrap();
    assert_eq!(
        ennbo::backend::disk_observation::load_indexed_rows(dir.path()),
        Some(2)
    );
}

#[test]
fn load_index_backend() {
    let dir = TempDir::new().unwrap();
    ennbo::backend::disk_observation::write_metadata(dir.path(), 0, 4, 1, false, 0, "hnsw_disk")
        .unwrap();
    assert_eq!(
        ennbo::backend::disk_observation::load_index_backend(dir.path()).as_deref(),
        Some("hnsw_disk")
    );
}

#[test]
fn write_metadata() {
    let dir = TempDir::new().unwrap();
    ennbo::backend::disk_observation::write_metadata(dir.path(), 0, 4, 1, false, 0, "hnsw_disk")
        .unwrap();
}

#[test]
fn validate_dim_limits() {
    ennbo::backend::disk_observation::validate_dim_limits(4, 1).unwrap();
}

#[test]
fn parse_json_usize_field() {
    let text = r#"{"num_obs":42,"index_backend":"hnsw_disk"}"#;
    assert_eq!(
        ennbo::backend::disk_observation::parse_json_usize_field(text, "num_obs"),
        Some(42)
    );
}

#[test]
fn parse_json_string_field() {
    let text = r#"{"index_backend":"hnsw_disk"}"#;
    assert_eq!(
        ennbo::backend::disk_observation::parse_json_string_field(text, "index_backend")
            .as_deref(),
        Some("hnsw_disk")
    );
}
