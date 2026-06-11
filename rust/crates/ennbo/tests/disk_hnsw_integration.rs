//! Disk HNSW integration: incremental add, sync, flat neighbor match.

mod disk_streaming_helper;

use ennbo::{EnnStorage, EpistemicNearestNeighbors, IndexDriver};
use ndarray::{array, Array2};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;
use tempfile::TempDir;

#[test]
fn disk_hnsw_integration_flat_neighbor_match() {
    let seed = 42_u64;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n = 60usize;
    let d = 4usize;
    let dir = TempDir::new().expect("tempdir");

    let mut x_all = Array2::zeros((n, d));
    let mut y_all = Array2::zeros((n, 1));
    for i in 0..n {
        for j in 0..d {
            x_all[[i, j]] = rng.gen::<f64>();
        }
        y_all[[i, 0]] = rng.gen::<f64>();
    }

    let init = 50usize;
    let mut disk = EpistemicNearestNeighbors::new_with_storage(
        x_all.slice(ndarray::s![0..init, ..]).to_owned(),
        y_all.slice(ndarray::s![0..init, ..]).to_owned(),
        None,
        false,
        IndexDriver::HNSWDisk,
        EnnStorage::Disk,
        Some(dir.path().to_path_buf()),
    )
    .expect("new disk hnsw");

    disk.add(
        &x_all.slice(ndarray::s![init..n, ..]),
        &y_all.slice(ndarray::s![init..n, ..]),
        None,
    )
    .expect("add");
    disk.index_access().ensure_sync().expect("sync");

    let flat = EpistemicNearestNeighbors::new_with_storage(
        x_all.clone(),
        y_all.clone(),
        None,
        false,
        IndexDriver::Exact,
        EnnStorage::InMemory,
        None,
    )
    .expect("flat");

    for qi in 0..10 {
        let query = x_all.slice(ndarray::s![qi, ..]).insert_axis(ndarray::Axis(0));
        let disk_idx = disk.neighbors(&query, 1, false).expect("disk neighbors");
        let flat_idx = flat.neighbors(&query, 1, false).expect("flat neighbors");
        assert_eq!(
            disk_idx[[0, 0]],
            flat_idx[[0, 0]],
            "query row {qi} neighbor mismatch"
        );
    }
}

#[test]
fn disk_hnsw_streaming_crosses_flush_threshold() {
    disk_streaming_helper::run_disk_streaming_crosses_flush_threshold(IndexDriver::HNSWDisk);
}

#[test]
fn disk_hnsw_streaming_add_sync_search() {
    disk_streaming_helper::run_disk_streaming_add_sync_search(IndexDriver::HNSWDisk);
}

#[test]
fn disk_hnsw_reopen_scale_x_neighbors_match_without_explicit_sync() {
    let seed = 7_u64;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n = 15usize;
    let d = 3usize;
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().to_path_buf();

    let mut x_all = Array2::zeros((n, d));
    let mut y_all = Array2::zeros((n, 1));
    for i in 0..n {
        for j in 0..d {
            x_all[[i, j]] = rng.gen::<f64>();
        }
        y_all[[i, 0]] = rng.gen::<f64>();
    }

    let model = EpistemicNearestNeighbors::new_with_storage(
        x_all.clone(),
        y_all.clone(),
        None,
        true,
        IndexDriver::HNSWDisk,
        EnnStorage::Disk,
        Some(path.clone()),
    )
    .expect("new disk model");
    model.index_access().ensure_sync().expect("sync");
    drop(model);

    let reopened = EpistemicNearestNeighbors::new_with_storage(
        Array2::zeros((0, d)),
        Array2::zeros((0, 1)),
        None,
        true,
        IndexDriver::HNSWDisk,
        EnnStorage::Disk,
        Some(path.clone()),
    )
    .expect("reopen disk model");

    let fresh = EpistemicNearestNeighbors::new_with_storage(
        x_all,
        y_all,
        None,
        true,
        IndexDriver::HNSWDisk,
        EnnStorage::Disk,
        Some(dir.path().join("fresh")),
    )
    .expect("fresh disk model");
    fresh.index_access().ensure_sync().expect("fresh sync");

    let mut query = Array2::zeros((4, d));
    for i in 0..4 {
        for j in 0..d {
            query[[i, j]] = rng.gen::<f64>();
        }
    }

    let fresh_idx = fresh
        .neighbors(&query.view(), 3, false)
        .expect("fresh neighbors");
    let reopen_idx = reopened
        .neighbors(&query.view(), 3, false)
        .expect("reopen neighbors");
    assert_eq!(
        reopen_idx, fresh_idx,
        "scale_x disk reopen must match fresh neighbors without explicit ensure_index_sync"
    );
}

#[test]
fn disk_hnsw_reopen_wrapper_syncs_num_obs_neighbors_and_y_scale() {
    let dir = TempDir::new().expect("tempdir");
    let path = dir.path().to_path_buf();

    let model = EpistemicNearestNeighbors::new_with_storage(
        array![[0.0, 0.0], [1.0, 2.0]],
        array![[1.0], [5.0]],
        None,
        false,
        IndexDriver::HNSWDisk,
        EnnStorage::Disk,
        Some(path.clone()),
    )
    .expect("new disk model");
    model.index_access().ensure_sync().expect("sync");
    drop(model);

    let reopened = EpistemicNearestNeighbors::new_with_storage(
        Array2::zeros((0, 2)),
        Array2::zeros((0, 1)),
        None,
        false,
        IndexDriver::HNSWDisk,
        EnnStorage::Disk,
        Some(path),
    )
    .expect("reopen disk model");

    assert_eq!(reopened.len(), 2);
    let y_scale = reopened.y_scale_row();
    assert!(
        (y_scale[[0, 0]] - 2.0).abs() < 1e-6,
        "expected y std 2.0 for [1,5], got {}",
        y_scale[[0, 0]]
    );

    let neighbors = reopened
        .neighbors(&array![[0.0, 0.0]].view(), 1, false)
        .expect("neighbors");
    assert_eq!(neighbors.ncols(), 1);
    assert_eq!(neighbors[[0, 0]], 0);
}
