//! Disk HNSW integration: incremental add, sync, flat neighbor match.

mod disk_streaming_helper;

use ennbo::{EnnStorage, EpistemicNearestNeighbors, IndexDriver};
use ndarray::Array2;
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
