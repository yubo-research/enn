//! Shared disk backend streaming smoke test body.

use ennbo::{EnnStorage, EpistemicNearestNeighbors, IndexDriver};
use ndarray::Array2;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;
use tempfile::TempDir;

const STREAMING_SEED: u64 = 99;
const STREAMING_D: usize = 4;
const STREAMING_BATCH: usize = 50;
const STREAMING_FLUSH_THRESHOLD: usize = 5;
const STREAMING_CROSS_ROWS: usize = STREAMING_FLUSH_THRESHOLD + 1;
const STREAMING_N: usize = 120;

fn random_batch(
    rng: &mut ChaCha8Rng,
    rows: usize,
    d: usize,
) -> (Array2<f64>, Array2<f64>) {
    let mut x = Array2::zeros((rows, d));
    let mut y = Array2::zeros((rows, 1));
    for i in 0..rows {
        for j in 0..d {
            x[[i, j]] = rng.gen::<f64>();
        }
        y[[i, 0]] = rng.gen::<f64>();
    }
    (x, y)
}

pub fn run_disk_streaming_crosses_flush_threshold(driver: IndexDriver) {
    assert_eq!(driver, IndexDriver::BpAnnDisk);
    let mut rng = ChaCha8Rng::seed_from_u64(STREAMING_SEED);
    let dir = TempDir::new().expect("tempdir");
    let mut model = EpistemicNearestNeighbors::new_empty(
        STREAMING_D,
        1,
        driver,
        EnnStorage::Disk,
        Some(dir.path().to_path_buf()),
        Some(STREAMING_FLUSH_THRESHOLD),
    )
    .expect("new_empty disk");

    let mut row = 0usize;
    while row < STREAMING_CROSS_ROWS {
        let end = (row + STREAMING_BATCH).min(STREAMING_CROSS_ROWS);
        let (x, y) = random_batch(&mut rng, end - row, STREAMING_D);
        model.add(&x.view(), &y.view(), None).expect("add");
        row = end;
    }
    assert_eq!(model.len(), STREAMING_CROSS_ROWS);
    model.index_access().ensure_sync().expect("sync");
    assert_eq!(model.len(), STREAMING_CROSS_ROWS);
}

pub fn run_disk_streaming_add_sync_search(driver: IndexDriver) {
    let mut rng = ChaCha8Rng::seed_from_u64(STREAMING_SEED);
    let dir = TempDir::new().expect("tempdir");
    let mut model = EpistemicNearestNeighbors::new_empty(
        STREAMING_D,
        1,
        driver,
        EnnStorage::Disk,
        Some(dir.path().to_path_buf()),
        None,
    )
    .expect("new_empty disk");

    let mut row = 0usize;
    while row < STREAMING_N {
        let end = (row + STREAMING_BATCH).min(STREAMING_N);
        let (x, y) = random_batch(&mut rng, end - row, STREAMING_D);
        model.add(&x.view(), &y.view(), None).expect("add");
        row = end;
    }
    assert_eq!(model.len(), STREAMING_N);
    model.index_access().ensure_sync().expect("sync");

    let query = Array2::from_shape_fn((3, STREAMING_D), |(_, _)| rng.gen::<f64>());
    let idx = model.neighbors(&query.view(), 5, false).expect("neighbors");
    assert_eq!(idx.nrows(), 3);
    for r in 0..3 {
        for c in 0..5 {
            let id = idx[[r, c]];
            assert!(id < STREAMING_N, "neighbor id {id} out of range");
        }
    }
    let mem = model.index_access().memory_bytes().expect("mem");
    assert!(mem > 0);
}
