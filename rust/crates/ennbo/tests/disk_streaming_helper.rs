//! Shared disk backend streaming smoke test body.

use ennbo::{EnnStorage, EpistemicNearestNeighbors, IndexDriver};
use ndarray::Array2;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::SeedableRng;
use tempfile::TempDir;

pub fn run_disk_streaming_add_sync_search(driver: IndexDriver) {
    let seed = 99_u64;
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let n = 10_000usize;
    let d = 4usize;
    let dir = TempDir::new().expect("tempdir");
    let mut model = EpistemicNearestNeighbors::new_empty(
        d,
        1,
        driver,
        EnnStorage::Disk,
        Some(dir.path().to_path_buf()),
    )
    .expect("new_empty disk");

    let batch = 500usize;
    let mut row = 0usize;
    while row < n {
        let end = (row + batch).min(n);
        let rows = end - row;
        let mut x = Array2::zeros((rows, d));
        let mut y = Array2::zeros((rows, 1));
        for i in 0..rows {
            for j in 0..d {
                x[[i, j]] = rng.gen::<f64>();
            }
            y[[i, 0]] = rng.gen::<f64>();
        }
        model.add(&x.view(), &y.view(), None).expect("add");
        row = end;
    }
    assert_eq!(model.len(), n);
    model.index_access().ensure_sync().expect("sync");

    let query = Array2::from_shape_fn((3, d), |(_, _)| rng.gen::<f64>());
    let idx = model.neighbors(&query.view(), 5, false).expect("neighbors");
    assert_eq!(idx.nrows(), 3);
    for r in 0..3 {
        for c in 0..5 {
            let id = idx[[r, c]];
            assert!(id < n, "neighbor id {id} out of range");
        }
    }
    let mem = model.index_access().memory_bytes().expect("mem");
    assert!(mem > 0);
}
