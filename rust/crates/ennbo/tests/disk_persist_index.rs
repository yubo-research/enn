//! Disk ENN persist-on-close: multi-fragment ingest, persist, fast reopen.

use std::fs;
use std::time::Instant;

use ennbo::backend::EnnStorage;
use ennbo::index::IndexDriver;
use ennbo::EpistemicNearestNeighbors;
use ndarray::{Array2, ArrayView2};
use tempfile::TempDir;

fn append_rows(
    model: &mut EpistemicNearestNeighbors,
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
) {
    model.add(x, y, None).expect("add");
}

#[test]
fn disk_persist_index_multi_batch_reopen_is_fast_and_correct() {
    let dir = TempDir::new().expect("tempdir");
    let work_dir = dir.path().to_path_buf();
    let dim = 32usize;
    let rows = 2500usize;
    let query = Array2::from_shape_fn((1, dim), |(_, j)| j as f64 * 0.01);

    let pre_idx = {
        let mut model = EpistemicNearestNeighbors::new_empty(
            dim,
            1,
            IndexDriver::BpAnnDisk,
            EnnStorage::Disk,
            Some(work_dir.clone()),
            Some(1000),
        )
        .expect("new_empty");
        for (start, count) in [(0, 1000usize), (1000, 1000usize), (2000, 500usize)] {
            let x = Array2::from_shape_fn((count, dim), |(i, j)| (start + i + j) as f64);
            let y = Array2::from_shape_fn((count, 1), |(i, _)| (start + i) as f64);
            append_rows(&mut model, &x.view(), &y.view());
            model.schedule_background_flush().expect("flush");
        }
        model.index_access().ensure_sync().expect("sync");
        let idx = model
            .neighbors(&query.view(), 5, false)
            .expect("neighbors");
        model.persist_index_to_disk().expect("persist");
        idx
    };

    let header_text =
        fs::read_to_string(work_dir.join("index/header.json")).expect("header.json");
    assert!(header_text.contains(&format!("\"indexed_rows\": {rows}")));

    let t0 = Instant::now();
    let reopened = EpistemicNearestNeighbors::new_with_storage(
        Array2::zeros((0, dim)),
        Array2::zeros((0, 1)),
        None,
        false,
        IndexDriver::BpAnnDisk,
        EnnStorage::Disk,
        Some(work_dir.clone()),
    )
    .expect("reopen");
    let reopen_s = t0.elapsed().as_secs_f64();
    assert!(
        reopen_s < 1.0,
        "reopen took {reopen_s:.3}s; expected fast mmap open after persist"
    );
    reopened.index_access().ensure_sync().expect("post-reopen sync");
    let post_idx = reopened
        .neighbors(&query.view(), 5, false)
        .expect("neighbors");
    assert_eq!(pre_idx, post_idx);
}
