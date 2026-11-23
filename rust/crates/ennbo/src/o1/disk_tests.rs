use super::buckets::{rebuild_partitions, slot_bytes};
use super::*;
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::path::Path;
use tempfile::TempDir;

const D: usize = 4;

fn store_with(dir: &Path, params: O1Params, n: usize, batch: usize) -> O1Store {
    let mut store = O1Store::open_with_params(dir.to_path_buf(), (D, 1), params).unwrap();
    let mut rng = StdRng::seed_from_u64(11);
    let x = Array2::from_shape_fn((n, D), |_| rng.gen::<f64>());
    let y = x.column(0).to_owned().insert_axis(ndarray::Axis(1));
    for lo in (0..n).step_by(batch) {
        let hi = (lo + batch).min(n);
        let xs = x.slice(ndarray::s![lo..hi, ..]);
        let ys = y.slice(ndarray::s![lo..hi, ..]);
        store.append_rows(&xs, &ys, None).unwrap();
    }
    store
}

fn bucket_files(dir: &Path, num_tables: usize) -> Vec<Vec<u8>> {
    (0..num_tables)
        .map(|t| std::fs::read(dir.join(format!("o1_buckets_{t}.bin"))).unwrap())
        .collect()
}

fn queries() -> Vec<Vec<f64>> {
    let mut rng = StdRng::seed_from_u64(5);
    (0..10).map(|_| (0..D).map(|_| rng.gen::<f64>()).collect()).collect()
}

#[test]
fn partitioned_rebuild_matches_single_pass_rebuild() {
    let base = O1Params::default();
    let small = O1Params { build_budget_bytes: 16 * slot_bytes(base.bucket_cap), ..base };
    let (a, b) = (TempDir::new().unwrap(), TempDir::new().unwrap());
    let one = store_with(a.path(), base, 3000, 700);
    let many = store_with(b.path(), small, 3000, 700);
    assert!(rebuild_partitions(many.hash_bits(), base.bucket_cap, small.build_budget_bytes).1 > 1);
    assert_eq!(bucket_files(a.path(), base.num_tables), bucket_files(b.path(), base.num_tables));
    assert!(!b.path().join("o1_buckets_0.parts").exists());
    drop((one, many));
}

#[test]
fn incremental_inserts_match_a_full_rebuild() {
    let dir = TempDir::new().unwrap();
    let mut store = store_with(dir.path(), O1Params::default(), 3000, 100);
    store.index.flush_pending().unwrap();
    let before = bucket_files(dir.path(), 8);
    store.index.rebuild_buckets().unwrap();
    assert_eq!(before, bucket_files(dir.path(), 8));
}

#[test]
fn reopen_reads_the_disk_index_without_rebuilding() {
    let dir = TempDir::new().unwrap();
    let store = store_with(dir.path(), O1Params::default(), 3000, 500);
    let cands: Vec<Vec<u32>> = queries().iter().map(|q| store.index.candidate_ids(q).unwrap()).collect();
    let bits = store.hash_bits();
    drop(store);
    let stamp = |t: usize| std::fs::metadata(dir.path().join(format!("o1_buckets_{t}.bin"))).unwrap().modified().unwrap();
    let stamps: Vec<_> = (0..8).map(stamp).collect();
    let reopened = O1Store::open_or_create(dir.path().to_path_buf(), D, 1).unwrap();
    assert_eq!(stamps, (0..8).map(stamp).collect::<Vec<_>>());
    assert_eq!(reopened.hash_bits(), bits);
    let again: Vec<Vec<u32>> = queries().iter().map(|q| reopened.index.candidate_ids(q).unwrap()).collect();
    assert_eq!(cands, again);
    assert!(reopened.index.pending_bytes() > 0);
    assert_eq!(reopened.index_memory_bytes(), 8 * bits * (D + 2) * 8 + 8 * D + reopened.index.pending_bytes());
}

#[test]
fn missing_index_state_rebuilds_the_index_from_rows() {
    let dir = TempDir::new().unwrap();
    let store = store_with(dir.path(), O1Params::default(), 3000, 500);
    drop(store);
    let state_path = dir.path().join(disk_index::O1_INDEX_FILE);
    std::fs::remove_file(&state_path).unwrap();
    for t in 0..8 {
        std::fs::remove_file(dir.path().join(format!("o1_buckets_{t}.bin"))).unwrap();
    }
    let reopened = O1Store::open_or_create(dir.path().to_path_buf(), D, 1).unwrap();
    assert_eq!(reopened.len(), 3000);
    let saved: disk_index::IndexState = bpann_read_json(&state_path).unwrap().unwrap();
    assert_eq!(saved.indexed_rows, 3000);
    for q in queries() {
        let (_, idx) = reopened.search(&ndarray::arr2(&[[q[0], q[1], q[2], q[3]]]).view(), 1, false).unwrap();
        assert!(reopened.index.candidate_ids(&q).unwrap().len() > 8);
        assert!(idx[[0, 0]] >= 0);
    }
}

#[test]
fn stale_index_state_is_never_trusted() {
    let dir = TempDir::new().unwrap();
    let mut store = store_with(dir.path(), O1Params::default(), 3000, 500);
    let state_path = dir.path().join(disk_index::O1_INDEX_FILE);
    let stale = std::fs::read(&state_path).unwrap();
    let (x, y) = (Array2::from_elem((5000, D), 0.5), Array2::from_elem((5000, 1), 0.5));
    store.append_rows(&x.view(), &y.view(), None).unwrap();
    drop(store);
    std::fs::write(&state_path, stale).unwrap();
    let reopened = O1Store::open_or_create(dir.path().to_path_buf(), D, 1).unwrap();
    assert_eq!(reopened.len(), 8000);
    let saved: disk_index::IndexState = bpann_read_json(&state_path).unwrap().unwrap();
    assert_eq!(saved.indexed_rows, 8000);
    assert_eq!(saved.hash_scale, vec![1.0; D]);
}

#[test]
fn buffered_inserts_match_immediate_writes() {
    let base = O1Params::default();
    let eager = O1Params { pending_budget_bytes: 0, ..base };
    let (a, b) = (TempDir::new().unwrap(), TempDir::new().unwrap());
    let immediate = store_with(a.path(), eager, 3000, 100);
    let mut buffered = store_with(b.path(), base, 3000, 100);
    assert_eq!(immediate.index.pending_rows(), 0);
    assert!(buffered.index.pending_rows() > 0);
    for q in queries() {
        assert_eq!(immediate.index.candidate_ids(&q).unwrap(), buffered.index.candidate_ids(&q).unwrap());
    }
    buffered.index.flush_pending().unwrap();
    assert_eq!(buffered.index.pending_rows(), 0);
    assert_eq!(bucket_files(a.path(), 8), bucket_files(b.path(), 8));
}

#[test]
fn reopen_replays_buffered_rows_from_the_key_files() {
    let dir = TempDir::new().unwrap();
    let store = store_with(dir.path(), O1Params::default(), 3000, 500);
    let pending = store.index.pending_rows();
    assert!(pending > 0);
    let cands: Vec<Vec<u32>> = queries().iter().map(|q| store.index.candidate_ids(q).unwrap()).collect();
    drop(store);
    let reopened = O1Store::open_or_create(dir.path().to_path_buf(), D, 1).unwrap();
    assert_eq!(reopened.index.pending_rows(), pending);
    let again: Vec<Vec<u32>> = queries().iter().map(|q| reopened.index.candidate_ids(q).unwrap()).collect();
    assert_eq!(cands, again);
}
