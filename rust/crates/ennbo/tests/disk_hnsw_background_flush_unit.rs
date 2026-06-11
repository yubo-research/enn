//! Unit tests for disk HNSW background flush scheduling (`disk_hnsw/flush.rs`).

#[path = "flush_hnsw_helpers.rs"]
mod flush_hnsw_helpers;

use flush_hnsw_helpers::{append_row, flush_arc, hnsw_arc, schedule_background_flush};
use ennbo::backend::DiskEnnBackend;
use ennbo::disk_hnsw::flush::{
    wait_for_background_flush, BackgroundFlushState, FlushTestBarrier,
};
use ennbo::disk_hnsw::DiskHnswEnnBackend;
use std::sync::Arc;
use tempfile::TempDir;

#[test]
fn flush_inline_completes_without_barrier() {
    let dir = TempDir::new().expect("tempdir");
    let mut backend = DiskHnswEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1)
        .unwrap()
        .with_pending_flush_threshold(3)
        .with_defer_append_indexing(true);
    for i in 0..3 {
        append_row(&mut backend, i as f64);
    }
    let arc = hnsw_arc(backend);
    schedule_background_flush(&arc);
    let flush = flush_arc(&arc);
    let st = flush.lock().expect("flush lock");
    assert!(!st.in_progress);
    assert!(st.join_handle.is_none());
    drop(st);
    let guard = arc.lock().expect("disk lock");
    let DiskEnnBackend::Hnsw(ref b) = *guard;
    assert_eq!(b.indexed_rows(), b.len());
}

#[test]
fn flush_test_barrier_hold_and_release() {
    let barrier = FlushTestBarrier::default();
    assert!(!barrier.is_holding());
    barrier.set_hold(true);
    assert!(barrier.is_holding());
    barrier.set_hold(false);
    assert!(!barrier.is_holding());
    barrier.wait_if_holding();
}

#[test]
fn flush_schedule_respects_barrier_hold() {
    let dir = TempDir::new().expect("tempdir");
    let mut backend = DiskHnswEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1)
        .unwrap()
        .with_pending_flush_threshold(3)
        .with_defer_append_indexing(true);
    for i in 0..3 {
        append_row(&mut backend, i as f64);
    }
    let arc = hnsw_arc(backend);
    {
        let guard = arc.lock().expect("disk lock");
        let DiskEnnBackend::Hnsw(ref b) = *guard;
        b.flush_test_barrier_hold(true);
    }
    schedule_background_flush(&arc);
    {
        let guard = arc.lock().expect("disk lock");
        let DiskEnnBackend::Hnsw(ref b) = *guard;
        let indexed_before = b.indexed_rows();
        assert_eq!(b.indexed_rows(), indexed_before);
    }
    flush_arc(&arc)
        .lock()
        .expect("flush lock")
        .barrier
        .set_hold(false);
    wait_for_background_flush(&flush_arc(&arc)).unwrap();
    let guard = arc.lock().expect("disk lock");
    let DiskEnnBackend::Hnsw(ref b) = *guard;
    assert_eq!(b.indexed_rows(), b.len());
}

#[test]
fn background_flush_state_inject_failure() {
    let mut st = BackgroundFlushState::default();
    st.inject_failure();
}

#[test]
fn flush_schedule_indexes_pending_rows() {
    let dir = TempDir::new().expect("tempdir");
    let mut backend = DiskHnswEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1)
        .unwrap()
        .with_pending_flush_threshold(3)
        .with_defer_append_indexing(true);
    for i in 0..3 {
        append_row(&mut backend, i as f64);
    }
    let arc = hnsw_arc(backend);
    schedule_background_flush(&arc);
    wait_for_background_flush(&flush_arc(&arc)).unwrap();
    let guard = arc.lock().expect("disk lock");
    let DiskEnnBackend::Hnsw(ref b) = *guard;
    assert_eq!(b.indexed_rows(), b.len());
}

#[test]
fn flush_success_clears_prior_error_after_reschedule() {
    let dir = TempDir::new().expect("tempdir");
    let mut backend = DiskHnswEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1)
        .unwrap()
        .with_pending_flush_threshold(3)
        .with_defer_append_indexing(true);
    for i in 0..3 {
        append_row(&mut backend, i as f64);
    }
    backend.inject_next_flush_failure();
    let arc = hnsw_arc(backend);
    let flush = flush_arc(&arc);
    schedule_background_flush(&arc);
    assert!(wait_for_background_flush(&flush).is_err());
    schedule_background_flush(&arc);
    wait_for_background_flush(&flush).unwrap();
}

#[test]
fn flush_error_surfaces_on_wait() {
    let dir = TempDir::new().expect("tempdir");
    let mut backend = DiskHnswEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1)
        .unwrap()
        .with_pending_flush_threshold(3)
        .with_defer_append_indexing(true);
    for i in 0..3 {
        append_row(&mut backend, i as f64);
    }
    backend.inject_next_flush_failure();
    let arc = hnsw_arc(backend);
    schedule_background_flush(&arc);
    let err = wait_for_background_flush(&flush_arc(&arc)).unwrap_err();
    assert!(err.to_string().contains("injected background flush failure"));
}

#[test]
fn background_flush_clears_join_handle_when_idle() {
    let dir = TempDir::new().expect("tempdir");
    let mut backend = DiskHnswEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1)
        .unwrap()
        .with_pending_flush_threshold(3)
        .with_defer_append_indexing(true);
    for i in 0..3 {
        append_row(&mut backend, i as f64);
    }
    let arc = hnsw_arc(backend);
    {
        let guard = arc.lock().expect("disk lock");
        let DiskEnnBackend::Hnsw(ref b) = *guard;
        b.flush_test_barrier_hold(true);
    }
    schedule_background_flush(&arc);
    let flush = flush_arc(&arc);
    assert!(flush.lock().expect("flush lock").in_progress);
    flush
        .lock()
        .expect("flush lock")
        .barrier
        .set_hold(false);
    for _ in 0..50 {
        let st = flush.lock().expect("flush lock");
        if !st.in_progress {
            assert!(
                st.join_handle.is_none(),
                "idle flush state must not retain a stale join handle"
            );
            return;
        }
        drop(st);
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    panic!("background flush did not finish within timeout");
}

#[test]
fn flush_schedule_coalesces_when_in_progress() {
    let dir = TempDir::new().expect("tempdir");
    let mut backend = DiskHnswEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1)
        .unwrap()
        .with_pending_flush_threshold(3)
        .with_defer_append_indexing(true);
    for i in 0..3 {
        append_row(&mut backend, i as f64);
    }
    let arc = hnsw_arc(backend);
    schedule_background_flush(&arc);
    schedule_background_flush(&arc);
    wait_for_background_flush(&flush_arc(&arc)).unwrap();
    let guard = arc.lock().expect("disk lock");
    let DiskEnnBackend::Hnsw(ref b) = *guard;
    assert_eq!(b.indexed_rows(), b.len());
}

#[test]
fn disk_hnsw_stale_skips_background_schedule() {
    let dir = TempDir::new().expect("tempdir");
    let mut backend = DiskHnswEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1)
        .unwrap()
        .with_pending_flush_threshold(3)
        .with_defer_append_indexing(true);
    for i in 0..3 {
        append_row(&mut backend, i as f64);
    }
    backend.mark_index_stale();
    let arc = hnsw_arc(backend);
    {
        let guard = arc.lock().expect("disk lock");
        let DiskEnnBackend::Hnsw(ref b) = *guard;
        b.schedule_background_flush(Arc::clone(&arc)).unwrap();
    }
    let flush = flush_arc(&arc);
    let st = flush.lock().expect("flush lock");
    assert!(!st.in_progress);
    assert!(st.join_handle.is_none());
}

#[test]
fn disk_hnsw_drop_joins_background_flush() {
    use ennbo::backend::{EnnBackend, EnnStorage};
    use ennbo::index::IndexDriver;

    let dir = TempDir::new().expect("tempdir");
    let backend = EnnBackend::new_empty(
        2,
        1,
        IndexDriver::HNSWDisk,
        EnnStorage::Disk,
        Some(dir.path().to_path_buf()),
    )
    .unwrap();
    let arc = match &backend {
        EnnBackend::Disk(a) => Arc::clone(a),
        _ => panic!("expected disk backend"),
    };
    {
        let mut guard = arc.lock().expect("disk lock");
        let DiskEnnBackend::Hnsw(ref mut b) = *guard;
        b.set_pending_flush_threshold(3);
        b.set_defer_append_indexing(true);
        for i in 0..3 {
            append_row(b, i as f64);
        }
        b.flush_test_barrier_hold(true);
    }
    schedule_background_flush(&arc);
    let drop_handle = std::thread::spawn(move || drop(backend));
    let flush = flush_arc(&arc);
    {
        let st = flush.lock().expect("flush lock");
        assert!(st.in_progress);
    }
    flush.lock().expect("flush lock").barrier.set_hold(false);
    drop_handle.join().expect("drop thread");
    wait_for_background_flush(&flush).unwrap();
    let guard = arc.lock().expect("disk lock");
    let DiskEnnBackend::Hnsw(ref b) = *guard;
    assert_eq!(b.indexed_rows(), b.len());
}

#[test]
fn ensure_index_sync_propagates_background_flush_error() {
    use ennbo::backend::{EnnBackend, EnnStorage};
    use ennbo::index::IndexDriver;
    use ndarray::Array1;

    let dir = TempDir::new().expect("tempdir");
    let backend = EnnBackend::new_empty(
        2,
        1,
        IndexDriver::HNSWDisk,
        EnnStorage::Disk,
        Some(dir.path().to_path_buf()),
    )
    .unwrap();
    let arc = match &backend {
        EnnBackend::Disk(a) => Arc::clone(a),
        _ => panic!("expected disk backend"),
    };
    {
        let mut guard = arc.lock().expect("disk lock");
        let DiskEnnBackend::Hnsw(ref mut b) = *guard;
        b.set_pending_flush_threshold(3);
        b.set_defer_append_indexing(true);
        for i in 0..3 {
            append_row(b, i as f64);
        }
        b.inject_next_flush_failure();
    }
    schedule_background_flush(&arc);
    let err = backend
        .ensure_index_sync(false, &Array1::ones(2))
        .unwrap_err();
    assert!(err.to_string().contains("injected background flush failure"));
}
