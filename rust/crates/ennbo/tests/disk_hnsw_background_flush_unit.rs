//! Unit tests for disk HNSW background flush coordination (`disk_hnsw/flush.rs`).

use ennbo::backend::DiskEnnBackend;
use ennbo::disk_hnsw::flush::{
    finish_flush_thread, lock_flush_state, try_schedule_background_flush,
    wait_for_background_flush, BackgroundFlushState, FlushTestBarrier,
};
use ennbo::disk_hnsw::DiskHnswEnnBackend;
use ennbo::error::ENNError;
use ndarray::array;
use std::sync::{Arc, Mutex};
use tempfile::TempDir;

fn hnsw_arc(backend: DiskHnswEnnBackend) -> Arc<Mutex<DiskEnnBackend>> {
    Arc::new(Mutex::new(DiskEnnBackend::Hnsw(backend)))
}

fn flush_arc(arc: &Arc<Mutex<DiskEnnBackend>>) -> Arc<Mutex<BackgroundFlushState>> {
    let guard = arc.lock().expect("disk lock");
    let DiskEnnBackend::Hnsw(b) = &*guard;
    b.flush_arc()
}

fn schedule_background_flush(arc: &Arc<Mutex<DiskEnnBackend>>) {
    let (flush, should) = {
        let guard = arc.lock().expect("disk lock");
        let DiskEnnBackend::Hnsw(b) = &*guard;
        (
            b.flush_arc(),
            !b.is_index_stale() && b.pending_rows() >= b.pending_flush_threshold(),
        )
    };
    if should {
        try_schedule_background_flush(&flush, Arc::clone(arc)).unwrap();
    }
}

fn append_row(backend: &mut DiskHnswEnnBackend, i: f64) {
    backend
        .append_rows(
            &array![[i, i]].view(),
            &array![[i]].view(),
            None,
        )
        .unwrap();
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
        .with_pending_flush_threshold(3);
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
fn lock_flush_state_locks_ok() {
    let st = Arc::new(Mutex::new(BackgroundFlushState::default()));
    {
        let guard = lock_flush_state(&st).expect("ok");
        assert!(!guard.in_progress);
    }
}

#[test]
fn background_flush_state_inject_failure() {
    let mut st = BackgroundFlushState::default();
    st.inject_failure();
}

#[test]
fn flush_state_default_and_finish_thread() {
    let st = Arc::new(Mutex::new(BackgroundFlushState::default()));
    assert!(!st.lock().expect("lock").in_progress);
    finish_flush_thread(
        &st,
        Err(ENNError::InvalidParameter("flush err".to_string())),
    );
    assert!(st.lock().expect("lock").error.is_some());
    finish_flush_thread(&st, Ok(()));
    assert!(st.lock().expect("lock").error.is_none());
}

#[test]
fn flush_wait_idle_is_noop() {
    let st = Arc::new(Mutex::new(BackgroundFlushState::default()));
    wait_for_background_flush(&st).unwrap();
}

#[test]
fn wait_for_background_flush_joins_completed_thread() {
    let st = Arc::new(Mutex::new(BackgroundFlushState::default()));
    {
        let mut guard = st.lock().expect("lock");
        guard.in_progress = true;
        guard.join_handle = Some(std::thread::spawn(|| {}));
    }
    wait_for_background_flush(&st).unwrap();
    let guard = st.lock().expect("lock");
    assert!(guard.join_handle.is_none());
}

#[test]
fn wait_for_background_flush_poisoned_mutex_returns_err() {
    let st = Arc::new(Mutex::new(BackgroundFlushState::default()));
    let st2 = Arc::clone(&st);
    let _ = std::thread::spawn(move || {
        let _guard = st2.lock().expect("lock");
        panic!("poison flush mutex");
    })
    .join();
    assert!(st.is_poisoned());
    let err = wait_for_background_flush(&st).unwrap_err();
    assert!(err.to_string().contains("flush state mutex poisoned"));
}

#[test]
fn flush_wait_returns_stored_error_when_idle() {
    let st = Arc::new(Mutex::new(BackgroundFlushState::default()));
    st.lock()
        .expect("lock")
        .error = Some(ENNError::InvalidParameter("stored flush err".to_string()));
    let err = wait_for_background_flush(&st).unwrap_err();
    assert!(err.to_string().contains("stored flush err"));
}

#[test]
fn flush_wait_reports_thread_panic() {
    let st = Arc::new(Mutex::new(BackgroundFlushState::default()));
    {
        let mut guard = st.lock().expect("lock");
        guard.in_progress = true;
        guard.join_handle = Some(std::thread::spawn(|| {
            panic!("flush boom");
        }));
    }
    let err = wait_for_background_flush(&st).unwrap_err();
    assert!(err.to_string().contains("panicked"));
}

#[test]
fn wait_for_background_flush_concurrent_waiters_block() {
    let dir = TempDir::new().expect("tempdir");
    let mut backend = DiskHnswEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1)
        .unwrap()
        .with_pending_flush_threshold(3);
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
    let f1 = Arc::clone(&flush);
    let f2 = Arc::clone(&flush);
    let w1 = std::thread::spawn(move || wait_for_background_flush(&f1));
    let w2 = std::thread::spawn(move || wait_for_background_flush(&f2));
    flush
        .lock()
        .expect("flush lock")
        .barrier
        .set_hold(false);
    w1.join().expect("waiter 1").unwrap();
    w2.join().expect("waiter 2").unwrap();
    let guard = arc.lock().expect("disk lock");
    let DiskEnnBackend::Hnsw(ref b) = *guard;
    assert_eq!(b.indexed_rows(), b.len());
}

#[test]
fn flush_schedule_indexes_pending_rows() {
    let dir = TempDir::new().expect("tempdir");
    let mut backend = DiskHnswEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1)
        .unwrap()
        .with_pending_flush_threshold(3);
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
        .with_pending_flush_threshold(3);
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
        .with_pending_flush_threshold(3);
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
fn flush_schedule_coalesces_when_in_progress() {
    let dir = TempDir::new().expect("tempdir");
    let mut backend = DiskHnswEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1)
        .unwrap()
        .with_pending_flush_threshold(3);
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
        .with_pending_flush_threshold(3);
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
