//! wait_for_background_flush and run_flush_body coverage (`disk_hnsw/flush.rs`).

#[path = "flush_hnsw_helpers.rs"]
mod flush_hnsw_helpers;

use flush_hnsw_helpers::{append_row, flush_arc, hnsw_arc, schedule_background_flush};
use ennbo::backend::DiskEnnBackend;
use ennbo::disk_hnsw::flush::{
    finish_flush_thread, lock_flush_state, run_flush_body, wait_for_background_flush,
    BackgroundFlushState,
};
use ennbo::disk_hnsw::DiskHnswEnnBackend;
use ennbo::error::ENNError;
use std::sync::{Arc, Mutex};
use tempfile::TempDir;

#[test]
fn run_flush_body_indexes_pending_rows() {
    let dir = TempDir::new().expect("tempdir");
    let mut backend = DiskHnswEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1)
        .unwrap()
        .with_pending_flush_threshold(3)
        .with_defer_append_indexing(true);
    for i in 0..3 {
        append_row(&mut backend, i as f64);
    }
    let arc = hnsw_arc(backend);
    let flush = flush_arc(&arc);
    run_flush_body(&flush, &arc).unwrap();
    let guard = arc.lock().expect("disk lock");
    let DiskEnnBackend::Hnsw(ref b) = *guard;
    assert_eq!(b.indexed_rows(), b.len());
}

#[test]
fn run_flush_body_disk_mutex_poison_returns_err() {
    let st = Arc::new(Mutex::new(BackgroundFlushState::default()));
    let dir = TempDir::new().expect("tempdir");
    let backend = DiskHnswEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1).unwrap();
    let arc = hnsw_arc(backend);
    let arc2 = Arc::clone(&arc);
    let _ = std::thread::spawn(move || {
        let _guard = arc2.lock().expect("disk lock");
        panic!("poison disk mutex");
    })
    .join();
    assert!(arc.is_poisoned());
    let err = run_flush_body(&st, &arc).unwrap_err();
    assert!(err.to_string().contains("disk backend mutex poisoned"));
}

#[test]
fn run_flush_body_propagates_injected_failure() {
    let dir = TempDir::new().expect("tempdir");
    let mut backend = DiskHnswEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1)
        .unwrap()
        .with_defer_append_indexing(true);
    append_row(&mut backend, 0.0);
    backend.inject_next_flush_failure();
    let arc = hnsw_arc(backend);
    let flush = flush_arc(&arc);
    let err = run_flush_body(&flush, &arc).unwrap_err();
    assert!(err.to_string().contains("injected background flush failure"));
}

#[test]
fn wait_for_background_flush_yields_while_in_progress() {
    let st = Arc::new(Mutex::new(BackgroundFlushState::default()));
    st.lock().expect("lock").in_progress = true;
    let st2 = Arc::clone(&st);
    let releaser = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(50));
        st2.lock().expect("lock").in_progress = false;
    });
    wait_for_background_flush(&st).unwrap();
    releaser.join().expect("join");
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
    let st = Arc::new(Mutex::new(BackgroundFlushState::default()));
    {
        let mut guard = st.lock().expect("flush lock");
        guard.in_progress = true;
        guard.join_handle = None;
    }
    let s1 = Arc::clone(&st);
    let s2 = Arc::clone(&st);
    let w1 = std::thread::spawn(move || wait_for_background_flush(&s1));
    let w2 = std::thread::spawn(move || wait_for_background_flush(&s2));
    std::thread::sleep(std::time::Duration::from_millis(20));
    finish_flush_thread(&st, Ok(()));
    w1.join().expect("waiter 1").unwrap();
    w2.join().expect("waiter 2").unwrap();
    assert!(!st.lock().expect("flush lock").in_progress);
}

#[test]
fn wait_for_background_flush_concurrent_waiters_both_see_stored_error() {
    let st = Arc::new(Mutex::new(BackgroundFlushState::default()));
    st.lock()
        .expect("lock")
        .error = Some(ENNError::InvalidParameter("stored flush err".to_string()));
    let s1 = Arc::clone(&st);
    let s2 = Arc::clone(&st);
    let w1 = std::thread::spawn(move || wait_for_background_flush(&s1));
    let w2 = std::thread::spawn(move || wait_for_background_flush(&s2));
    let r1 = w1.join().expect("waiter 1");
    let r2 = w2.join().expect("waiter 2");
    assert!(
        r1.is_err(),
        "waiter 1 should see stored error, got {r1:?}"
    );
    assert!(
        r2.is_err(),
        "waiter 2 should see stored error, got {r2:?}"
    );
}

#[test]
fn wait_for_background_flush_concurrent_waiters_both_see_flush_failure() {
    let st = Arc::new(Mutex::new(BackgroundFlushState::default()));
    {
        let mut guard = st.lock().expect("flush lock");
        guard.in_progress = true;
        guard.join_handle = None;
    }
    let s1 = Arc::clone(&st);
    let s2 = Arc::clone(&st);
    let w1 = std::thread::spawn(move || wait_for_background_flush(&s1));
    let w2 = std::thread::spawn(move || wait_for_background_flush(&s2));
    std::thread::sleep(std::time::Duration::from_millis(20));
    finish_flush_thread(
        &st,
        Err(ENNError::InvalidParameter("stored flush err".to_string())),
    );
    let r1 = w1.join().expect("waiter 1");
    let r2 = w2.join().expect("waiter 2");
    assert!(
        r1.is_err(),
        "waiter 1 should see flush failure, got {r1:?}"
    );
    assert!(
        r2.is_err(),
        "waiter 2 should see flush failure, got {r2:?}"
    );
}

#[test]
fn wait_for_background_flush_concurrent_waiters_join_handle_path() {
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
