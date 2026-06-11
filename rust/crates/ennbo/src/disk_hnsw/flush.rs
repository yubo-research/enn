//! Background index flush coordination for disk HNSW backends.

use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use crate::backend::DiskEnnBackend;
use crate::error::ENNError;

/// Per-backend background flush lifecycle (separate from the disk data mutex).
pub struct BackgroundFlushState {
    pub in_progress: bool,
    pub join_handle: Option<JoinHandle<()>>,
    pub error: Option<ENNError>,
    pub barrier: Arc<FlushTestBarrier>,
    done_cv: Arc<Condvar>,
    fail_next: bool,
}

impl Default for BackgroundFlushState {
    fn default() -> Self {
        Self {
            in_progress: false,
            join_handle: None,
            error: None,
            barrier: Arc::new(FlushTestBarrier::default()),
            done_cv: Arc::new(Condvar::new()),
            fail_next: false,
        }
    }
}

impl BackgroundFlushState {
    #[doc(hidden)]
    pub fn inject_failure(&mut self) {
        self.fail_next = true;
    }
}

#[doc(hidden)]
#[derive(Default)]
pub struct FlushTestBarrier {
    hold: Mutex<bool>,
    cv: std::sync::Condvar,
}

#[doc(hidden)]
impl FlushTestBarrier {
    pub fn set_hold(&self, hold: bool) {
        let mut g = self.hold.lock().expect("barrier hold mutex poisoned");
        *g = hold;
        if !hold {
            self.cv.notify_all();
        }
    }

    pub fn is_holding(&self) -> bool {
        *self.hold.lock().expect("barrier hold mutex poisoned")
    }

    pub fn wait_if_holding(&self) {
        let mut g = self.hold.lock().expect("barrier hold mutex poisoned");
        while *g {
            g = self.cv.wait(g).expect("barrier condvar poisoned");
        }
    }
}

#[doc(hidden)]
pub fn lock_flush_state(
    flush: &Arc<Mutex<BackgroundFlushState>>,
) -> Result<std::sync::MutexGuard<'_, BackgroundFlushState>, ENNError> {
    flush
        .lock()
        .map_err(|_| ENNError::InvalidParameter("flush state mutex poisoned".to_string()))
}

pub fn wait_for_background_flush(
    flush: &Arc<Mutex<BackgroundFlushState>>,
) -> Result<(), ENNError> {
    let mut st = lock_flush_state(flush)?;
    if !st.in_progress {
        if let Some(err) = st.error.clone() {
            return Err(err);
        }
        return Ok(());
    }
    if let Some(h) = st.join_handle.take() {
        drop(st);
        if h.join().is_err() {
            finish_flush_thread(
                flush,
                Err(ENNError::InvalidParameter(
                    "background flush thread panicked".to_string(),
                )),
            );
            return Err(ENNError::InvalidParameter(
                "background flush thread panicked".to_string(),
            ));
        }
        st = lock_flush_state(flush)?;
        if let Some(err) = st.error.clone() {
            return Err(err);
        }
        return Ok(());
    }
    let cv = Arc::clone(&st.done_cv);
    while st.in_progress {
        st = cv
            .wait_timeout(st, Duration::from_millis(5))
            .map_err(|_| ENNError::InvalidParameter("flush state mutex poisoned".to_string()))?
            .0;
    }
    if let Some(err) = st.error.clone() {
        return Err(err);
    }
    Ok(())
}

pub fn finish_flush_thread(
    flush: &Arc<Mutex<BackgroundFlushState>>,
    result: Result<(), ENNError>,
) {
    let mut st = flush.lock().expect("flush state mutex poisoned");
    st.in_progress = false;
    st.join_handle = None;
    match result {
        Ok(()) => st.error = None,
        Err(e) => st.error = Some(e),
    }
    st.done_cv.notify_all();
}

#[doc(hidden)]
pub fn run_flush_body(
    flush: &Arc<Mutex<BackgroundFlushState>>,
    disk_arc: &Arc<Mutex<DiskEnnBackend>>,
) -> Result<(), ENNError> {
    {
        let barrier = {
            let st = flush.lock().map_err(|_| {
                ENNError::InvalidParameter("flush state mutex poisoned".to_string())
            })?;
            Arc::clone(&st.barrier)
        };
        barrier.wait_if_holding();
    }
    {
        let mut st = flush.lock().map_err(|_| {
            ENNError::InvalidParameter("flush state mutex poisoned".to_string())
        })?;
        if st.fail_next {
            st.fail_next = false;
            return Err(ENNError::InvalidParameter(
                "injected background flush failure".to_string(),
            ));
        }
    }
    let mut guard = disk_arc.lock().map_err(|_| {
        ENNError::InvalidParameter("disk backend mutex poisoned".to_string())
    })?;
    let DiskEnnBackend::Hnsw(ref mut backend) = *guard;
    backend.flush_pending_index_rows()
}

pub fn try_schedule_background_flush(
    flush: &Arc<Mutex<BackgroundFlushState>>,
    disk_arc: Arc<Mutex<DiskEnnBackend>>,
) -> Result<(), ENNError> {
    let mut st = flush
        .lock()
        .map_err(|_| ENNError::InvalidParameter("flush state mutex poisoned".to_string()))?;
    if st.in_progress {
        return Ok(());
    }
    st.in_progress = true;

    let use_background_thread = st.barrier.is_holding();
    if use_background_thread {
        let flush_arc = Arc::clone(flush);
        let handle = std::thread::spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                run_flush_body(&flush_arc, &disk_arc)
            }))
            .unwrap_or_else(|_| {
                Err(ENNError::InvalidParameter(
                    "background flush thread panicked".to_string(),
                ))
            });
            finish_flush_thread(&flush_arc, result);
        });
        st.join_handle = Some(handle);
        return Ok(());
    }

    let flush_arc = Arc::clone(flush);
    drop(st);
    let result = run_flush_body(&flush_arc, &disk_arc);
    finish_flush_thread(&flush_arc, result);
    Ok(())
}

#[cfg(test)]
mod flush_inline_tests {
    use super::*;
    use crate::disk_hnsw::DiskHnswEnnBackend;
    use ndarray::array;
    use tempfile::TempDir;

    #[test]
    fn wait_for_background_flush_waits_for_in_progress_flag() {
        let st = Arc::new(Mutex::new(BackgroundFlushState::default()));
        {
            let mut guard = st.lock().expect("lock");
            guard.in_progress = true;
        }
        let st2 = Arc::clone(&st);
        let waiter = std::thread::spawn(move || {
            std::thread::sleep(std::time::Duration::from_millis(20));
            st2.lock().expect("lock").in_progress = false;
        });
        wait_for_background_flush(&st).expect("wait");
        waiter.join().expect("join");
    }

    #[test]
    fn try_schedule_runs_inline_when_barrier_open() {
        let dir = TempDir::new().expect("tempdir");
        let mut backend = DiskHnswEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1)
            .expect("backend")
            .with_pending_flush_threshold(2)
            .with_defer_append_indexing(true);
        backend
            .append_rows(
                &array![[0.0, 0.0], [1.0, 0.0]].view(),
                &array![[0.0], [1.0]].view(),
                None,
            )
            .expect("append");
        let disk_arc = Arc::new(Mutex::new(DiskEnnBackend::Hnsw(backend)));
        let flush = {
            let guard = disk_arc.lock().expect("disk lock");
            let DiskEnnBackend::Hnsw(ref b) = *guard;
            b.flush_arc()
        };
        try_schedule_background_flush(&flush, Arc::clone(&disk_arc)).expect("inline flush");
        wait_for_background_flush(&flush).expect("wait after inline");
        let guard = disk_arc.lock().expect("disk lock");
        let DiskEnnBackend::Hnsw(ref b) = *guard;
        assert_eq!(b.indexed_rows(), b.len());
    }

    #[test]
    fn try_schedule_uses_background_thread_when_barrier_held() {
        let dir = TempDir::new().expect("tempdir");
        let mut backend = DiskHnswEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1)
            .expect("backend")
            .with_pending_flush_threshold(2)
            .with_defer_append_indexing(true);
        backend
            .append_rows(
                &array![[0.0, 0.0], [1.0, 0.0]].view(),
                &array![[0.0], [1.0]].view(),
                None,
            )
            .expect("append");
        let disk_arc = Arc::new(Mutex::new(DiskEnnBackend::Hnsw(backend)));
        let flush = {
            let guard = disk_arc.lock().expect("disk lock");
            let DiskEnnBackend::Hnsw(ref b) = *guard;
            b.flush_test_barrier_hold(true);
            b.flush_arc()
        };
        try_schedule_background_flush(&flush, Arc::clone(&disk_arc)).expect("schedule");
        assert!(flush.lock().expect("flush lock").in_progress);
        flush.lock().expect("flush lock").barrier.set_hold(false);
        wait_for_background_flush(&flush).expect("wait");
        let guard = disk_arc.lock().expect("disk lock");
        let DiskEnnBackend::Hnsw(ref b) = *guard;
        assert_eq!(b.indexed_rows(), b.len());
    }

    #[test]
    fn try_schedule_inline_propagates_injected_failure() {
        let dir = TempDir::new().expect("tempdir");
        let mut backend = DiskHnswEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1)
            .expect("backend")
            .with_pending_flush_threshold(2)
            .with_defer_append_indexing(true);
        backend
            .append_rows(
                &array![[0.0, 0.0], [1.0, 0.0]].view(),
                &array![[0.0], [1.0]].view(),
                None,
            )
            .expect("append");
        backend.inject_next_flush_failure();
        let disk_arc = Arc::new(Mutex::new(DiskEnnBackend::Hnsw(backend)));
        let flush = {
            let guard = disk_arc.lock().expect("disk lock");
            let DiskEnnBackend::Hnsw(ref b) = *guard;
            b.flush_arc()
        };
        try_schedule_background_flush(&flush, Arc::clone(&disk_arc)).expect("schedule");
        let err = wait_for_background_flush(&flush).expect_err("injected");
        assert!(err.to_string().contains("injected background flush failure"));
    }
}
