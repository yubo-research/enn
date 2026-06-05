//! Background index flush coordination for disk HNSW backends.

use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use crate::backend::DiskEnnBackend;
use crate::error::ENNError;

/// Per-backend background flush lifecycle (separate from the disk data mutex).
pub struct BackgroundFlushState {
    pub in_progress: bool,
    pub join_handle: Option<JoinHandle<()>>,
    pub error: Option<ENNError>,
    pub barrier: Arc<FlushTestBarrier>,
    fail_next: bool,
}

impl Default for BackgroundFlushState {
    fn default() -> Self {
        Self {
            in_progress: false,
            join_handle: None,
            error: None,
            barrier: Arc::new(FlushTestBarrier::default()),
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
    let handle = {
        let mut st = lock_flush_state(flush)?;
        st.join_handle.take()
    };
    if let Some(h) = handle {
        h.join().map_err(|_| {
            ENNError::InvalidParameter("background flush thread panicked".to_string())
        })?;
    } else {
        loop {
            let st = lock_flush_state(flush)?;
            if !st.in_progress {
                break;
            }
            drop(st);
            std::thread::yield_now();
        }
    }
    let mut st = lock_flush_state(flush)?;
    if let Some(err) = st.error.take() {
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
    match result {
        Ok(()) => st.error = None,
        Err(e) => st.error = Some(e),
    }
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
    let flush_arc = Arc::clone(flush);
    let handle = std::thread::spawn(move || {
        let result: Result<(), ENNError> = (|| {
            {
                let barrier = {
                    let st = flush_arc.lock().expect("flush state mutex poisoned");
                    Arc::clone(&st.barrier)
                };
                barrier.wait_if_holding();
            }
            {
                let mut st = flush_arc.lock().expect("flush state mutex poisoned");
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
        })();
        finish_flush_thread(&flush_arc, result);
    });
    st.join_handle = Some(handle);
    Ok(())
}
