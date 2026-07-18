//! Background soft-sync flush controller for disk BPANN backends.
//!
//! Job state lives beside the data mutex so `wait_for_flush` / `schedule_background_flush`
//! never join or spawn while holding the data lock.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use crate::disk_bpann::DiskBpannEnnBackend;
use crate::error::ENNError;

fn disk_lock(
    b: &Arc<Mutex<DiskBpannEnnBackend>>,
) -> Result<std::sync::MutexGuard<'_, DiskBpannEnnBackend>, ENNError> {
    b.lock()
        .map_err(|_| ENNError::InvalidParameter("disk backend mutex poisoned".to_string()))
}

fn flush_lock(
    b: &Mutex<FlushState>,
) -> Result<std::sync::MutexGuard<'_, FlushState>, ENNError> {
    b.lock()
        .map_err(|_| ENNError::InvalidParameter("flush controller mutex poisoned".to_string()))
}

struct FlushState {
    job: Option<JoinHandle<Result<(), String>>>,
    last_error: Option<String>,
}

/// Disk backend handle: data mutex + sibling flush controller.
pub struct DiskBackendHandle {
    data: Arc<Mutex<DiskBpannEnnBackend>>,
    flush: Mutex<FlushState>,
}

/// When true, the flush worker waits on [`TEST_FLUSH_WORKER_GATE`] after start
/// (test-only async schedule observability).
static TEST_FLUSH_DELAY_ARMED: AtomicBool = AtomicBool::new(false);
static TEST_FLUSH_WORKER_GATE: Mutex<Option<Arc<std::sync::Barrier>>> = Mutex::new(None);

#[cfg(test)]
pub(crate) fn arm_test_flush_worker_barrier(barrier: Arc<std::sync::Barrier>) {
    TEST_FLUSH_DELAY_ARMED.store(true, Ordering::SeqCst);
    *TEST_FLUSH_WORKER_GATE.lock().expect("gate") = Some(barrier);
}

#[cfg(test)]
pub(crate) fn clear_test_flush_worker_barrier() {
    TEST_FLUSH_DELAY_ARMED.store(false, Ordering::SeqCst);
    *TEST_FLUSH_WORKER_GATE.lock().expect("gate") = None;
}

fn maybe_wait_test_flush_gate() {
    if !TEST_FLUSH_DELAY_ARMED.load(Ordering::SeqCst) {
        return;
    }
    if let Ok(mut guard) = TEST_FLUSH_WORKER_GATE.lock() {
        if let Some(barrier) = guard.take() {
            TEST_FLUSH_DELAY_ARMED.store(false, Ordering::SeqCst);
            barrier.wait();
        }
    }
}

impl DiskBackendHandle {
    pub fn new(inner: DiskBpannEnnBackend) -> Self {
        Self {
            data: Arc::new(Mutex::new(inner)),
            flush: Mutex::new(FlushState {
                job: None,
                last_error: None,
            }),
        }
    }

    pub fn data(&self) -> &Arc<Mutex<DiskBpannEnnBackend>> {
        &self.data
    }

    pub fn wait_for_flush(&self) -> Result<(), ENNError> {
        let job = {
            let mut flush = flush_lock(&self.flush)?;
            flush.job.take()
        };
        if let Some(handle) = job {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(msg)) => {
                    let mut flush = flush_lock(&self.flush)?;
                    flush.last_error = Some(msg.clone());
                    return Err(ENNError::InvalidParameter(msg));
                }
                Err(_) => {
                    let msg = "background soft sync panicked".to_string();
                    let mut flush = flush_lock(&self.flush)?;
                    flush.last_error = Some(msg.clone());
                    return Err(ENNError::InvalidParameter(msg));
                }
            }
        }
        let mut flush = flush_lock(&self.flush)?;
        if let Some(msg) = flush.last_error.take() {
            return Err(ENNError::InvalidParameter(msg));
        }
        Ok(())
    }

    pub fn schedule_background_flush(&self) -> Result<(), ENNError> {
        {
            let flush = flush_lock(&self.flush)?;
            if let Some(msg) = flush.last_error.as_ref() {
                return Err(ENNError::InvalidParameter(msg.clone()));
            }
            if flush.job.is_some() {
                // Single-slot: add waits before append, so pending cannot grow mid-flight.
                return Ok(());
            }
        }

        let should_run = {
            let g = disk_lock(&self.data)?;
            g.defer_append_indexing_for_flush()
                && g.pending_unindexed_count() >= g.pending_flush_threshold()
        };
        if !should_run {
            return Ok(());
        }

        let data = Arc::clone(&self.data);
        let handle = thread::spawn(move || {
            maybe_wait_test_flush_gate();
            let mut g = data
                .lock()
                .map_err(|_| "disk backend mutex poisoned during soft sync".to_string())?;
            g.soft_sync_inner().map_err(|e| e.to_string())
        });

        let mut flush = flush_lock(&self.flush)?;
        if flush.job.is_some() {
            // Extremely unlikely race; join the spare handle to avoid leak.
            drop(flush);
            let _ = handle.join();
            return Ok(());
        }
        flush.job = Some(handle);
        Ok(())
    }

    pub fn persist_index_to_disk(&self) -> Result<(), ENNError> {
        self.wait_for_flush()?;
        disk_lock(&self.data)?.persist_index_to_disk()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::EnnStorage;
    use crate::index::IndexDriver;
    use crate::EpistemicNearestNeighbors;
    use ndarray::Array2;
    use std::sync::Barrier;
    use std::time::{Duration, Instant};
    use tempfile::TempDir;

    #[test]
    fn schedule_returns_while_soft_sync_in_flight() {
        let dir = TempDir::new().expect("tempdir");
        let mut model = EpistemicNearestNeighbors::new_empty(
            4,
            1,
            IndexDriver::BpAnnDisk,
            EnnStorage::Disk,
            Some(dir.path().to_path_buf()),
            Some(1),
        )
        .expect("new_empty");
        let x = Array2::from_shape_fn((8, 4), |(i, j)| (i + j) as f64);
        let y = Array2::from_shape_fn((8, 1), |(i, _)| i as f64);
        model.add(&x.view(), &y.view(), None).expect("add");

        let barrier = Arc::new(Barrier::new(2));
        arm_test_flush_worker_barrier(Arc::clone(&barrier));

        let t0 = Instant::now();
        model.schedule_background_flush().expect("schedule");
        let schedule_ms = t0.elapsed();
        assert!(
            schedule_ms < Duration::from_millis(200),
            "schedule must return without waiting for soft sync; took {schedule_ms:?}"
        );

        barrier.wait();
        model.backend.wait_for_flush().expect("wait");
        clear_test_flush_worker_barrier();

        assert!(!dir.path().join("index/pages.bin").exists());
    }

    #[test]
    fn wait_idle_and_schedule_below_threshold_are_noop() {
        let dir = TempDir::new().expect("tempdir");
        let handle = DiskBackendHandle::new(
            crate::disk_bpann::DiskBpannEnnBackend::new_empty_with_flush_threshold(
                dir.path().to_path_buf(),
                2,
                1,
                100,
            )
            .expect("backend"),
        );
        handle.wait_for_flush().expect("idle wait");
        handle.schedule_background_flush().expect("below threshold");
        // Touch flush_lock / FlushState via a second idle wait after schedule.
        handle.wait_for_flush().expect("still idle");
        drop(flush_lock(&handle.flush).expect("flush_lock"));
        maybe_wait_test_flush_gate();
    }

    #[test]
    fn schedule_while_job_running_is_single_slot_noop() {
        let dir = TempDir::new().expect("tempdir");
        let mut model = EpistemicNearestNeighbors::new_empty(
            2,
            1,
            IndexDriver::BpAnnDisk,
            EnnStorage::Disk,
            Some(dir.path().to_path_buf()),
            Some(1),
        )
        .expect("new_empty");
        let x = Array2::from_shape_fn((4, 2), |(i, j)| (i + j) as f64);
        let y = Array2::from_shape_fn((4, 1), |(i, _)| i as f64);
        model.add(&x.view(), &y.view(), None).expect("add");

        let barrier = Arc::new(Barrier::new(2));
        arm_test_flush_worker_barrier(Arc::clone(&barrier));
        model.schedule_background_flush().expect("first schedule");
        model.schedule_background_flush().expect("second schedule while running");
        barrier.wait();
        model.backend.wait_for_flush().expect("wait");
        clear_test_flush_worker_barrier();
    }
}
