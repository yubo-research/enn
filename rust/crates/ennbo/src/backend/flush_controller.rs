//! Background soft-sync flush controller for disk BPANN backends.
//!
//! Job state lives beside the data lock so `wait_for_flush` / `schedule_background_flush`
//! never join or spawn while holding the data lock. Soft sync builds under a shared
//! read lock and publishes under a short write lock so search can proceed mid-flight.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::thread::{self, JoinHandle};

use crate::disk_bpann::DiskBpannEnnBackend;
use crate::error::ENNError;

fn disk_read(
    b: &Arc<RwLock<DiskBpannEnnBackend>>,
) -> Result<std::sync::RwLockReadGuard<'_, DiskBpannEnnBackend>, ENNError> {
    b.read()
        .map_err(|_| ENNError::InvalidParameter("disk backend lock poisoned".to_string()))
}

/// Disk backend handle: data RwLock + sibling flush controller.
pub struct DiskBackendHandle {
    data: Arc<RwLock<DiskBpannEnnBackend>>,
    job: Mutex<Option<JoinHandle<Result<(), String>>>>,
    last_error: Mutex<Option<String>>,
}

/// When true, the flush worker waits on [`TEST_FLUSH_WORKER_GATE`] after taking the
/// soft-sync read lock (test-only mid-flight observability).
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
            data: Arc::new(RwLock::new(inner)),
            job: Mutex::new(None),
            last_error: Mutex::new(None),
        }
    }

    pub fn data(&self) -> &Arc<RwLock<DiskBpannEnnBackend>> {
        &self.data
    }

    pub fn wait_for_flush(&self) -> Result<(), ENNError> {
        let job = self
            .job
            .lock()
            .map_err(|_| ENNError::InvalidParameter("flush job poisoned".to_string()))?
            .take();
        if let Some(handle) = job {
            match handle.join() {
                Ok(Ok(())) => {}
                Ok(Err(msg)) => {
                    *self
                        .last_error
                        .lock()
                        .map_err(|_| {
                            ENNError::InvalidParameter("flush last_error poisoned".to_string())
                        })? = Some(msg.clone());
                    return Err(ENNError::InvalidParameter(msg));
                }
                Err(_) => {
                    let msg = "background soft sync panicked".to_string();
                    *self
                        .last_error
                        .lock()
                        .map_err(|_| {
                            ENNError::InvalidParameter("flush last_error poisoned".to_string())
                        })? = Some(msg.clone());
                    return Err(ENNError::InvalidParameter(msg));
                }
            }
        }
        if let Some(msg) = self
            .last_error
            .lock()
            .map_err(|_| ENNError::InvalidParameter("flush last_error poisoned".to_string()))?
            .take()
        {
            return Err(ENNError::InvalidParameter(msg));
        }
        Ok(())
    }

    pub fn schedule_background_flush(&self) -> Result<(), ENNError> {
        {
            let err = self
                .last_error
                .lock()
                .map_err(|_| ENNError::InvalidParameter("flush last_error poisoned".to_string()))?;
            if let Some(msg) = err.as_ref() {
                return Err(ENNError::InvalidParameter(msg.clone()));
            }
        }

        let (defer, pending, soft, hard) = {
            let g = disk_read(&self.data)?;
            (
                g.defer_append_indexing_for_flush(),
                g.pending_unindexed_count(),
                g.pending_flush_threshold(),
                g.pending_hard_flush_threshold(),
            )
        };


        if defer && pending >= hard {
            self.wait_for_flush()?;
            let mut g = self
                .data
                .write()
                .map_err(|_| ENNError::InvalidParameter("disk backend lock poisoned".to_string()))?;

            if g.pending_unindexed_count() >= g.pending_hard_flush_threshold() {
                if let Some(built) = g.soft_sync_build_detached()? {
                    g.soft_sync_publish_detached(built)?;
                }
            }
            return Ok(());
        }

        {
            let job = self
                .job
                .lock()
                .map_err(|_| ENNError::InvalidParameter("flush job poisoned".to_string()))?;
            if job.is_some() {

                return Ok(());
            }
        }

        if !(defer && pending >= soft) {
            return Ok(());
        }

        let data = Arc::clone(&self.data);
        let handle = thread::spawn(move || {
            let built = {
                let g = data
                    .read()
                    .map_err(|_| "disk backend lock poisoned during soft sync".to_string())?;


                maybe_wait_test_flush_gate();
                g.soft_sync_build_detached()
                    .map_err(|e| e.to_string())?
            };
            if let Some(built) = built {
                let mut g = data
                    .write()
                    .map_err(|_| "disk backend lock poisoned during soft sync publish".to_string())?;
                g.soft_sync_publish_detached(built)
                    .map_err(|e| e.to_string())?;
            }
            Ok(())
        });

        let mut job = self
            .job
            .lock()
            .map_err(|_| ENNError::InvalidParameter("flush job poisoned".to_string()))?;
        if job.is_some() {

            drop(job);
            let _ = handle.join();
            return Ok(());
        }
        *job = Some(handle);
        Ok(())
    }

    pub fn persist_index_to_disk(&self) -> Result<(), ENNError> {
        self.wait_for_flush()?;
        self.data
            .write()
            .map_err(|_| ENNError::InvalidParameter("disk backend lock poisoned".to_string()))?
            .persist_index_to_disk()
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

    fn write_backend(
        data: &Arc<RwLock<DiskBpannEnnBackend>>,
    ) -> std::sync::RwLockWriteGuard<'_, DiskBpannEnnBackend> {
        data.write().expect("disk write")
    }

    fn disk_model(dir: &TempDir, n: usize, dim: usize) -> EpistemicNearestNeighbors {
        let mut model = EpistemicNearestNeighbors::new_empty(
            dim,
            1,
            IndexDriver::BpAnnDisk,
            EnnStorage::Disk,
            Some(dir.path().to_path_buf()),
            Some(1),
        )
        .expect("new_empty");
        let x = Array2::from_shape_fn((n, dim), |(i, j)| (i + j) as f64);
        let y = Array2::from_shape_fn((n, 1), |(i, _)| i as f64);
        model.add(&x.view(), &y.view(), None).expect("add");
        model
    }

    #[test]
    fn schedule_returns_while_soft_sync_in_flight() {
        let dir = TempDir::new().expect("tempdir");
        let model = disk_model(&dir, 8, 4);

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
    fn search_during_soft_sync_does_not_deadlock() {
        let dir = TempDir::new().expect("tempdir");
        let model = disk_model(&dir, 8, 4);
        let query = Array2::from_shape_fn((1, 4), |(_, j)| j as f64);

        let barrier = Arc::new(Barrier::new(2));
        arm_test_flush_worker_barrier(Arc::clone(&barrier));
        model.schedule_background_flush().expect("schedule");

        let t0 = Instant::now();
        let neighbors = model.neighbors(&query.view(), 1, false).expect("search");
        let elapsed = t0.elapsed();
        assert!(
            elapsed < Duration::from_secs(2),
            "search blocked during soft sync; took {elapsed:?}"
        );
        assert_eq!(neighbors.nrows(), 1);
        assert_eq!(neighbors.ncols(), 1);

        barrier.wait();
        model.backend.wait_for_flush().expect("wait");
        clear_test_flush_worker_barrier();
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
        handle.wait_for_flush().expect("still idle");
        drop(handle.job.lock().expect("job"));
        drop(handle.last_error.lock().expect("last_error"));
        maybe_wait_test_flush_gate();
    }

    #[test]
    fn schedule_while_job_running_is_single_slot_noop() {
        let dir = TempDir::new().expect("tempdir");
        let model = disk_model(&dir, 4, 2);

        let barrier = Arc::new(Barrier::new(2));
        arm_test_flush_worker_barrier(Arc::clone(&barrier));
        model.schedule_background_flush().expect("first schedule");
        model.schedule_background_flush().expect("second schedule while running");
        barrier.wait();
        model.backend.wait_for_flush().expect("wait");
        clear_test_flush_worker_barrier();
    }

    #[test]
    fn search_observes_pending_rows_while_soft_sync_in_flight() {
        let dir = TempDir::new().expect("tempdir");
        let model = disk_model(&dir, 6, 3);
        let query = Array2::from_shape_fn((1, 3), |(i, j)| ((5 + i) + j) as f64);

        let barrier = Arc::new(Barrier::new(2));
        arm_test_flush_worker_barrier(Arc::clone(&barrier));
        model.schedule_background_flush().expect("schedule");

        let neighbors = model
            .neighbors(&query.view(), 1, false)
            .expect("search while soft sync in flight");
        assert_eq!(neighbors[[0, 0]], 5);

        barrier.wait();
        model.backend.wait_for_flush().expect("wait");
        clear_test_flush_worker_barrier();
    }

    #[test]
    fn soft_sync_publish_matches_exclusive_ensure_metamorphic() {
        let dir_a = TempDir::new().expect("tempdir a");
        let dir_b = TempDir::new().expect("tempdir b");
        let a = DiskBackendHandle::new(
            crate::disk_bpann::DiskBpannEnnBackend::new_empty_with_flush_threshold(
                dir_a.path().to_path_buf(),
                3,
                1,
                100,
            )
            .expect("a"),
        );
        let b = DiskBackendHandle::new(
            crate::disk_bpann::DiskBpannEnnBackend::new_empty_with_flush_threshold(
                dir_b.path().to_path_buf(),
                3,
                1,
                100,
            )
            .expect("b"),
        );
        let x = Array2::from_shape_fn((12, 3), |(i, j)| (i * 3 + j) as f64);
        let y = Array2::from_shape_fn((12, 1), |(i, _)| i as f64);
        write_backend(a.data())
            .append_rows(&x.view(), &y.view(), None)
            .expect("append a");
        write_backend(b.data())
            .append_rows(&x.view(), &y.view(), None)
            .expect("append b");

        {
            let mut ga = write_backend(a.data());
            if let Some(built) = ga.soft_sync_build_detached().expect("build") {
                ga.soft_sync_publish_detached(built).expect("publish");
            }
        }
        write_backend(b.data())
            .ensure_index_sync(false, &ndarray::Array1::ones(3))
            .expect("ensure b");

        let q = Array2::from_shape_fn((4, 3), |(i, j)| ((i + 2) + j) as f64);
        let (da, ia) = disk_read(a.data())
            .expect("a read")
            .search(&q.view(), 3, false)
            .expect("search a");
        let (db, ib) = disk_read(b.data())
            .expect("b read")
            .search(&q.view(), 3, false)
            .expect("search b");
        assert_eq!(ia, ib);
        assert_eq!(da.shape(), db.shape());
        for (xa, xb) in da.iter().zip(db.iter()) {
            assert!((xa - xb).abs() < 1e-9, "dist mismatch {xa} vs {xb}");
        }
    }

    #[test]
    fn fuzz_concurrent_search_during_soft_sync_all_seeds() {
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;
        let seed = std::env::var("ENN_FUZZ_SEED")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(0xC0FFEE_u64);
        eprintln!("fuzz_concurrent_search_during_soft_sync_all_seeds seed={seed}");
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        for trial in 0..8 {
            let dir = TempDir::new().expect("tempdir");
            let n = rng.gen_range(4..=16);
            let dim = rng.gen_range(2..=5);
            let model = disk_model(&dir, n, dim);
            let qrow = rng.gen_range(0..n);
            let query = Array2::from_shape_fn((1, dim), |(_, j)| (qrow + j) as f64);

            let barrier = Arc::new(Barrier::new(2));
            arm_test_flush_worker_barrier(Arc::clone(&barrier));
            model.schedule_background_flush().expect("schedule");

            let t0 = Instant::now();
            let neighbors = model
                .neighbors(&query.view(), 1, false)
                .unwrap_or_else(|e| panic!("trial={trial} search: {e}"));
            assert!(
                t0.elapsed() < Duration::from_secs(2),
                "trial={trial} search timeout"
            );
            assert_eq!(neighbors.ncols(), 1);
            assert!(neighbors[[0, 0]] < n);

            barrier.wait();
            model.backend.wait_for_flush().expect("wait");
            clear_test_flush_worker_barrier();
        }
    }

    #[test]
    fn persist_index_to_disk_after_soft_sync_writes_pages() {
        let dir = TempDir::new().expect("tempdir");
        let model = disk_model(&dir, 6, 2);
        model.schedule_background_flush().expect("schedule");
        model.backend.wait_for_flush().expect("wait soft");
        if let crate::backend::EnnBackend::Disk(handle) = &model.backend {
            handle.persist_index_to_disk().expect("hard persist");
        } else {
            panic!("expected disk backend");
        }
        assert!(dir.path().join("index/pages.bin").exists());
    }

    #[test]
    fn flush_last_error_blocks_schedule_and_wait() {
        let dir = TempDir::new().expect("tempdir");
        let handle = DiskBackendHandle::new(
            crate::disk_bpann::DiskBpannEnnBackend::new_empty_with_flush_threshold(
                dir.path().to_path_buf(),
                2,
                1,
                1,
            )
            .expect("backend"),
        );
        {
            *handle.last_error.lock().expect("last_error") =
                Some("injected soft sync failure".to_string());
        }
        let schedule_err = handle.schedule_background_flush().expect_err("schedule");
        assert!(schedule_err.to_string().contains("injected soft sync failure"));
        let wait_err = handle.wait_for_flush().expect_err("wait");
        assert!(wait_err.to_string().contains("injected soft sync failure"));
        handle.wait_for_flush().expect("cleared");
        let x = Array2::from_shape_fn((2, 2), |(i, j)| (i + j) as f64);
        let y = Array2::from_shape_fn((2, 1), |(i, _)| i as f64);
        write_backend(handle.data())
            .append_rows(&x.view(), &y.view(), None)
            .expect("append");
        handle.schedule_background_flush().expect("schedule ok");
        handle.wait_for_flush().expect("drain");
        handle.persist_index_to_disk().expect("persist");
    }

    #[test]
    fn schedule_soft_band_stays_async() {

        let dir = TempDir::new().expect("tempdir");
        let handle = DiskBackendHandle::new(
            crate::disk_bpann::DiskBpannEnnBackend::new_empty_with_flush_thresholds(
                dir.path().to_path_buf(),
                2,
                1,
                2,
                5,
            )
            .expect("backend"),
        );
        let x = Array2::from_shape_fn((3, 2), |(i, j)| (i + j) as f64);
        let y = Array2::from_shape_fn((3, 1), |(i, _)| i as f64);
        write_backend(handle.data())
            .append_rows(&x.view(), &y.view(), None)
            .expect("append");
        assert_eq!(
            disk_read(handle.data())
                .expect("read")
                .pending_unindexed_count(),
            3
        );

        let barrier = Arc::new(Barrier::new(2));
        arm_test_flush_worker_barrier(Arc::clone(&barrier));
        let t0 = Instant::now();
        handle.schedule_background_flush().expect("schedule");
        assert!(
            t0.elapsed() < Duration::from_millis(200),
            "soft-band schedule must return without waiting"
        );

        {
            let job = handle.job.lock().expect("job");
            assert!(job.is_some(), "soft-band must leave an in-flight job");
        }
        barrier.wait();
        handle.wait_for_flush().expect("wait");
        clear_test_flush_worker_barrier();
        assert_eq!(
            disk_read(handle.data())
                .expect("read")
                .pending_unindexed_count(),
            0
        );
    }

    #[test]
    fn schedule_hard_path_drains_pending_synchronously() {
        let dir = TempDir::new().expect("tempdir");
        let handle = DiskBackendHandle::new(
            crate::disk_bpann::DiskBpannEnnBackend::new_empty_with_flush_thresholds(
                dir.path().to_path_buf(),
                2,
                1,
                2,
                100,
            )
            .expect("backend"),
        );
        let x = Array2::from_shape_fn((6, 2), |(i, j)| (i + j) as f64);
        let y = Array2::from_shape_fn((6, 1), |(i, _)| i as f64);
        write_backend(handle.data())
            .append_rows(&x.view(), &y.view(), None)
            .expect("append");
        assert_eq!(
            disk_read(handle.data())
                .expect("read")
                .pending_unindexed_count(),
            6
        );

        write_backend(handle.data()).reconfigure_flush_thresholds(2, 5);
        handle.schedule_background_flush().expect("hard schedule");
        assert!(handle.job.lock().expect("job").is_none());
        assert_eq!(
            disk_read(handle.data())
                .expect("read")
                .pending_unindexed_count(),
            0
        );
        assert!(!dir.path().join("index/pages.bin").exists());
    }

    #[test]
    fn append_past_hard_without_schedule_syncs_on_add() {
        let dir = TempDir::new().expect("tempdir");
        let handle = DiskBackendHandle::new(
            crate::disk_bpann::DiskBpannEnnBackend::new_empty_with_flush_thresholds(
                dir.path().to_path_buf(),
                2,
                1,
                2,
                5,
            )
            .expect("backend"),
        );
        let x = Array2::from_shape_fn((5, 2), |(i, j)| (i + j) as f64);
        let y = Array2::from_shape_fn((5, 1), |(i, _)| i as f64);
        write_backend(handle.data())
            .append_rows(&x.view(), &y.view(), None)
            .expect("append");
        assert_eq!(
            disk_read(handle.data())
                .expect("read")
                .pending_unindexed_count(),
            0
        );

        assert!(handle.job.lock().expect("job").is_none());
    }
}
