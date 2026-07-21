//! ENN storage and indexing backends.

pub mod disk_observation;
mod flush_controller;
mod in_memory;
pub(crate) mod row_storage;

pub use in_memory::InMemoryEnnBackend;
pub use crate::disk_bpann::DiskBpannEnnBackend;
pub(crate) use flush_controller::DiskBackendHandle;

use ndarray::{Array1, Array2, ArrayView2};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use crate::error::ENNError;
use crate::index::{ENNIndex, IndexDriver};

/// Gathered training rows: `x`, `y`, optional `yvar`.
pub(crate) type TrainRowsAtResult = (Array2<f64>, Array2<f64>, Option<Array2<f64>>);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnnStorage {
    InMemory,
    Disk,
}

impl EnnStorage {
    pub fn from_env() -> Self {
        if std::env::var("ENN_WORK_DIR").is_ok() {
            Self::Disk
        } else {
            Self::InMemory
        }
    }

    pub fn work_dir_from_env() -> Option<PathBuf> {
        std::env::var("ENN_WORK_DIR").ok().map(PathBuf::from)
    }
}

pub enum EnnBackend {
    InMemory(Box<InMemoryEnnBackend>),
    Disk(DiskBackendHandle),
}

fn disk_read<'a>(
    b: &'a Arc<RwLock<DiskBpannEnnBackend>>,
) -> Result<std::sync::RwLockReadGuard<'a, DiskBpannEnnBackend>, ENNError> {
    b.read()
        .map_err(|_| ENNError::InvalidParameter("disk backend lock poisoned".to_string()))
}

fn disk_write<'a>(
    b: &'a Arc<RwLock<DiskBpannEnnBackend>>,
) -> Result<std::sync::RwLockWriteGuard<'a, DiskBpannEnnBackend>, ENNError> {
    b.write()
        .map_err(|_| ENNError::InvalidParameter("disk backend lock poisoned".to_string()))
}

pub(crate) fn persist_enn_backend_index(backend: &EnnBackend) -> Result<(), ENNError> {
    match backend {
        EnnBackend::InMemory(_) => Ok(()),
        EnnBackend::Disk(handle) => handle.persist_index_to_disk(),
    }
}

impl EnnBackend {
    pub fn new_in_memory(
        train_x: Array2<f64>,
        train_y: Array2<f64>,
        train_yvar: Option<Array2<f64>>,
        scale_x: bool,
        x_scale: Array1<f64>,
        driver: IndexDriver,
    ) -> Result<Self, ENNError> {
        Ok(Self::InMemory(Box::new(InMemoryEnnBackend::new(
            train_x, train_y, train_yvar, scale_x, x_scale, driver,
        )?)))
    }

    pub fn new_disk(
        work_dir: PathBuf,
        train_x: Array2<f64>,
        train_y: Array2<f64>,
        train_yvar: Option<Array2<f64>>,
        scale_x: bool,
        x_scale: Array1<f64>,
        driver: IndexDriver,
    ) -> Result<Self, ENNError> {
        if driver != IndexDriver::BpAnnDisk {
            return Err(ENNError::InvalidParameter(
                "Disk storage requires IndexDriver::BpAnnDisk".to_string(),
            ));
        }
        let inner = DiskBpannEnnBackend::new(
            work_dir,
            train_x,
            train_y,
            train_yvar,
            scale_x,
            x_scale,
            driver,
        )?;
        Ok(Self::Disk(DiskBackendHandle::new(inner)))
    }

    pub fn new_empty(
        num_dim: usize,
        num_metrics: usize,
        driver: IndexDriver,
        storage: EnnStorage,
        work_dir: Option<PathBuf>,
        pending_flush_threshold: Option<usize>,
    ) -> Result<Self, ENNError> {
        match storage {
            EnnStorage::InMemory => Ok(Self::InMemory(Box::new(InMemoryEnnBackend::new_empty(
                num_dim, num_metrics, driver,
            )?))),
            EnnStorage::Disk => {
                let dir = work_dir.or_else(EnnStorage::work_dir_from_env).ok_or_else(|| {
                    ENNError::InvalidParameter(
                        "Disk storage requires work_dir or ENN_WORK_DIR".to_string(),
                    )
                })?;
                if driver != IndexDriver::BpAnnDisk {
                    return Err(ENNError::InvalidParameter(
                        "Disk storage requires IndexDriver::BpAnnDisk".to_string(),
                    ));
                }
                let inner = match pending_flush_threshold {
                    Some(threshold) => DiskBpannEnnBackend::new_empty_with_flush_threshold(
                        dir,
                        num_dim,
                        num_metrics,
                        threshold,
                    )?,
                    None => DiskBpannEnnBackend::new_empty(dir, num_dim, num_metrics)?,
                };
                Ok(Self::Disk(DiskBackendHandle::new(inner)))
            }
        }
    }

    pub fn wait_for_flush(&self) -> Result<(), ENNError> {
        match self {
            Self::InMemory(_) => Ok(()),
            Self::Disk(handle) => handle.wait_for_flush(),
        }
    }

    pub fn schedule_background_flush(&self) -> Result<(), ENNError> {
        match self {
            Self::InMemory(_) => Ok(()),
            Self::Disk(handle) => handle.schedule_background_flush(),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::InMemory(b) => b.len(),
            Self::Disk(h) => disk_read(h.data()).map(|g| g.len()).unwrap_or(0),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn num_dim(&self) -> usize {
        match self {
            Self::InMemory(b) => b.num_dim(),
            Self::Disk(h) => disk_read(h.data()).map(|g| g.num_dim()).unwrap_or(0),
        }
    }

    pub fn num_metrics(&self) -> usize {
        match self {
            Self::InMemory(b) => b.num_metrics(),
            Self::Disk(h) => disk_read(h.data()).map(|g| g.num_metrics()).unwrap_or(0),
        }
    }

    pub fn driver(&self) -> IndexDriver {
        match self {
            Self::InMemory(b) => b.driver(),
            Self::Disk(h) => disk_read(h.data())
                .map(|g| g.driver())
                .unwrap_or(IndexDriver::BpAnnDisk),
        }
    }

    pub fn mark_index_stale(&self) {
        match self {
            Self::InMemory(b) => b.mark_index_stale(),
            Self::Disk(h) => {
                if let Ok(mut g) = disk_write(h.data()) {
                    g.mark_index_stale();
                }
            }
        }
    }

    pub fn is_index_stale(&self) -> bool {
        match self {
            Self::InMemory(b) => b.is_index_stale(),
            Self::Disk(h) => disk_read(h.data())
                .map(|g| g.is_index_stale())
                .unwrap_or(false),
        }
    }

    pub fn append_rows(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Result<(), ENNError> {
        match self {
            Self::InMemory(b) => b.append_rows(x, y, yvar),
            Self::Disk(h) => disk_write(h.data())?.append_rows(x, y, yvar),
        }
    }

    pub fn defer_index_sync_for_search(&self) -> bool {
        match self {
            Self::InMemory(_) => false,
            Self::Disk(h) => disk_read(h.data())
                .map(|g| g.defer_index_sync_for_search())
                .unwrap_or(false),
        }
    }

    pub fn ensure_index_sync(
        &self,
        scale_x: bool,
        x_scale: &Array1<f64>,
    ) -> Result<(), ENNError> {
        self.wait_for_flush()?;
        match self {
            Self::InMemory(b) => b.ensure_index_sync(scale_x, x_scale),
            Self::Disk(h) => disk_write(h.data())?.ensure_index_sync(scale_x, x_scale),
        }
    }

    pub fn train_rows_at(
        &self,
        indices: &[usize],
    ) -> Result<TrainRowsAtResult, ENNError> {
        match self {
            Self::InMemory(b) => b.train_rows_at(indices),
            Self::Disk(h) => disk_read(h.data())?.train_rows_at(indices),
        }
    }

    pub fn row_x(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        match self {
            Self::InMemory(b) => b.row_x(i),
            Self::Disk(h) => disk_read(h.data())?.row_x(i),
        }
    }

    pub fn row_y(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        match self {
            Self::InMemory(b) => b.row_y(i),
            Self::Disk(h) => disk_read(h.data())?.row_y(i),
        }
    }

    pub fn row_yvar(&self, i: usize) -> Result<Option<Array1<f64>>, ENNError> {
        match self {
            Self::InMemory(b) => b.row_yvar(i),
            Self::Disk(h) => disk_read(h.data())?.row_yvar(i),
        }
    }

    pub fn search(
        &self,
        x: &ArrayView2<f64>,
        search_k: i32,
        exclude_nearest: bool,
    ) -> Result<(Array2<f64>, Array2<i64>), ENNError> {
        match self {
            Self::InMemory(b) => b.search(x, search_k, exclude_nearest),
            Self::Disk(h) => disk_read(h.data())?.search(x, search_k, exclude_nearest),
        }
    }

    pub fn index_memory_bytes(&self) -> Result<usize, ENNError> {
        match self {
            Self::InMemory(b) => b.index_memory_bytes(),
            Self::Disk(h) => disk_read(h.data())?.index_memory_bytes(),
        }
    }

    pub fn index_len(&self) -> usize {
        match self {
            Self::InMemory(b) => b.index_len(),
            Self::Disk(h) => disk_read(h.data()).map(|g| g.len()).unwrap_or(0),
        }
    }

    pub fn in_memory_index(&self) -> Option<&ENNIndex> {
        match self {
            Self::InMemory(b) => Some(b.index()),
            Self::Disk(_) => None,
        }
    }

    pub fn in_memory_train_x_view(&self) -> Option<ndarray::ArrayView2<'_, f64>> {
        match self {
            Self::InMemory(b) => Some(b.train_x_view()),
            Self::Disk(_) => None,
        }
    }

    pub fn in_memory_train_y_view(&self) -> Option<ndarray::ArrayView2<'_, f64>> {
        match self {
            Self::InMemory(b) => Some(b.train_y_view()),
            Self::Disk(_) => None,
        }
    }
}

/// Drop faulted observation mmap pages from process RSS (disk backend only).
pub(crate) fn release_enn_observation_pages(backend: &EnnBackend) -> Result<(), ENNError> {
    match backend {
        EnnBackend::InMemory(_) => Ok(()),
        EnnBackend::Disk(h) => disk_write(h.data())?.release_observation_pages(),
    }
}

impl Drop for EnnBackend {
    fn drop(&mut self) {
        if let Err(e) = persist_enn_backend_index(self) {
            eprintln!("ennbo: persist_index_to_disk on drop failed: {e}");
        }
    }
}

#[cfg(test)]
mod backend_dispatch_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn in_memory_enum_dispatch_covers_view_helpers() {
        let backend = EnnBackend::new_in_memory(
            array![[0.0, 0.0], [1.0, 0.0]],
            array![[0.0], [1.0]],
            None,
            false,
            Array1::ones(2),
            IndexDriver::Exact,
        )
        .unwrap();
        assert_eq!(backend.len(), 2);
        assert_eq!(backend.driver(), IndexDriver::Exact);
        backend
            .search(&array![[0.1, 0.2]].view(), 1, false)
            .unwrap();
    }

    #[test]
    fn disk_bpann_dispatch() {
        use tempfile::TempDir;
        let dir = TempDir::new().expect("tempdir");
        let backend = EnnBackend::new_disk(
            dir.path().to_path_buf(),
            array![[0.0, 0.0], [1.0, 0.0]],
            array![[0.0], [1.0]],
            None,
            false,
            Array1::ones(2),
            IndexDriver::BpAnnDisk,
        )
        .unwrap();
        backend.ensure_index_sync(false, &Array1::ones(2)).unwrap();
        assert_eq!(backend.driver(), IndexDriver::BpAnnDisk);
        assert!(backend.in_memory_index().is_none());
    }

    #[test]
    fn disk_bpann_new_empty_without_work_dir_errors() {
        let err = EnnBackend::new_empty(2, 1, IndexDriver::BpAnnDisk, EnnStorage::Disk, None, None);
        assert!(err.is_err());
    }

    #[test]
    fn enn_storage_env_helpers_and_index_len() {
        let _ = EnnStorage::from_env();
        let _ = EnnStorage::work_dir_from_env();
        let backend = EnnBackend::new_in_memory(
            array![[0.0, 0.0], [1.0, 0.0]],
            array![[0.0], [1.0]],
            None,
            false,
            Array1::ones(2),
            IndexDriver::Exact,
        )
        .unwrap();
        assert_eq!(backend.index_len(), backend.len());
        backend.wait_for_flush().unwrap();
    }

    #[test]
    fn disk_read_used() {
        use tempfile::TempDir;
        let dir = TempDir::new().expect("tempdir");
        let backend = EnnBackend::new_disk(
            dir.path().to_path_buf(),
            array![[0.0, 0.0], [1.0, 0.0]],
            array![[0.0], [1.0]],
            None,
            false,
            Array1::ones(2),
            IndexDriver::BpAnnDisk,
        )
        .unwrap();
        if let EnnBackend::Disk(handle) = &backend {
            let guard = disk_read(handle.data()).unwrap();
            assert_eq!(guard.driver(), IndexDriver::BpAnnDisk);
        }
    }
}
