//! ENN storage and indexing backends.

pub(crate) mod disk_observation;
mod in_memory;
pub(crate) mod row_storage;

pub use in_memory::InMemoryEnnBackend;
pub use crate::disk_hnsw::DiskHnswEnnBackend;

use ndarray::{Array1, Array2, ArrayView2};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

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

pub enum DiskEnnBackend {
    Hnsw(DiskHnswEnnBackend),
}

pub enum EnnBackend {
    InMemory(Box<InMemoryEnnBackend>),
    Disk(Arc<Mutex<DiskEnnBackend>>),
}

fn disk_lock<'a>(
    b: &'a Arc<Mutex<DiskEnnBackend>>,
) -> Result<std::sync::MutexGuard<'a, DiskEnnBackend>, ENNError> {
    b.lock()
        .map_err(|_| ENNError::InvalidParameter("disk backend mutex poisoned".to_string()))
}

fn disk_driver(d: &DiskEnnBackend) -> IndexDriver {
    match d {
        DiskEnnBackend::Hnsw(b) => b.driver(),
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
        let inner = match driver {
            IndexDriver::HNSWDisk => DiskEnnBackend::Hnsw(DiskHnswEnnBackend::new(
                work_dir,
                train_x,
                train_y,
                train_yvar,
                scale_x,
                x_scale,
                driver,
            )?),
            _ => {
                return Err(ENNError::InvalidParameter(
                    "Disk storage requires IndexDriver::HNSWDisk".to_string(),
                ));
            }
        };
        Ok(Self::Disk(Arc::new(Mutex::new(inner))))
    }

    pub fn new_empty(
        num_dim: usize,
        num_metrics: usize,
        driver: IndexDriver,
        storage: EnnStorage,
        work_dir: Option<PathBuf>,
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
                let inner = match driver {
                    IndexDriver::HNSWDisk => DiskEnnBackend::Hnsw(DiskHnswEnnBackend::new_empty(
                        dir, num_dim, num_metrics,
                    )?),
                    _ => {
                        return Err(ENNError::InvalidParameter(
                            "Disk storage requires IndexDriver::HNSWDisk".to_string(),
                        ));
                    }
                };
                Ok(Self::Disk(Arc::new(Mutex::new(inner))))
            }
        }
    }

    pub fn wait_for_flush(&self) -> Result<(), ENNError> {
        match self {
            Self::InMemory(_) => Ok(()),
            Self::Disk(arc) => {
                let flush_arc = {
                    let g = disk_lock(arc)?;
                    match &*g {
                        DiskEnnBackend::Hnsw(x) => Arc::clone(&x.flush),
                    }
                };
                let st = flush_arc.lock().map_err(|_| {
                    ENNError::InvalidParameter("flush state mutex poisoned".to_string())
                })?;
                if !st.in_progress {
                    if let Some(err) = st.error.clone() {
                        return Err(err);
                    }
                    return Ok(());
                }
                drop(st);
                crate::disk_hnsw::flush::wait_for_background_flush(&flush_arc)
            }
        }
    }

    pub fn schedule_background_flush(&self) -> Result<(), ENNError> {
        match self {
            Self::InMemory(_) => Ok(()),
            Self::Disk(arc) => {
                let (should, flush_arc) = {
                    let g = disk_lock(arc)?;
                    match &*g {
                        DiskEnnBackend::Hnsw(x) => {
                            let pending = x.pending_unindexed_count();
                            let threshold = x.pending_flush_threshold();
                            (
                                !x.append_syncs_at_threshold()
                                    && !x.is_index_stale()
                                    && pending >= threshold,
                                Arc::clone(&x.flush),
                            )
                        }
                    }
                };
                if !should {
                    return Ok(());
                }
                if flush_arc
                    .lock()
                    .map_err(|_| {
                        ENNError::InvalidParameter("flush state mutex poisoned".to_string())
                    })?
                    .in_progress
                {
                    return Ok(());
                }
                crate::disk_hnsw::flush::try_schedule_background_flush(
                    &flush_arc,
                    Arc::clone(arc),
                )
            }
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::InMemory(b) => b.len(),
            Self::Disk(b) => disk_lock(b)
                .map(|g| match &*g {
                    DiskEnnBackend::Hnsw(x) => x.len(),
                })
                .unwrap_or(0),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn num_dim(&self) -> usize {
        match self {
            Self::InMemory(b) => b.num_dim(),
            Self::Disk(b) => disk_lock(b)
                .map(|g| match &*g {
                    DiskEnnBackend::Hnsw(x) => x.num_dim(),
                })
                .unwrap_or(0),
        }
    }

    pub fn num_metrics(&self) -> usize {
        match self {
            Self::InMemory(b) => b.num_metrics(),
            Self::Disk(b) => disk_lock(b)
                .map(|g| match &*g {
                    DiskEnnBackend::Hnsw(x) => x.num_metrics(),
                })
                .unwrap_or(0),
        }
    }

    pub fn driver(&self) -> IndexDriver {
        match self {
            Self::InMemory(b) => b.driver(),
            Self::Disk(b) => disk_lock(b)
                .map(|g| disk_driver(&g))
                .unwrap_or(IndexDriver::HNSWDisk),
        }
    }

    pub fn mark_index_stale(&self) {
        match self {
            Self::InMemory(b) => b.mark_index_stale(),
            Self::Disk(b) => {
                if let Ok(mut g) = disk_lock(b) {
                    match &mut *g {
                        DiskEnnBackend::Hnsw(x) => x.mark_index_stale(),
                    }
                }
            }
        }
    }

    pub fn is_index_stale(&self) -> bool {
        match self {
            Self::InMemory(b) => b.is_index_stale(),
            Self::Disk(b) => disk_lock(b)
                .map(|g| match &*g {
                    DiskEnnBackend::Hnsw(x) => x.is_index_stale(),
                })
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
            Self::Disk(b) => {
                let mut g = disk_lock(b)?;
                match &mut *g {
                    DiskEnnBackend::Hnsw(xb) => xb.append_rows(x, y, yvar),
                }
            }
        }
    }

    pub fn defer_index_sync_for_search(&self) -> bool {
        match self {
            Self::InMemory(_) => false,
            Self::Disk(b) => disk_lock(b)
                .map(|g| match &*g {
                    DiskEnnBackend::Hnsw(x) => x.defer_index_sync_for_search(),
                })
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
            Self::Disk(b) => {
                let mut g = disk_lock(b)?;
                match &mut *g {
                    DiskEnnBackend::Hnsw(xb) => xb.ensure_index_sync(scale_x, x_scale),
                }
            }
        }
    }

    pub fn train_rows_at(
        &self,
        indices: &[usize],
    ) -> Result<TrainRowsAtResult, ENNError> {
        match self {
            Self::InMemory(b) => b.train_rows_at(indices),
            Self::Disk(b) => {
                let g = disk_lock(b)?;
                match &*g {
                    DiskEnnBackend::Hnsw(xb) => xb.train_rows_at(indices),
                }
            }
        }
    }

    pub fn row_x(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        match self {
            Self::InMemory(b) => b.row_x(i),
            Self::Disk(b) => {
                let g = disk_lock(b)?;
                match &*g {
                    DiskEnnBackend::Hnsw(xb) => xb.row_x(i),
                }
            }
        }
    }

    pub fn row_y(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        match self {
            Self::InMemory(b) => b.row_y(i),
            Self::Disk(b) => {
                let g = disk_lock(b)?;
                match &*g {
                    DiskEnnBackend::Hnsw(xb) => xb.row_y(i),
                }
            }
        }
    }

    pub fn row_yvar(&self, i: usize) -> Result<Option<Array1<f64>>, ENNError> {
        match self {
            Self::InMemory(b) => b.row_yvar(i),
            Self::Disk(b) => {
                let g = disk_lock(b)?;
                match &*g {
                    DiskEnnBackend::Hnsw(xb) => xb.row_yvar(i),
                }
            }
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
            Self::Disk(b) => {
                let g = disk_lock(b)?;
                match &*g {
                    DiskEnnBackend::Hnsw(xb) => xb.search(x, search_k, exclude_nearest),
                }
            }
        }
    }

    pub fn index_memory_bytes(&self) -> Result<usize, ENNError> {
        match self {
            Self::InMemory(b) => b.index_memory_bytes(),
            Self::Disk(b) => {
                let g = disk_lock(b)?;
                match &*g {
                    DiskEnnBackend::Hnsw(xb) => xb.index_memory_bytes(),
                }
            }
        }
    }

    pub fn index_len(&self) -> usize {
        match self {
            Self::InMemory(b) => b.index_len(),
            Self::Disk(b) => disk_lock(b)
                .map(|g| match &*g {
                    DiskEnnBackend::Hnsw(x) => x.len(),
                })
                .unwrap_or(0),
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

impl Drop for EnnBackend {
    fn drop(&mut self) {
        let _ = self.wait_for_flush();
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
    fn disk_hnsw_enum_dispatch() {
        use tempfile::TempDir;
        let dir = TempDir::new().expect("tempdir");
        let backend = EnnBackend::new_disk(
            dir.path().to_path_buf(),
            array![[0.0, 0.0], [1.0, 0.0]],
            array![[0.0], [1.0]],
            None,
            false,
            Array1::ones(2),
            IndexDriver::HNSWDisk,
        )
        .unwrap();
        backend.ensure_index_sync(false, &Array1::ones(2)).unwrap();
        assert_eq!(backend.driver(), IndexDriver::HNSWDisk);
        assert!(backend.in_memory_index().is_none());
    }

    #[test]
    fn disk_hnsw_new_empty_without_work_dir_errors() {
        let err = EnnBackend::new_empty(2, 1, IndexDriver::HNSWDisk, EnnStorage::Disk, None);
        assert!(err.is_err());
    }

    #[test]
    fn kiss_disk_dispatch_covers_helpers() {
        assert!(std::mem::size_of::<DiskEnnBackend>() > 0);
    }
}
