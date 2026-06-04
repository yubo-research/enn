//! ENN storage and indexing backends.

mod in_memory;
pub(crate) mod row_storage;

#[cfg(feature = "hannoy")]
mod disk_hannoy;

pub use in_memory::InMemoryEnnBackend;
#[cfg(feature = "hannoy")]
pub use disk_hannoy::DiskHannoyEnnBackend;

use ndarray::{Array1, Array2, ArrayView2};
use std::path::PathBuf;
use std::sync::Mutex;

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
    InMemory(InMemoryEnnBackend),
    #[cfg(feature = "hannoy")]
    Disk(Mutex<DiskHannoyEnnBackend>),
}

#[cfg(feature = "hannoy")]
fn disk_lock<'a>(
    b: &'a Mutex<DiskHannoyEnnBackend>,
) -> Result<std::sync::MutexGuard<'a, DiskHannoyEnnBackend>, ENNError> {
    b.lock()
        .map_err(|_| ENNError::InvalidParameter("disk backend mutex poisoned".to_string()))
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
        Ok(Self::InMemory(InMemoryEnnBackend::new(
            train_x, train_y, train_yvar, scale_x, x_scale, driver,
        )?))
    }

    #[cfg(feature = "hannoy")]
    pub fn new_disk(
        work_dir: PathBuf,
        train_x: Array2<f64>,
        train_y: Array2<f64>,
        train_yvar: Option<Array2<f64>>,
        scale_x: bool,
        x_scale: Array1<f64>,
    ) -> Result<Self, ENNError> {
        Ok(Self::Disk(Mutex::new(DiskHannoyEnnBackend::new(
            work_dir,
            train_x,
            train_y,
            train_yvar,
            scale_x,
            x_scale,
            IndexDriver::HNSWHannoy,
        )?)))
    }

    pub fn new_empty(
        num_dim: usize,
        num_metrics: usize,
        driver: IndexDriver,
        storage: EnnStorage,
        work_dir: Option<PathBuf>,
    ) -> Result<Self, ENNError> {
        match storage {
            EnnStorage::InMemory => Ok(Self::InMemory(InMemoryEnnBackend::new_empty(
                num_dim, num_metrics, driver,
            )?)),
            #[cfg(feature = "hannoy")]
            EnnStorage::Disk => {
                if driver != IndexDriver::HNSWHannoy {
                    return Err(ENNError::InvalidParameter(
                        "Disk storage requires IndexDriver::HNSWHannoy".to_string(),
                    ));
                }
                let dir = work_dir.or_else(EnnStorage::work_dir_from_env).ok_or_else(|| {
                    ENNError::InvalidParameter(
                        "Disk storage requires work_dir or ENN_WORK_DIR".to_string(),
                    )
                })?;
                Ok(Self::Disk(Mutex::new(DiskHannoyEnnBackend::new_empty(
                    dir, num_dim, num_metrics,
                )?)))
            }
            #[cfg(not(feature = "hannoy"))]
            EnnStorage::Disk => Err(ENNError::InvalidParameter(
                "Disk storage requires the `hannoy` feature".to_string(),
            )),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::InMemory(b) => b.len(),
            #[cfg(feature = "hannoy")]
            Self::Disk(b) => disk_lock(b).map(|g| g.len()).unwrap_or(0),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn num_dim(&self) -> usize {
        match self {
            Self::InMemory(b) => b.num_dim(),
            #[cfg(feature = "hannoy")]
            Self::Disk(b) => disk_lock(b).map(|g| g.num_dim()).unwrap_or(0),
        }
    }

    pub fn num_metrics(&self) -> usize {
        match self {
            Self::InMemory(b) => b.num_metrics(),
            #[cfg(feature = "hannoy")]
            Self::Disk(b) => disk_lock(b).map(|g| g.num_metrics()).unwrap_or(0),
        }
    }

    pub fn driver(&self) -> IndexDriver {
        match self {
            Self::InMemory(b) => b.driver(),
            #[cfg(feature = "hannoy")]
            Self::Disk(b) => disk_lock(b)
                .map(|g| g.driver())
                .unwrap_or(IndexDriver::HNSWHannoy),
        }
    }

    pub fn mark_index_stale(&self) {
        match self {
            Self::InMemory(b) => b.mark_index_stale(),
            #[cfg(feature = "hannoy")]
            Self::Disk(b) => {
                if let Ok(g) = disk_lock(b) {
                    g.mark_index_stale();
                }
            }
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
            #[cfg(feature = "hannoy")]
            Self::Disk(b) => disk_lock(b)?.append_rows(x, y, yvar),
        }
    }

    pub fn ensure_index_sync(
        &self,
        scale_x: bool,
        x_scale: &Array1<f64>,
    ) -> Result<(), ENNError> {
        match self {
            Self::InMemory(b) => b.ensure_index_sync(scale_x, x_scale),
            #[cfg(feature = "hannoy")]
            Self::Disk(b) => disk_lock(b)?.ensure_index_sync(scale_x, x_scale),
        }
    }

    pub fn train_rows_at(
        &self,
        indices: &[usize],
    ) -> Result<TrainRowsAtResult, ENNError> {
        match self {
            Self::InMemory(b) => b.train_rows_at(indices),
            #[cfg(feature = "hannoy")]
            Self::Disk(b) => disk_lock(b)?.train_rows_at(indices),
        }
    }

    pub fn row_x(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        match self {
            Self::InMemory(b) => b.row_x(i),
            #[cfg(feature = "hannoy")]
            Self::Disk(b) => disk_lock(b)?.row_x(i),
        }
    }

    pub fn row_y(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        match self {
            Self::InMemory(b) => b.row_y(i),
            #[cfg(feature = "hannoy")]
            Self::Disk(b) => disk_lock(b)?.row_y(i),
        }
    }

    pub fn row_yvar(&self, i: usize) -> Result<Option<Array1<f64>>, ENNError> {
        match self {
            Self::InMemory(b) => b.row_yvar(i),
            #[cfg(feature = "hannoy")]
            Self::Disk(b) => disk_lock(b)?.row_yvar(i),
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
            #[cfg(feature = "hannoy")]
            Self::Disk(b) => disk_lock(b)?.search(x, search_k, exclude_nearest),
        }
    }

    pub fn index_memory_bytes(&self) -> Result<usize, ENNError> {
        match self {
            Self::InMemory(b) => b.index_memory_bytes(),
            #[cfg(feature = "hannoy")]
            Self::Disk(b) => disk_lock(b)?.index_memory_bytes(),
        }
    }

    pub fn index_len(&self) -> usize {
        match self {
            Self::InMemory(b) => b.index_len(),
            #[cfg(feature = "hannoy")]
            Self::Disk(b) => disk_lock(b).map(|g| g.len()).unwrap_or(0),
        }
    }

    pub fn in_memory_index(&self) -> Option<&ENNIndex> {
        match self {
            Self::InMemory(b) => Some(b.index()),
            #[cfg(feature = "hannoy")]
            Self::Disk(_) => None,
        }
    }

    pub fn in_memory_train_x_view(&self) -> Option<ndarray::ArrayView2<'_, f64>> {
        match self {
            Self::InMemory(b) => Some(b.train_x_view()),
            #[cfg(feature = "hannoy")]
            Self::Disk(_) => None,
        }
    }

    pub fn in_memory_train_y_view(&self) -> Option<ndarray::ArrayView2<'_, f64>> {
        match self {
            Self::InMemory(b) => Some(b.train_y_view()),
            #[cfg(feature = "hannoy")]
            Self::Disk(_) => None,
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
        assert_eq!(backend.num_dim(), 2);
        assert_eq!(backend.num_metrics(), 1);
        assert_eq!(backend.driver(), IndexDriver::Exact);
        assert!(backend.in_memory_train_x_view().is_some());
        assert!(backend.in_memory_train_y_view().is_some());
        assert_eq!(backend.index_len(), 2);
        assert!(backend.in_memory_index().is_some());
        backend
            .search(&array![[0.1, 0.2]].view(), 1, false)
            .unwrap();
        assert!(backend.index_memory_bytes().unwrap() > 0);
        let (x, y, _) = backend.train_rows_at(&[0]).unwrap();
        assert_eq!(x.nrows(), 1);
        assert_eq!(y.nrows(), 1);
        assert_eq!(backend.row_x(0).unwrap()[0], 0.0);
        assert_eq!(backend.row_y(0).unwrap()[0], 0.0);
        assert!(backend.row_yvar(0).unwrap().is_none());
    }

    #[cfg(feature = "hannoy")]
    #[test]
    fn disk_enum_dispatch_exercises_mod_rs() {
        use tempfile::TempDir;
        let dir = TempDir::new().expect("tempdir");
        let mut backend = EnnBackend::new_disk(
            dir.path().to_path_buf(),
            array![[0.0, 0.0], [1.0, 0.0]],
            array![[0.0], [1.0]],
            None,
            false,
            Array1::ones(2),
        )
        .unwrap();
        backend.mark_index_stale();
        backend
            .append_rows(
                &array![[2.0, 2.0]].view(),
                &array![[2.0]].view(),
                None,
            )
            .unwrap();
        backend.ensure_index_sync(false, &Array1::ones(2)).unwrap();
        assert!(backend.in_memory_train_x_view().is_none());
        assert!(backend.in_memory_index().is_none());
        assert_eq!(backend.driver(), IndexDriver::HNSWHannoy);
    }
}
