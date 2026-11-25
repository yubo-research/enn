pub mod disk_observation;
mod flush_controller;
mod mmap_column_store;

pub use crate::disk_bpann::DiskBpannEnnBackend;
pub(crate) use flush_controller::DiskBackendHandle;
pub use mmap_column_store::MmapColumnStore;

use ndarray::{Array1, Array2, ArrayView2};
use std::path::PathBuf;
use std::sync::{Arc, RwLock};

use crate::error::ENNError;
use crate::o1::O1Store;

pub type TrainRowsAtResult = (Array2<f64>, Array2<f64>, Option<Array2<f64>>);

pub(crate) enum BackendKind {
    OLogN(DiskBackendHandle),
    O1(Box<RwLock<O1Store>>),
}

pub struct EnnBackend {
    pub(crate) kind: BackendKind,
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

fn o1_read(s: &RwLock<O1Store>) -> Result<std::sync::RwLockReadGuard<'_, O1Store>, ENNError> {
    s.read()
        .map_err(|_| ENNError::InvalidParameter("O1 store lock poisoned".to_string()))
}

fn o1_write(s: &RwLock<O1Store>) -> Result<std::sync::RwLockWriteGuard<'_, O1Store>, ENNError> {
    s.write()
        .map_err(|_| ENNError::InvalidParameter("O1 store lock poisoned".to_string()))
}

macro_rules! read_either {
    ($self:expr, $g:ident => $e:expr) => {
        match &$self.kind {
            BackendKind::OLogN(h) => {
                let $g = disk_read(h.data())?;
                $e
            }
            BackendKind::O1(s) => {
                let $g = o1_read(s)?;
                $e
            }
        }
    };
}

impl EnnBackend {
    pub fn new_disk(
        work_dir: PathBuf,
        train_x: Array2<f64>,
        train_y: Array2<f64>,
        train_yvar: Option<Array2<f64>>,
        x_scale: Array1<f64>,
    ) -> Result<Self, ENNError> {
        let inner =
            DiskBpannEnnBackend::new(work_dir, train_x, train_y, train_yvar, false, x_scale)?;
        Ok(Self {
            kind: BackendKind::OLogN(DiskBackendHandle::new(inner)),
        })
    }

    pub fn new_o1(
        mut store: O1Store,
        train_x: Array2<f64>,
        train_y: Array2<f64>,
        train_yvar: Option<Array2<f64>>,
    ) -> Result<Self, ENNError> {
        let yvar = train_yvar.as_ref().map(|v| v.view());
        store.append_rows(&train_x.view(), &train_y.view(), yvar.as_ref())?;
        Ok(Self::new_empty_o1(store))
    }

    pub fn new_empty_disk(
        work_dir: PathBuf,
        num_dim: usize,
        num_metrics: usize,
        pending_flush_threshold: Option<usize>,
    ) -> Result<Self, ENNError> {
        let inner = match pending_flush_threshold {
            Some(threshold) => DiskBpannEnnBackend::new_empty_with_flush_threshold(
                work_dir,
                num_dim,
                num_metrics,
                threshold,
            )?,
            None => DiskBpannEnnBackend::new_empty(work_dir, num_dim, num_metrics)?,
        };
        Ok(Self {
            kind: BackendKind::OLogN(DiskBackendHandle::new(inner)),
        })
    }

    pub fn new_empty_o1(store: O1Store) -> Self {
        Self {
            kind: BackendKind::O1(Box::new(RwLock::new(store))),
        }
    }

    pub(crate) fn ologn_handle(&self) -> Option<&DiskBackendHandle> {
        match &self.kind {
            BackendKind::OLogN(h) => Some(h),
            BackendKind::O1(_) => None,
        }
    }

    pub fn wait_for_flush(&self) -> Result<(), ENNError> {
        self.ologn_handle().map_or(Ok(()), |h| h.wait_for_flush())
    }

    pub fn schedule_background_flush(&self) -> Result<(), ENNError> {
        self.ologn_handle()
            .map_or(Ok(()), |h| h.schedule_background_flush())
    }

    fn len_checked(&self) -> Result<usize, ENNError> {
        read_either!(self, g => Ok(g.len()))
    }

    pub fn len(&self) -> usize {
        self.len_checked().unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn shape_checked(&self) -> Result<(usize, usize), ENNError> {
        read_either!(self, g => Ok((g.num_dim(), g.num_metrics())))
    }

    pub fn num_dim(&self) -> usize {
        self.shape_checked().map(|s| s.0).unwrap_or(0)
    }

    pub fn num_metrics(&self) -> usize {
        self.shape_checked().map(|s| s.1).unwrap_or(0)
    }

    pub fn mark_index_stale(&self) {
        if let Some(h) = self.ologn_handle() {
            if let Ok(mut g) = disk_write(h.data()) {
                g.mark_index_stale();
            }
        }
    }

    pub fn append_rows(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Result<(), ENNError> {
        match &self.kind {
            BackendKind::OLogN(h) => disk_write(h.data())?.append_rows(x, y, yvar),
            BackendKind::O1(s) => o1_write(s)?.append_rows(x, y, yvar),
        }
    }

    pub fn ensure_index_sync(&self, scale_x: bool, x_scale: &Array1<f64>) -> Result<(), ENNError> {
        match self.ologn_handle() {
            Some(h) => {
                h.wait_for_flush()?;
                disk_write(h.data())?.ensure_index_sync(scale_x, x_scale)
            }
            None => Ok(()),
        }
    }

    pub fn metric_scale(&self) -> Result<Option<Array1<f64>>, ENNError> {
        read_either!(self, g => Ok(g.metric_scale()))
    }

    pub fn set_metric_scale(&self, x_scale: &Array1<f64>, rebuild: bool) -> Result<(), ENNError> {
        match &self.kind {
            BackendKind::OLogN(h) => {
                h.wait_for_flush()?;
                disk_write(h.data())?.set_metric_scale(x_scale, rebuild)
            }
            BackendKind::O1(s) => o1_write(s)?.set_metric_scale(x_scale, rebuild),
        }
    }

    pub fn train_rows_at(&self, indices: &[usize]) -> Result<TrainRowsAtResult, ENNError> {
        read_either!(self, g => g.train_rows_at(indices))
    }

    pub fn scattered_rows(&self, indices: &[usize]) -> Result<(Array2<f64>, Array2<f64>), ENNError> {
        match &self.kind {
            BackendKind::O1(s) => crate::o1::store_rows::scattered_rows(&*o1_read(s)?, indices),
            BackendKind::OLogN(_) => self.train_rows_at(indices).map(|(x, y, _)| (x, y)),
        }
    }

    pub fn y_rows(&self, start: usize, end: usize) -> Result<Array2<f64>, ENNError> {
        match &self.kind {
            BackendKind::OLogN(h) => disk_read(h.data())?.y_rows(start, end),
            BackendKind::O1(s) => crate::o1::store_rows::y_rows(&*o1_read(s)?, start, end),
        }
    }

    pub fn row_x(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        read_either!(self, g => g.row_x(i))
    }

    pub fn row_y(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        read_either!(self, g => g.row_y(i))
    }

    pub fn row_yvar(&self, i: usize) -> Result<Option<Array1<f64>>, ENNError> {
        read_either!(self, g => g.row_yvar(i))
    }

    pub fn search(
        &self,
        x: &ArrayView2<f64>,
        search_k: i32,
        exclude_nearest: bool,
    ) -> Result<(Array2<f64>, Array2<i64>), ENNError> {
        if search_k <= 0 {
            return Err(ENNError::InvalidParameter(format!(
                "search_k must be > 0, got {search_k}"
            )));
        }
        let num_dim = self.num_dim();
        if x.ncols() != num_dim {
            return Err(ENNError::InvalidShape {
                expected: vec![x.nrows(), num_dim],
                got: x.shape().to_vec(),
            });
        }
        match &self.kind {
            BackendKind::OLogN(h) => disk_read(h.data())?.search(x, search_k, exclude_nearest),
            BackendKind::O1(s) => o1_read(s)?.search(x, search_k as usize, exclude_nearest),
        }
    }

    pub fn with_o1<T>(&self, f: impl FnOnce(&mut O1Store) -> Result<T, ENNError>) -> Result<Option<T>, ENNError> {
        match &self.kind {
            BackendKind::OLogN(_) => Ok(None),
            BackendKind::O1(s) => f(&mut *o1_write(s)?).map(Some),
        }
    }

    pub fn index_memory_bytes(&self) -> Result<usize, ENNError> {
        match &self.kind {
            BackendKind::OLogN(h) => disk_read(h.data())?.index_memory_bytes(),
            BackendKind::O1(s) => Ok(o1_read(s)?.index_memory_bytes()),
        }
    }

    pub fn index_len(&self) -> usize {
        self.len()
    }

    pub(crate) fn persist_index_to_disk(&self) -> Result<(), ENNError> {
        self.ologn_handle()
            .map_or(Ok(()), |h| h.persist_index_to_disk())
    }

    pub(crate) fn release_observation_pages(&self) -> Result<(), ENNError> {
        match self.ologn_handle() {
            Some(h) => disk_write(h.data())?.release_observation_pages(),
            None => Ok(()),
        }
    }
}

impl Drop for EnnBackend {
    fn drop(&mut self) {
        if let Err(e) = self.persist_index_to_disk() {
            eprintln!("ennbo: persist_index_to_disk on drop failed: {e}");
        }
    }
}

#[cfg(test)]
mod backend_tests {
    use super::*;
    use ndarray::array;

    fn two_row_backend(dir: &tempfile::TempDir) -> EnnBackend {
        EnnBackend::new_disk(
            dir.path().to_path_buf(),
            array![[0.0, 0.0], [1.0, 0.0]],
            array![[0.0], [1.0]],
            None,
            Array1::ones(2),
        )
        .unwrap()
    }

    #[test]
    fn disk_backend_reports_rows_and_searches() {
        let dir = tempfile::TempDir::new().unwrap();
        let backend = two_row_backend(&dir);
        backend.ensure_index_sync(false, &Array1::ones(2)).unwrap();
        assert_eq!(backend.len(), 2);
        assert_eq!(backend.index_len(), backend.len());
        let (_, idx) = backend
            .search(&array![[0.1, 0.2]].view(), 1, false)
            .unwrap();
        assert_eq!(idx[[0, 0]], 0);
        assert_eq!(backend.y_rows(0, 2).unwrap(), array![[0.0], [1.0]]);
        assert_eq!(backend.scattered_rows(&[1]).unwrap().1, array![[1.0]]);
        backend.wait_for_flush().unwrap();
    }

    #[test]
    fn o1_backend_dispatches_rows_search_and_metric() {
        let dir = tempfile::TempDir::new().unwrap();
        let backend = EnnBackend::new_o1(
            O1Store::open_or_create(dir.path().to_path_buf(), 2, 1).unwrap(),
            array![[0.0, 0.0], [1.0, 0.0]],
            array![[0.0], [1.0]],
            None,
        )
        .unwrap();
        assert!(backend.ologn_handle().is_none());
        backend.ensure_index_sync(false, &Array1::ones(2)).unwrap();
        backend.wait_for_flush().unwrap();
        backend.schedule_background_flush().unwrap();
        backend.mark_index_stale();
        backend.release_observation_pages().unwrap();
        assert_eq!((backend.len(), backend.num_dim(), backend.num_metrics()), (2, 2, 1));
        let (_, idx) = backend.search(&array![[0.9, 0.0]].view(), 1, false).unwrap();
        assert_eq!(idx[[0, 0]], 1);
        assert!(backend.search(&array![[0.9]].view(), 1, false).is_err());
        backend.set_metric_scale(&array![1.0, 2.0], true).unwrap();
        assert_eq!(backend.metric_scale().unwrap(), Some(array![1.0, 2.0]));
        assert_eq!(backend.row_y(1).unwrap()[0], 1.0);
        assert_eq!(backend.y_rows(1, 2).unwrap(), array![[1.0]]);
        assert_eq!(backend.scattered_rows(&[1, 0]).unwrap(), (array![[1.0, 0.0], [0.0, 0.0]], array![[1.0], [0.0]]));
        assert!(backend.scattered_rows(&[2]).is_err());
        assert!(backend.y_rows(1, 3).is_err());
        assert!(backend.row_yvar(1).unwrap().is_none());
        assert!(backend.index_memory_bytes().is_ok());
        assert_eq!(backend.with_o1(|s| s.occupancy()).unwrap().unwrap().num_rows, 2);
        let cands = backend.with_o1(|s| s.candidates(&array![[0.9, 0.0]].view(), 1)).unwrap().unwrap();
        assert_eq!(cands, vec![vec![0, 1]]);
        let empty = EnnBackend::new_empty_o1(O1Store::open_or_create(dir.path().join("e"), 2, 1).unwrap());
        assert!(empty.is_empty());
    }

    #[test]
    fn disk_read_used() {
        let dir = tempfile::TempDir::new().unwrap();
        let backend = two_row_backend(&dir);
        assert_eq!(disk_read(backend.ologn_handle().unwrap().data()).unwrap().len(), 2);
        assert!(backend.with_o1(|s| s.occupancy()).unwrap().is_none());
    }
}
