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

pub type TrainRowsAtResult = (Array2<f64>, Array2<f64>, Option<Array2<f64>>);

pub struct EnnBackend {
    pub(crate) handle: DiskBackendHandle,
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
            handle: DiskBackendHandle::new(inner),
        })
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
            handle: DiskBackendHandle::new(inner),
        })
    }

    pub fn wait_for_flush(&self) -> Result<(), ENNError> {
        self.handle.wait_for_flush()
    }

    pub fn schedule_background_flush(&self) -> Result<(), ENNError> {
        self.handle.schedule_background_flush()
    }

    pub fn len(&self) -> usize {
        disk_read(self.handle.data()).map(|g| g.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn num_dim(&self) -> usize {
        disk_read(self.handle.data())
            .map(|g| g.num_dim())
            .unwrap_or(0)
    }

    pub fn num_metrics(&self) -> usize {
        disk_read(self.handle.data())
            .map(|g| g.num_metrics())
            .unwrap_or(0)
    }

    pub fn mark_index_stale(&self) {
        if let Ok(mut g) = disk_write(self.handle.data()) {
            g.mark_index_stale();
        }
    }

    pub fn append_rows(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Result<(), ENNError> {
        disk_write(self.handle.data())?.append_rows(x, y, yvar)
    }

    pub fn ensure_index_sync(&self, scale_x: bool, x_scale: &Array1<f64>) -> Result<(), ENNError> {
        self.wait_for_flush()?;
        disk_write(self.handle.data())?.ensure_index_sync(scale_x, x_scale)
    }

    pub fn metric_scale(&self) -> Result<Option<Array1<f64>>, ENNError> {
        Ok(disk_read(self.handle.data())?.metric_scale())
    }

    pub fn set_metric_scale(&self, x_scale: &Array1<f64>, rebuild: bool) -> Result<(), ENNError> {
        self.wait_for_flush()?;
        disk_write(self.handle.data())?.set_metric_scale(x_scale, rebuild)
    }

    pub fn train_rows_at(&self, indices: &[usize]) -> Result<TrainRowsAtResult, ENNError> {
        disk_read(self.handle.data())?.train_rows_at(indices)
    }

    pub fn row_x(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        disk_read(self.handle.data())?.row_x(i)
    }

    pub fn row_y(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        disk_read(self.handle.data())?.row_y(i)
    }

    pub fn row_yvar(&self, i: usize) -> Result<Option<Array1<f64>>, ENNError> {
        disk_read(self.handle.data())?.row_yvar(i)
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
        let backend = disk_read(self.handle.data())?;
        if x.ncols() != backend.num_dim() {
            return Err(ENNError::InvalidShape {
                expected: vec![x.nrows(), backend.num_dim()],
                got: x.shape().to_vec(),
            });
        }
        backend.search(x, search_k, exclude_nearest)
    }

    pub fn index_memory_bytes(&self) -> Result<usize, ENNError> {
        disk_read(self.handle.data())?.index_memory_bytes()
    }

    pub fn index_len(&self) -> usize {
        disk_read(self.handle.data()).map(|g| g.len()).unwrap_or(0)
    }

    pub(crate) fn persist_index_to_disk(&self) -> Result<(), ENNError> {
        self.handle.persist_index_to_disk()
    }

    pub(crate) fn release_observation_pages(&self) -> Result<(), ENNError> {
        disk_write(self.handle.data())?.release_observation_pages()
    }
}

impl Drop for EnnBackend {
    fn drop(&mut self) {
        if let Err(e) = self.handle.persist_index_to_disk() {
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
        backend.wait_for_flush().unwrap();
    }

    #[test]
    fn disk_read_used() {
        let dir = tempfile::TempDir::new().unwrap();
        let backend = two_row_backend(&dir);
        assert_eq!(disk_read(backend.handle.data()).unwrap().len(), 2);
    }
}
