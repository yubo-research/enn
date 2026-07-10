//! Disk-backed ENN backend wrapping the standalone `bpann` crate.

use std::path::PathBuf;

use bpann::BpannBackend;
use ndarray::{Array1, Array2, ArrayView2};

use crate::backend::TrainRowsAtResult;
use crate::error::ENNError;
use crate::index::IndexDriver;

fn bpann_err(e: bpann::BpannError) -> ENNError {
    match e {
        bpann::BpannError::InvalidShape { expected, got } => ENNError::InvalidShape { expected, got },
        bpann::BpannError::InvalidParameter(s) => ENNError::InvalidParameter(s),
    }
}

pub struct DiskBpannEnnBackend {
    inner: BpannBackend,
    driver: IndexDriver,
    num_metrics: usize,
}

impl DiskBpannEnnBackend {
    pub fn new(
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
                "DiskBpannEnnBackend requires IndexDriver::BpAnnDisk".to_string(),
            ));
        }
        let inner = if train_x.nrows() == 0
            && train_y.nrows() == 0
            && work_dir.join("metadata.json").exists()
        {
            BpannBackend::reopen(work_dir.clone()).map_err(bpann_err)?
        } else {
            BpannBackend::new(work_dir, train_x, train_y, train_yvar, scale_x, x_scale)
                .map_err(bpann_err)?
        };
        let num_metrics = inner.num_metrics();
        Ok(Self {
            inner,
            driver,
            num_metrics,
        })
    }

    pub fn new_empty(work_dir: PathBuf, num_dim: usize, num_metrics: usize) -> Result<Self, ENNError> {
        Ok(Self {
            inner: BpannBackend::new_empty(work_dir, num_dim, num_metrics).map_err(bpann_err)?,
            driver: IndexDriver::BpAnnDisk,
            num_metrics,
        })
    }

    pub fn new_empty_with_flush_threshold(
        work_dir: PathBuf,
        num_dim: usize,
        num_metrics: usize,
        pending_flush_threshold: usize,
    ) -> Result<Self, ENNError> {
        let inner = BpannBackend::new_empty(work_dir, num_dim, num_metrics)
            .map_err(bpann_err)?
            .with_pending_flush_threshold(pending_flush_threshold)
            .with_defer_append_indexing(true);
        Ok(Self {
            inner,
            driver: IndexDriver::BpAnnDisk,
            num_metrics,
        })
    }

    pub fn driver(&self) -> IndexDriver {
        self.driver
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn num_dim(&self) -> usize {
        self.inner.num_dim()
    }

    pub fn num_metrics(&self) -> usize {
        self.num_metrics
    }

    pub fn pending_unindexed_count(&self) -> usize {
        self.inner.pending_rows()
    }

    pub fn pending_flush_threshold(&self) -> usize {
        self.inner.pending_flush_threshold()
    }

    pub fn append_syncs_at_threshold(&self) -> bool {
        !self.inner.defer_append_indexing()
    }

    pub fn defer_index_sync_for_search(&self) -> bool {
        true
    }

    pub fn is_index_stale(&self) -> bool {
        false
    }

    pub fn mark_index_stale(&mut self) {
        self.inner.mark_index_stale();
    }

    pub fn wait_for_flush(&self) -> Result<(), ENNError> {
        Ok(())
    }

    pub fn schedule_background_flush(&mut self) -> Result<(), ENNError> {
        if self.inner.defer_append_indexing()
            && self.pending_unindexed_count() >= self.pending_flush_threshold()
        {
            self.inner.ensure_index_sync().map_err(bpann_err)?;
        }
        Ok(())
    }

    pub fn persist_index_to_disk(&mut self) -> Result<(), ENNError> {
        self.inner.persist_index_to_disk().map_err(bpann_err)
    }

    pub fn append_rows(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Result<(), ENNError> {
        self.inner.append_rows(x, y, yvar).map_err(bpann_err)
    }

    pub fn ensure_index_sync(
        &mut self,
        scale_x: bool,
        x_scale: &Array1<f64>,
    ) -> Result<(), ENNError> {
        self.inner
            .ensure_index_sync_with_scale(scale_x, x_scale)
            .map_err(bpann_err)
    }

    pub fn train_rows_at(&self, indices: &[usize]) -> Result<TrainRowsAtResult, ENNError> {
        self.inner.train_rows_at(indices).map_err(bpann_err)
    }

    pub fn row_x(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        Ok(Array1::from(
            self.inner.mmap_row_slice(i).map_err(bpann_err)?.to_vec(),
        ))
    }

    pub fn row_y(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        let (_, y, _) = self.inner.train_rows_at(&[i]).map_err(bpann_err)?;
        Ok(y.row(0).to_owned())
    }

    pub fn row_yvar(&self, i: usize) -> Result<Option<Array1<f64>>, ENNError> {
        let (_, _, yvar) = self.inner.train_rows_at(&[i]).map_err(bpann_err)?;
        Ok(yvar.map(|v| v.row(0).to_owned()))
    }

    pub fn search(
        &self,
        x: &ArrayView2<f64>,
        search_k: i32,
        exclude_nearest: bool,
    ) -> Result<(Array2<f64>, Array2<i64>), ENNError> {
        self.inner
            .search(x, search_k as usize, exclude_nearest)
            .map_err(bpann_err)
    }

    pub fn index_memory_bytes(&self) -> Result<usize, ENNError> {
        Ok(self.inner.index_memory_bytes())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use tempfile::TempDir;

    #[test]
    fn new_empty_with_flush_threshold_sets_defer_sync() {
        let dir = TempDir::new().expect("tempdir");
        let backend = DiskBpannEnnBackend::new_empty_with_flush_threshold(
            dir.path().to_path_buf(),
            2,
            1,
            5,
        )
        .expect("backend");
        assert_eq!(backend.pending_flush_threshold(), 5);
        assert!(!backend.append_syncs_at_threshold());
        assert_eq!(backend.pending_unindexed_count(), 0);
    }

    #[test]
    fn new_rejects_non_bpann_driver() {
        let dir = TempDir::new().expect("tempdir");
        let result = DiskBpannEnnBackend::new(
            dir.path().to_path_buf(),
            array![[0.0, 0.0]],
            array![[0.0]],
            None,
            false,
            array![1.0, 1.0],
            IndexDriver::Exact,
        );
        assert!(matches!(result, Err(ENNError::InvalidParameter(_))));
    }

    #[test]
    fn append_rows_shape_error_maps_invalid_shape() {
        let dir = TempDir::new().expect("tempdir");
        let mut backend =
            DiskBpannEnnBackend::new_empty(dir.path().to_path_buf(), 2, 1).expect("backend");
        let err = backend
            .append_rows(
                &array![[0.0, 0.0, 0.0]].view(),
                &array![[0.0]].view(),
                None,
            )
            .unwrap_err();
        assert!(matches!(err, ENNError::InvalidShape { .. }));
    }

    #[test]
    fn reopen_from_existing_work_dir() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().to_path_buf();
        let _fresh = DiskBpannEnnBackend::new(
            path.clone(),
            array![[0.0, 0.0], [1.0, 0.0]],
            array![[0.0], [1.0]],
            None,
            false,
            array![1.0, 1.0],
            IndexDriver::BpAnnDisk,
        )
        .expect("fresh");
        let reopened = DiskBpannEnnBackend::new(
            path,
            Array2::zeros((0, 2)),
            Array2::zeros((0, 1)),
            None,
            false,
            array![1.0, 1.0],
            IndexDriver::BpAnnDisk,
        )
        .expect("reopen");
        assert_eq!(reopened.len(), 2);
    }

    #[test]
    fn reopen_uses_persisted_num_metrics_not_placeholder_train_y() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().to_path_buf();
        let _fresh = DiskBpannEnnBackend::new(
            path.clone(),
            array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            array![[0.0, 1.0], [1.0, 2.0], [1.0, 3.0]],
            None,
            false,
            array![1.0, 1.0],
            IndexDriver::BpAnnDisk,
        )
        .expect("fresh");
        let reopened = DiskBpannEnnBackend::new(
            path,
            Array2::zeros((0, 2)),
            Array2::zeros((0, 1)),
            None,
            false,
            array![1.0, 1.0],
            IndexDriver::BpAnnDisk,
        )
        .expect("reopen");
        assert_eq!(reopened.num_metrics(), 2);
    }

    #[test]
    fn pending_unindexed_count_after_append_without_sync() {
        let dir = TempDir::new().expect("tempdir");
        let mut backend = DiskBpannEnnBackend::new_empty_with_flush_threshold(
            dir.path().to_path_buf(),
            2,
            1,
            100,
        )
        .expect("backend");
        backend
            .append_rows(
                &array![[0.0, 0.0]].view(),
                &array![[1.0]].view(),
                None,
            )
            .expect("append");
        assert!(backend.pending_unindexed_count() > 0);
    }
}
