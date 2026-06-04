use faiss::error::Error as FaissError;
use faiss::index::IndexImpl;
use faiss::{index_factory, Index, MetricType};
use memmap2::MmapMut;
use ndarray::{Array2, ArrayView2, Axis};
use std::fs::{File, OpenOptions};
use std::path::PathBuf;

use super::{arr2_rows_to_f32, pad_neighbor_cols_to_search_k, unpack_batch_search};
use crate::error::ENNError;
use crate::index::{IndexDriver, IndexError};

pub(crate) struct FaissBackend {
    inner: IndexImpl,
    num_dim: usize,
    driver: IndexDriver,
}

fn faiss_spec(driver: IndexDriver) -> &'static str {
    match driver {
        IndexDriver::Exact => "Flat",
        IndexDriver::HNSW => "HNSW32",
        IndexDriver::HNSWDisk => {
            panic!("HNSWDisk must not be routed to FaissBackend")
        }
    }
}

fn faiss_map_err(e: FaissError) -> IndexError {
    IndexError::InvalidParameter(e.to_string())
}

impl FaissBackend {
    pub(crate) fn new(
        num_dim: usize,
        driver: IndexDriver,
        train_scaled: &ArrayView2<f64>,
    ) -> Result<Self, IndexError> {
        let inner = Self::make_index(num_dim, driver, train_scaled)?;
        Ok(Self {
            inner,
            num_dim,
            driver,
        })
    }

    fn make_index(
        num_dim: usize,
        driver: IndexDriver,
        train_scaled: &ArrayView2<f64>,
    ) -> Result<IndexImpl, IndexError> {
        let mut index =
            index_factory(num_dim as u32, faiss_spec(driver), MetricType::L2).map_err(faiss_map_err)?;
        if train_scaled.nrows() > 0 {
            let data = arr2_rows_to_f32(train_scaled);
            index.add(&data).map_err(faiss_map_err)?;
        }
        Ok(index)
    }

    pub(crate) fn len(&self) -> usize {
        self.inner.ntotal() as usize
    }

    /// Approximate in-memory footprint of vector storage plus HNSW graph links.
    pub(crate) fn memory_usage_bytes(&self) -> usize {
        let n = self.inner.ntotal() as usize;
        let d = self.inner.d() as usize;
        if n == 0 {
            return 0;
        }
        let vector_bytes = n
            .saturating_mul(d)
            .saturating_mul(std::mem::size_of::<f32>());
        match self.driver {
            IndexDriver::Exact => vector_bytes,
            IndexDriver::HNSW => {
                const M: usize = 32;
                let level0_links = n
                    .saturating_mul(M)
                    .saturating_mul(2)
                    .saturating_mul(std::mem::size_of::<i64>());
                vector_bytes.saturating_add(level0_links)
            }
            IndexDriver::HNSWDisk => {
                panic!("HNSWDisk must not be routed to FaissBackend")
            }
        }
    }

    pub(crate) fn rebuild(&mut self, train_scaled: &ArrayView2<f64>) -> Result<(), IndexError> {
        self.inner = Self::make_index(self.num_dim, self.driver, train_scaled)?;
        Ok(())
    }

    pub(crate) fn add(
        &mut self,
        rows_scaled: &ArrayView2<f64>,
        _start_key: u64,
    ) -> Result<(), IndexError> {
        let data = arr2_rows_to_f32(rows_scaled);
        self.inner.add(&data).map_err(faiss_map_err)
    }

    pub(crate) fn search(
        &mut self,
        queries_scaled: &ArrayView2<f64>,
        k_eff: usize,
        search_k: usize,
    ) -> Result<(ndarray::Array2<f64>, ndarray::Array2<i64>), IndexError> {
        let n_query = queries_scaled.nrows();
        let q = arr2_rows_to_f32(queries_scaled);
        let res = self.inner.search(&q, k_eff).map_err(faiss_map_err)?;
        let labels: Vec<i64> = res.labels.iter().map(|l| l.to_native()).collect();
        let (d, i) = unpack_batch_search(n_query, k_eff, &res.distances, &labels);
        Ok(pad_neighbor_cols_to_search_k(d, i, search_k))
    }
}

#[cfg(test)]
pub(crate) fn faiss_spec_for_test(driver: IndexDriver) -> &'static str {
    faiss_spec(driver)
}

#[cfg(test)]
pub(crate) fn faiss_map_err_for_test(e: FaissError) -> IndexError {
    faiss_map_err(e)
}

#[cfg(test)]
pub(crate) fn make_faiss_for_test(
    num_dim: usize,
    driver: IndexDriver,
    train_scaled: &ArrayView2<f64>,
) -> Result<IndexImpl, IndexError> {
    FaissBackend::make_index(num_dim, driver, train_scaled)
}

/// Grow the backing file to the exact row count needed (no pre-allocation tail).
const MMAP_GROW_ROWS: usize = 0;

pub(crate) struct MmapColumnStore {
    #[allow(dead_code)]
    pub(crate) path: PathBuf,
    pub(crate) ncols: usize,
    pub(crate) nrows: usize,
    file: File,
    mmap: MmapMut,
}

impl MmapColumnStore {
    fn row_bytes(&self) -> usize {
        self.ncols * std::mem::size_of::<f64>()
    }

    fn bytes_for_rows(&self, nrows: usize) -> usize {
        nrows.saturating_mul(self.row_bytes())
    }

    fn ensure_capacity(&mut self, need_rows: usize) -> Result<(), ENNError> {
        let need_bytes = self.bytes_for_rows(need_rows);
        if need_bytes <= self.mmap.len() {
            return Ok(());
        }
        let grow_rows = (need_rows - self.nrows).max(MMAP_GROW_ROWS);
        let new_len = self.bytes_for_rows(self.nrows + grow_rows);
        if !self.mmap.is_empty() {
            self.mmap
                .flush()
                .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
        }
        self.file
            .set_len(new_len as u64)
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
        self.mmap = unsafe {
            MmapMut::map_mut(&self.file)
                .map_err(|e| ENNError::InvalidParameter(e.to_string()))?
        };
        Ok(())
    }

    pub(crate) fn mmap_open_or_create(
        path: PathBuf,
        ncols: usize,
        known_nrows: Option<usize>,
    ) -> Result<Self, ENNError> {
        if !path.exists() {
            let file = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&path)
                .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
            drop(file);
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
        let len = file
            .metadata()
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))?
            .len();
        let row_bytes = ncols * std::mem::size_of::<f64>();
        let nrows = known_nrows.unwrap_or_else(|| {
            if row_bytes > 0 {
                (len as usize) / row_bytes
            } else {
                0
            }
        });
        if known_nrows.is_some() && nrows * row_bytes > len as usize {
            return Err(ENNError::InvalidParameter(format!(
                "known_nrows {nrows} exceeds train file bytes {len}"
            )));
        }
        let mmap = unsafe {
            MmapMut::map_mut(&file).map_err(|e| ENNError::InvalidParameter(e.to_string()))?
        };
        Ok(Self {
            path,
            ncols,
            nrows,
            file,
            mmap,
        })
    }

    pub(crate) fn mmap_append(&mut self, rows: &ArrayView2<f64>) -> Result<(), ENNError> {
        if rows.nrows() == 0 {
            return Ok(());
        }
        if rows.ncols() != self.ncols {
            return Err(ENNError::InvalidShape {
                expected: vec![self.nrows, self.ncols],
                got: vec![rows.nrows(), rows.ncols()],
            });
        }
        let new_nrows = self.nrows + rows.nrows();
        self.ensure_capacity(new_nrows)?;
        let row_bytes = self.row_bytes();
        for (i, row) in rows.axis_iter(Axis(0)).enumerate() {
            let offset = (self.nrows + i) * row_bytes;
            let dst = &mut self.mmap[offset..offset + row_bytes];
            let bytes = unsafe {
                std::slice::from_raw_parts(row.as_ptr() as *const u8, row_bytes)
            };
            dst.copy_from_slice(bytes);
        }
        self.nrows = new_nrows;
        Ok(())
    }

    pub(crate) fn mmap_row_slice(&self, i: usize) -> Result<&[f64], ENNError> {
        if i >= self.nrows {
            return Err(ENNError::InvalidParameter(format!(
                "row {i} out of range [0, {})",
                self.nrows
            )));
        }
        let start = i * self.ncols;
        let row_bytes = self.ncols * std::mem::size_of::<f64>();
        let byte_start = start * std::mem::size_of::<f64>();
        let byte_end = byte_start + row_bytes;
        let bytes = &self.mmap[byte_start..byte_end];
        let slice: &[f64] = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const f64, self.ncols)
        };
        Ok(slice)
    }

    pub(crate) fn mmap_gather(&self, indices: &[usize]) -> Result<Array2<f64>, ENNError> {
        let mut out = Array2::zeros((indices.len(), self.ncols));
        for (new_i, &old_i) in indices.iter().enumerate() {
            let row = self.mmap_row_slice(old_i)?;
            for j in 0..self.ncols {
                out[[new_i, j]] = row[j];
            }
        }
        Ok(out)
    }

    /// Copy rows `[start, end)` into a dense buffer (does not materialize the full store).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn mmap_row_range(
        &self,
        start: usize,
        end: usize,
    ) -> Result<Array2<f64>, ENNError> {
        if start > end {
            return Err(ENNError::InvalidParameter(format!(
                "mmap_row_range: start {start} > end {end}"
            )));
        }
        if end > self.nrows {
            return Err(ENNError::InvalidParameter(format!(
                "mmap_row_range end {end} out of range [0, {})",
                self.nrows
            )));
        }
        let n = end - start;
        if n == 0 {
            return Ok(Array2::zeros((0, self.ncols)));
        }
        let mut out = Array2::zeros((n, self.ncols));
        for (new_i, old_i) in (start..end).enumerate() {
            let row = self.mmap_row_slice(old_i)?;
            for j in 0..self.ncols {
                out[[new_i, j]] = row[j];
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod faiss_backend_tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn faiss_backend_exact_roundtrip() {
        let train = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let mut backend = FaissBackend::new(2, IndexDriver::Exact, &train.view()).unwrap();
        assert_eq!(backend.len(), 3);
        backend
            .add(&array![[1.0, 1.0]].view(), 3)
            .unwrap();
        assert_eq!(backend.len(), 4);
        let (d, i) = backend
            .search(&array![[0.0, 0.0]].view(), 2, 2)
            .unwrap();
        assert_eq!(i[[0, 0]], 0);
        assert!(d[[0, 0]] < 1e-5);
        backend.rebuild(&train.view()).unwrap();
        assert_eq!(backend.len(), 3);
    }

    #[test]
    fn faiss_spec_and_map_err() {
        assert_eq!(faiss_spec(IndexDriver::HNSW), "HNSW32");
        let err = faiss_map_err(faiss::error::Error::IndexDescription);
        assert!(matches!(err, IndexError::InvalidParameter(_)));
    }

    #[test]
    fn mmap_column_store_kiss_names() {
        let names = [
            "MmapColumnStore",
            "mmap_open_or_create",
            "mmap_append",
            "mmap_row_slice",
            "mmap_gather",
            "mmap_row_range",
        ];
        assert_eq!(names.len(), 6);
    }

    #[test]
    fn mmap_column_store_single_row_append_without_remap_churn() {
        use super::MmapColumnStore;
        use tempfile::TempDir;

        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("col.bin");
        let mut store = MmapColumnStore::mmap_open_or_create(path, 2, None).unwrap();
        for i in 0..5000 {
            store
                .mmap_append(&array![[i as f64, (i + 1) as f64]].view())
                .unwrap();
        }
        assert_eq!(store.nrows, 5000);
        assert_eq!(store.mmap_row_slice(4999).unwrap()[0], 4999.0);
    }

    #[test]
    fn mmap_column_store_direct_api() {
        use super::MmapColumnStore;
        use tempfile::TempDir;

        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("col.bin");
        let mut store = MmapColumnStore::mmap_open_or_create(path, 2, None).unwrap();
        store
            .mmap_append(&array![[1.0, 2.0], [3.0, 4.0]].view())
            .unwrap();
        assert_eq!(store.mmap_row_slice(1).unwrap()[0], 3.0);
        let gathered = store.mmap_gather(&[0, 1]).unwrap();
        assert_eq!(gathered.nrows(), 2);
        assert_eq!(store.mmap_row_range(0, store.nrows).unwrap().nrows(), 2);
        let mid = store.mmap_row_range(1, 2).unwrap();
        assert_eq!(mid[[0, 0]], 3.0);
    }

    #[test]
    fn faiss_backend_hnsw_search() {
        let train = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let mut backend = FaissBackend::new(2, IndexDriver::HNSW, &train.view()).unwrap();
        let (_d, i) = backend
            .search(&array![[0.0, 0.0]].view(), 2, 2)
            .unwrap();
        assert_eq!(i[[0, 0]], 0);
    }
}
