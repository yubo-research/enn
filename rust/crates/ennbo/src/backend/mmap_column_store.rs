use memmap2::MmapMut;
use ndarray::{Array2, ArrayView2};
use std::fs::{File, OpenOptions};
use std::path::PathBuf;

use crate::error::ENNError;

const MMAP_GROW_ROWS: usize = 64;

pub struct MmapColumnStore {
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
            MmapMut::map_mut(&self.file).map_err(|e| ENNError::InvalidParameter(e.to_string()))?
        };
        Ok(())
    }

    pub fn mmap_open_or_create(
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
        let nrows =
            known_nrows.unwrap_or_else(|| (len as usize).checked_div(row_bytes).unwrap_or(0));
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

    pub fn mmap_append(&mut self, rows: &ArrayView2<f64>) -> Result<(), ENNError> {
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
        let offset = self.nrows * self.row_bytes();
        let n = rows.nrows() * self.ncols;
        let byte_len = n * std::mem::size_of::<f64>();
        let dst = &mut self.mmap[offset..offset + byte_len];

        let contiguous = rows.as_standard_layout();
        let src = contiguous
            .as_slice()
            .expect("as_standard_layout yields a contiguous f64 slice");
        let src_bytes = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u8, byte_len) };
        dst.copy_from_slice(src_bytes);
        self.nrows = new_nrows;
        Ok(())
    }

    pub fn mmap_row_slice(&self, i: usize) -> Result<&[f64], ENNError> {
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
        let slice: &[f64] =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f64, self.ncols) };
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

    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn mmap_row_range(&self, start: usize, end: usize) -> Result<Array2<f64>, ENNError> {
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
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    #[allow(non_snake_case)]
    fn MmapColumnStore() {
        use tempfile::TempDir;

        let dir = TempDir::new().expect("tempdir");
        let store =
            MmapColumnStore::mmap_open_or_create(dir.path().join("c.bin"), 2, None).unwrap();
        assert_eq!(store.ncols, 2);
    }

    #[test]
    fn row_bytes() {
        use tempfile::TempDir;

        let dir = TempDir::new().expect("tempdir");
        let store =
            MmapColumnStore::mmap_open_or_create(dir.path().join("c.bin"), 2, None).unwrap();
        assert_eq!(store.row_bytes(), 16);
    }

    #[test]
    fn bytes_for_rows() {
        use tempfile::TempDir;

        let dir = TempDir::new().expect("tempdir");
        let store =
            MmapColumnStore::mmap_open_or_create(dir.path().join("c.bin"), 2, None).unwrap();
        assert_eq!(store.bytes_for_rows(4), 64);
    }

    #[test]
    fn ensure_capacity() {
        use tempfile::TempDir;

        let dir = TempDir::new().expect("tempdir");
        let mut store =
            MmapColumnStore::mmap_open_or_create(dir.path().join("c.bin"), 2, None).unwrap();
        store.ensure_capacity(3).unwrap();
        store
            .mmap_append(&array![[1.0, 2.0], [3.0, 4.0]].view())
            .unwrap();
        assert_eq!(store.nrows, 2);
    }

    #[test]
    fn mmap_column_store_single_row_append_without_remap_churn() {
        use super::MmapColumnStore;
        use tempfile::TempDir;

        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("col.bin");
        let mut store = MmapColumnStore::mmap_open_or_create(path, 2, None).unwrap();
        let n = 400usize;
        store.ensure_capacity(n).unwrap();
        for i in 0..n {
            store
                .mmap_append(&array![[i as f64, (i + 1) as f64]].view())
                .unwrap();
        }
        assert_eq!(store.nrows, n);
        assert_eq!(store.mmap_row_slice(n - 1).unwrap()[0], (n - 1) as f64);
    }

    #[test]
    fn mmap_append_fortran_order_preserves_rows() {
        use ndarray::{Array2, ShapeBuilder};
        use tempfile::TempDir;

        let dir = TempDir::new().expect("tempdir");
        let mut store =
            MmapColumnStore::mmap_open_or_create(dir.path().join("c.bin"), 3, None).unwrap();
        let mut f = Array2::<f64>::zeros((2, 3).f());
        f[[0, 0]] = 1.0;
        f[[0, 1]] = 2.0;
        f[[0, 2]] = 3.0;
        f[[1, 0]] = 4.0;
        f[[1, 1]] = 5.0;
        f[[1, 2]] = 6.0;
        assert!(!f.is_standard_layout());
        store.mmap_append(&f.view()).unwrap();
        assert_eq!(
            store.mmap_row_slice(0).unwrap(),
            &[1.0, 2.0, 3.0],
            "row0 corrupted under Fortran layout"
        );
        assert_eq!(store.mmap_row_slice(1).unwrap(), &[4.0, 5.0, 6.0]);
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
}
