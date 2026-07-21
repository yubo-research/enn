use memmap2::MmapMut;
use ndarray::{Array2, ArrayView2, Axis};
use std::fs::{File, OpenOptions};
use std::path::PathBuf;

use crate::error::BpannError;

pub(crate) const MMAP_GROW_ROWS: usize = 64;

pub struct MmapColumnStore {
    pub path: PathBuf,
    pub ncols: usize,
    pub nrows: usize,
    file: File,
    mmap: MmapMut,
}

impl MmapColumnStore {
    pub(crate) fn row_bytes(&self) -> usize {
        self.ncols * std::mem::size_of::<f64>()
    }

    pub(crate) fn bytes_for_rows(&self, nrows: usize) -> usize {
        nrows.saturating_mul(self.row_bytes())
    }

    pub(crate) fn ensure_capacity(&mut self, need_rows: usize) -> Result<(), BpannError> {
        let need_bytes = self.bytes_for_rows(need_rows);
        if need_bytes <= self.mmap.len() {
            return Ok(());
        }
        let grow_rows = (need_rows - self.nrows).max(MMAP_GROW_ROWS);
        let new_len = self.bytes_for_rows(self.nrows + grow_rows);
        // No flush before remapping: dirty pages live in the page cache and
        // survive remapping the same file; msync is only needed for durability.
        self.file
            .set_len(new_len as u64)
            .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        self.mmap = unsafe {
            MmapMut::map_mut(&self.file).map_err(|e| BpannError::InvalidParameter(e.to_string()))?
        };
        Ok(())
    }

    pub fn mmap_open_or_create(
        path: PathBuf,
        ncols: usize,
        known_nrows: Option<usize>,
    ) -> Result<Self, BpannError> {
        if !path.exists() {
            let file = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(&path)
                .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
            drop(file);
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        let len = file
            .metadata()
            .map_err(|e| BpannError::InvalidParameter(e.to_string()))?
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
            return Err(BpannError::InvalidParameter(format!(
                "known_nrows {nrows} exceeds train file bytes {len}"
            )));
        }
        let mmap = unsafe {
            MmapMut::map_mut(&file).map_err(|e| BpannError::InvalidParameter(e.to_string()))?
        };
        Ok(Self {
            path,
            ncols,
            nrows,
            file,
            mmap,
        })
    }

    /// Touch every page of the current mapping so first appends pay no
    /// page-fault or block-allocation cost.
    pub(crate) fn pretouch(&mut self) {
        const PAGE: usize = 4096;
        let len = self.mmap.len();
        let mut i = 0;
        while i < len {
            unsafe { std::ptr::write_volatile(self.mmap.as_mut_ptr().add(i), 0) };
            i += PAGE;
        }
    }

    pub fn mmap_append(&mut self, rows: &ArrayView2<f64>) -> Result<(), BpannError> {
        if rows.nrows() == 0 {
            return Ok(());
        }
        if rows.ncols() != self.ncols {
            return Err(BpannError::InvalidShape {
                expected: vec![self.nrows, self.ncols],
                got: vec![rows.nrows(), rows.ncols()],
            });
        }
        let new_nrows = self.nrows + rows.nrows();
        self.ensure_capacity(new_nrows)?;
        let row_bytes = self.row_bytes();
        let offset = self.nrows * row_bytes;
        let n = rows.nrows() * self.ncols;
        let dst = &mut self.mmap[offset..offset + n * std::mem::size_of::<f64>()];
        // Bulk memcpy only when the source is C-contiguous. Non-standard layouts
        // (e.g. Fortran order) have strided rows; copying `row_bytes` from
        // `row.as_ptr()` would silently write the wrong values.
        if let Some(src) = rows.as_slice() {
            let src_bytes = unsafe {
                std::slice::from_raw_parts(src.as_ptr() as *const u8, n * std::mem::size_of::<f64>())
            };
            dst.copy_from_slice(src_bytes);
        } else {
            let dst_f64 =
                unsafe { std::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut f64, n) };
            for (i, row) in rows.axis_iter(Axis(0)).enumerate() {
                let row_dst = &mut dst_f64[i * self.ncols..(i + 1) * self.ncols];
                for (d, s) in row_dst.iter_mut().zip(row.iter()) {
                    *d = *s;
                }
            }
        }
        self.nrows = new_nrows;
        Ok(())
    }

    /// Drop faulted/dirty pages from process RSS while keeping the file mapping.
    ///
    /// Remaps the same file so previously resident pages are no longer charged to
    /// this process. File contents are unchanged; subsequent reads fault pages back
    /// in (typically via the OS file cache). Call only when no borrows into `mmap`
    /// are live.
    ///
    /// Does not `flush()` the whole mapping first: a full-map flush can briefly
    /// force large residency (hurting `ru_maxrss`) on multi-GB stores.
    pub fn release_resident_pages(&mut self) -> Result<(), BpannError> {
        self.mmap = unsafe {
            MmapMut::map_mut(&self.file).map_err(|e| BpannError::InvalidParameter(e.to_string()))?
        };
        Ok(())
    }

    pub fn mmap_row_slice(&self, i: usize) -> Result<&[f64], BpannError> {
        if i >= self.nrows {
            return Err(BpannError::InvalidParameter(format!(
                "row {i} out of range [0, {})",
                self.nrows
            )));
        }
        let byte_start = i * self.row_bytes();
        let byte_end = byte_start + self.row_bytes();
        let bytes = &self.mmap[byte_start..byte_end];
        let slice: &[f64] =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f64, self.ncols) };
        Ok(slice)
    }

    pub fn mmap_gather(&self, indices: &[usize]) -> Result<Array2<f64>, BpannError> {
        let mut out = Array2::zeros((indices.len(), self.ncols));
        for (new_i, &old_i) in indices.iter().enumerate() {
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
    use tempfile::TempDir;

    #[test]
    fn row_bytes() {
        let dir = TempDir::new().unwrap();
        let store =
            MmapColumnStore::mmap_open_or_create(dir.path().join("c.bin"), 2, None).unwrap();
        assert_eq!(store.row_bytes(), 16);
    }

    #[test]
    fn bytes_for_rows() {
        let dir = TempDir::new().unwrap();
        let store =
            MmapColumnStore::mmap_open_or_create(dir.path().join("c.bin"), 2, None).unwrap();
        assert_eq!(store.bytes_for_rows(5), 80);
    }

    #[test]
    fn ensure_capacity() {
        let dir = TempDir::new().unwrap();
        let mut store =
            MmapColumnStore::mmap_open_or_create(dir.path().join("c.bin"), 2, None).unwrap();
        store.ensure_capacity(8).unwrap();
        store
            .mmap_append(&array![[1.0, 2.0], [3.0, 4.0]].view())
            .unwrap();
        assert_eq!(store.nrows, 2);
    }

    #[test]
    fn release_resident_pages_preserves_rows() {
        let dir = TempDir::new().unwrap();
        let mut store =
            MmapColumnStore::mmap_open_or_create(dir.path().join("c.bin"), 2, None).unwrap();
        store
            .mmap_append(&array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]].view())
            .unwrap();
        store.release_resident_pages().unwrap();
        assert_eq!(store.mmap_row_slice(0).unwrap(), &[1.0, 2.0]);
        assert_eq!(store.mmap_row_slice(2).unwrap(), &[5.0, 6.0]);
        store
            .mmap_append(&array![[7.0, 8.0]].view())
            .unwrap();
        store.release_resident_pages().unwrap();
        assert_eq!(store.mmap_row_slice(3).unwrap(), &[7.0, 8.0]);
    }
}
