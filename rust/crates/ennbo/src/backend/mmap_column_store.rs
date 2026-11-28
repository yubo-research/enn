use memmap2::{Advice, MmapMut};
use ndarray::{Array2, ArrayView2};
use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::PathBuf;

use crate::error::ENNError;

const MMAP_GROW_ROWS: usize = 64;
const MAX_RUN_BYTES: usize = 1 << 20;

static LATCHED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub(crate) fn auto_us() -> Option<f64> {
    static US: std::sync::OnceLock<Option<f64>> = std::sync::OnceLock::new();
    *US.get_or_init(|| std::env::var("ENN_ADV_AUTO_US").ok().and_then(|v| v.parse().ok()))
}

fn pread_mode() -> u8 {
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| match std::env::var("ENN_STORE_PREAD").as_deref() {
        Ok("1") => 2,
        Ok("build") => 1,
        _ => 0,
    })
}

pub(crate) fn store_pread() -> bool {
    pread_mode() >= 1
}

pub(crate) fn query_pread() -> bool {
    pread_mode() >= 2
}

pub(crate) fn latch() -> bool {
    !LATCHED.swap(true, std::sync::atomic::Ordering::Relaxed)
}

pub(crate) fn random_access(path: &std::path::Path) -> bool {
    static SPEC: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    let Some(spec) = SPEC.get_or_init(|| std::env::var("ENN_ADV_RANDOM").ok()) else {
        return false;
    };
    let armed = auto_us().is_none() || LATCHED.load(std::sync::atomic::Ordering::Relaxed);
    armed && spec_matches(spec, path)
}

pub(crate) fn spec_matches(spec: &str, path: &std::path::Path) -> bool {
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    spec.split(',')
        .any(|t| t == "1" || t == "all" || (!t.is_empty() && name.contains(t)))
}

pub(crate) fn advise_random(
    path: &std::path::Path,
    map: &[u8],
    advise: impl FnOnce(Advice) -> std::io::Result<()>,
) {
    if !map.is_empty() && random_access(path) {
        let _ = advise(Advice::Random);
    }
}

pub struct MmapColumnStore {
    #[allow(dead_code)]
    pub(crate) path: PathBuf,
    pub(crate) ncols: usize,
    pub(crate) nrows: usize,
    file: File,
    mmap: MmapMut,
}

impl MmapColumnStore {
    pub(crate) fn row_bytes(&self) -> usize {
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
        let grow_rows = (need_rows - self.nrows).max(self.nrows).max(MMAP_GROW_ROWS);
        let new_len = self.bytes_for_rows(self.nrows + grow_rows);
        self.file
            .set_len(new_len as u64)
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
        self.mmap = unsafe {
            MmapMut::map_mut(&self.file).map_err(|e| ENNError::InvalidParameter(e.to_string()))?
        };
        advise_random(&self.path, &self.mmap, |a| self.mmap.advise(a));
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
        advise_random(&path, &mmap, |a| mmap.advise(a));
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
        crate::o1::io_streams::wrote(crate::o1::io_streams::Stream::StoreAppend, offset as u64, byte_len);
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

    pub(crate) fn advise_now(&self) {
        advise_random(&self.path, &self.mmap, |a| self.mmap.advise(a));
    }

    pub(crate) fn file_handle(&self) -> Result<File, ENNError> {
        self.file
            .try_clone()
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))
    }

    pub(crate) fn flush(&self) -> Result<(), ENNError> {
        self.mmap
            .flush()
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))
    }

    pub(crate) fn pread_runs(
        &self,
        ids: &[u32],
        gap: usize,
        out: &mut Vec<f64>,
    ) -> Result<(), ENNError> {
        let rb = self.row_bytes();
        let mut buf = Vec::new();
        let mut a = 0;
        while a < ids.len() {
            let mut b = a + 1;
            while b < ids.len()
                && (ids[b] - ids[b - 1]) as usize * rb <= gap + rb
                && (ids[b] - ids[a] + 1) as usize * rb <= MAX_RUN_BYTES
            {
                b += 1;
            }
            let (lo, hi) = (ids[a] as usize, ids[b - 1] as usize + 1);
            if hi > self.nrows {
                return Err(ENNError::InvalidParameter(format!(
                    "row {} out of range [0, {})",
                    hi - 1,
                    self.nrows
                )));
            }
            buf.resize((hi - lo) * rb, 0);
            let t0 = std::time::Instant::now();
            self.file
                .read_exact_at(&mut buf, (lo * rb) as u64)
                .map_err(|e| ENNError::InvalidParameter(format!("rows {lo}..{hi} read failed: {e}")))?;
            crate::o1::io_streams::read(crate::o1::io_streams::Stream::StoreRead, (lo * rb) as u64, buf.len(), t0);
            for &id in &ids[a..b] {
                let at = (id as usize - lo) * rb;
                out.extend(
                    buf[at..at + rb]
                        .chunks_exact(8)
                        .map(|c| f64::from_ne_bytes(c.try_into().unwrap_or([0; 8]))),
                );
            }
            a = b;
        }
        Ok(())
    }

    pub(crate) fn pread_row(&self, i: usize) -> Result<Vec<f64>, ENNError> {
        if i >= self.nrows {
            return Err(ENNError::InvalidParameter(format!(
                "row {i} out of range [0, {})",
                self.nrows
            )));
        }
        let mut bytes = vec![0u8; self.row_bytes()];
        let t0 = std::time::Instant::now();
        self.file
            .read_exact_at(&mut bytes, (i * self.row_bytes()) as u64)
            .map_err(|e| ENNError::InvalidParameter(format!("row {i} read failed: {e}")))?;
        crate::o1::io_streams::read(crate::o1::io_streams::Stream::StoreRead, (i * self.row_bytes()) as u64, bytes.len(), t0);
        Ok(bytes
            .chunks_exact(8)
            .map(|c| f64::from_ne_bytes(c.try_into().unwrap_or([0; 8])))
            .collect())
    }

    pub(crate) fn pread_bytes(&self, start: usize, end: usize) -> Result<Vec<u8>, ENNError> {
        if start > end || end > self.nrows {
            return Err(ENNError::InvalidParameter(format!(
                "pread_bytes {start}..{end} out of range [0, {})",
                self.nrows
            )));
        }
        let mut bytes = vec![0u8; (end - start) * self.row_bytes()];
        let t0 = std::time::Instant::now();
        self.file
            .read_exact_at(&mut bytes, (start * self.row_bytes()) as u64)
            .map_err(|e| {
                ENNError::InvalidParameter(format!("rows {start}..{end} read failed: {e}"))
            })?;
        crate::o1::io_streams::read(crate::o1::io_streams::Stream::StoreRead, (start * self.row_bytes()) as u64, bytes.len(), t0);
        Ok(bytes)
    }

    pub(crate) fn pread_gather(&self, indices: &[usize]) -> Result<Array2<f64>, ENNError> {
        let mut out = Array2::zeros((indices.len(), self.ncols));
        for (r, &i) in indices.iter().enumerate() {
            out.row_mut(r)
                .assign(&ndarray::ArrayView1::from(&self.pread_row(i)?[..]));
        }
        Ok(out)
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

    pub(crate) fn mmap_rows_view(
        &self,
        start: usize,
        end: usize,
    ) -> Result<ArrayView2<'_, f64>, ENNError> {
        if start > end || end > self.nrows {
            return Err(ENNError::InvalidParameter(format!(
                "mmap_rows_view {start}..{end} out of range [0, {})",
                self.nrows
            )));
        }
        let bytes = &self.mmap[start * self.row_bytes()..end * self.row_bytes()];
        let vals: &[f64] = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const f64, (end - start) * self.ncols)
        };
        ArrayView2::from_shape((end - start, self.ncols), vals)
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))
    }

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
    #[test]
    fn random_advice_spec_matches_file_name_tokens() {
        use super::spec_matches;
        let x = std::path::Path::new("/tmp/s/o1_x.bin");
        let copy = std::path::Path::new("/tmp/s/o1f_xo_copy_14.bin");
        assert!(spec_matches("1", x) && spec_matches("all", copy));
        assert!(spec_matches("map,o1_x", x) && !spec_matches("map,o1_x", copy));
        assert!(spec_matches("copy", copy) && !spec_matches("", copy));
    }

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
    fn pread_runs_matches_mapped_rows() {
        let dir = tempfile::TempDir::new().expect("tempdir");
        let mut store =
            MmapColumnStore::mmap_open_or_create(dir.path().join("c.bin"), 3, None).unwrap();
        let x = ndarray::Array2::from_shape_fn((900, 3), |(i, j)| (i * 3 + j) as f64);
        store.mmap_append(&x.view()).unwrap();
        let ids: Vec<u32> = vec![0, 1, 2, 7, 8, 200, 201, 650, 899];
        for gap in [0, 64, 1 << 20] {
            let mut out = Vec::new();
            store.pread_runs(&ids, gap, &mut out).unwrap();
            let want: Vec<f64> = ids
                .iter()
                .flat_map(|&i| store.mmap_row_slice(i as usize).unwrap().to_vec())
                .collect();
            assert_eq!(out, want);
        }
        assert!(store.pread_runs(&[900], 0, &mut Vec::new()).is_err());
        let big = ndarray::Array2::from_shape_fn((60_000, 3), |(i, j)| ((i + 900) * 3 + j) as f64);
        store.mmap_append(&big.view()).unwrap();
        let wide: Vec<u32> = (0..60_900).step_by(7).collect();
        let mut out = Vec::new();
        store.pread_runs(&wide, usize::MAX / 2, &mut out).unwrap();
        assert_eq!(out.len(), wide.len() * 3);
        assert!(wide.iter().enumerate().all(|(k, &i)| out[3 * k] == f64::from(i * 3)));
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
