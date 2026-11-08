use crate::error::ENNError;
use ndarray::{Array2, ArrayView2};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;
use std::sync::Mutex;

const HEADER_BYTES: u64 = 24;
const MAGIC: &[u8; 8] = b"ENNINPUT";

pub(crate) struct DiskInputs {
    file: Mutex<File>,
    ncols: usize,
    nrows: usize,
}

fn invalid(message: impl std::fmt::Display) -> ENNError {
    ENNError::InvalidParameter(format!("original inputs: {message}"))
}

impl DiskInputs {
    pub(crate) fn create(
        directory: &Path,
        filename: &str,
        rows: &ArrayView2<f64>,
    ) -> Result<Self, ENNError> {
        let mut file = OpenOptions::new()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(directory.join(filename))
            .map_err(invalid)?;
        let ncols = rows.ncols();
        file.write_all(MAGIC).map_err(invalid)?;
        file.write_all(&(ncols as u64).to_le_bytes())
            .map_err(invalid)?;
        file.write_all(&0_u64.to_le_bytes()).map_err(invalid)?;
        let mut store = Self {
            file: Mutex::new(file),
            ncols,
            nrows: 0,
        };
        store.append(rows)?;
        Ok(store)
    }

    pub(crate) fn reopen(
        directory: &Path,
        filename: &str,
    ) -> Result<Option<(Self, usize)>, ENNError> {
        let path = directory.join(filename);
        if !path.exists() {
            return Ok(None);
        }
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .map_err(invalid)?;
        let (ncols, nrows) = read_header(&mut file)?;
        let mut log = Self {
            file: Mutex::new(file),
            ncols,
            nrows,
        };
        log.keep_rows(nrows)?;
        Ok(Some((log, nrows)))
    }

    pub(crate) fn keep_rows(&mut self, rows: usize) -> Result<(), ENNError> {
        if rows > self.nrows {
            return Err(invalid("stored original rows do not match the disk model"));
        }
        let end = self.row_offset(rows)?;
        let file = self.file.get_mut().map_err(invalid)?;
        if file.metadata().map_err(invalid)?.len() < end {
            return Err(invalid("stored original rows do not match the disk model"));
        }
        file.set_len(end).map_err(invalid)?;
        file.seek(SeekFrom::Start(16)).map_err(invalid)?;
        file.write_all(&(rows as u64).to_le_bytes())
            .map_err(invalid)?;
        file.sync_data().map_err(invalid)?;
        self.nrows = rows;
        Ok(())
    }

    fn row_offset(&self, rows: usize) -> Result<u64, ENNError> {
        (self.ncols as u64)
            .checked_mul(rows as u64)
            .and_then(|v| v.checked_mul(8))
            .and_then(|v| v.checked_add(HEADER_BYTES))
            .ok_or_else(|| invalid("file size overflow"))
    }

    pub(crate) fn append(&mut self, rows: &ArrayView2<f64>) -> Result<(), ENNError> {
        if rows.ncols() != self.ncols {
            return Err(invalid("input width changed"));
        }
        let nrows = self
            .nrows
            .checked_add(rows.nrows())
            .ok_or_else(|| invalid("row count overflow"))?;
        let file = self.file.get_mut().map_err(invalid)?;
        file.seek(SeekFrom::End(0)).map_err(invalid)?;
        {
            let mut writer = BufWriter::new(&mut *file);
            for row in rows.rows() {
                for value in row {
                    writer.write_all(&value.to_le_bytes()).map_err(invalid)?;
                }
            }
            writer.flush().map_err(invalid)?;
        }
        file.seek(SeekFrom::Start(16)).map_err(invalid)?;
        file.write_all(&(nrows as u64).to_le_bytes())
            .map_err(invalid)?;
        file.sync_data().map_err(invalid)?;
        self.nrows = nrows;
        Ok(())
    }

    pub(crate) fn gather(&self, indices: &[usize]) -> Result<Array2<f64>, ENNError> {
        if indices.iter().any(|&i| i >= self.nrows) {
            return Err(invalid("observation index out of range"));
        }
        let mut file = self.file.lock().map_err(invalid)?;
        let mut out = Array2::zeros((indices.len(), self.ncols));
        let mut bytes = vec![
            0;
            self.ncols
                .checked_mul(8)
                .ok_or_else(|| invalid("input width overflow"))?
        ];
        for (row, &index) in indices.iter().enumerate() {
            file.seek(SeekFrom::Start(
                HEADER_BYTES + index as u64 * bytes.len() as u64,
            ))
            .map_err(invalid)?;
            file.read_exact(&mut bytes).map_err(invalid)?;
            for (j, value) in bytes.chunks_exact(8).enumerate() {
                out[[row, j]] = f64::from_le_bytes(value.try_into().map_err(invalid)?);
            }
        }
        Ok(out)
    }
}

fn read_header(file: &mut File) -> Result<(usize, usize), ENNError> {
    let mut header = [0; HEADER_BYTES as usize];
    file.read_exact(&mut header).map_err(invalid)?;
    if &header[..8] != MAGIC {
        return Err(invalid("not an observation log"));
    }
    let ncols = u64::from_le_bytes(header[8..16].try_into().map_err(invalid)?);
    let nrows = u64::from_le_bytes(header[16..24].try_into().map_err(invalid)?);
    Ok((ncols as usize, nrows as usize))
}
