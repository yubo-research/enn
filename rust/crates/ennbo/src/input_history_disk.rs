//! Bounded-memory original-input storage for noninvertible disk features.
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
    pub(crate) fn open(
        directory: &Path,
        rows: &ArrayView2<f64>,
        expected_rows: usize,
    ) -> Result<Self, ENNError> {
        let path = directory.join("original_inputs.bin");
        let reopen = rows.nrows() == 0 && path.exists();
        if !reopen && rows.nrows() != expected_rows {
            return Err(invalid(
                "missing original rows for semantic-only disk model",
            ));
        }
        let mut file = OpenOptions::new()
            .create(!reopen)
            .truncate(!reopen)
            .read(true)
            .write(true)
            .open(path)
            .map_err(invalid)?;
        let ncols = rows.ncols();
        let nrows = if reopen {
            read_header(&mut file, ncols, expected_rows)?;
            expected_rows
        } else {
            file.write_all(MAGIC).map_err(invalid)?;
            file.write_all(&(ncols as u64).to_le_bytes())
                .map_err(invalid)?;
            file.write_all(&0_u64.to_le_bytes()).map_err(invalid)?;
            0
        };
        let mut store = Self {
            file: Mutex::new(file),
            ncols,
            nrows,
        };
        if !reopen {
            store.append(rows)?;
        }
        Ok(store)
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

fn read_header(file: &mut File, dimensions: usize, rows: usize) -> Result<(), ENNError> {
    let mut header = [0; HEADER_BYTES as usize];
    file.read_exact(&mut header).map_err(invalid)?;
    let width = u64::from_le_bytes(header[8..16].try_into().map_err(invalid)?);
    let count = u64::from_le_bytes(header[16..24].try_into().map_err(invalid)?);
    let bytes = width
        .checked_mul(count)
        .and_then(|v| v.checked_mul(8))
        .and_then(|v| v.checked_add(HEADER_BYTES))
        .ok_or_else(|| invalid("file size overflow"))?;
    if &header[..8] != MAGIC
        || width != dimensions as u64
        || count != rows as u64
        || file.metadata().map_err(invalid)?.len() != bytes
    {
        return Err(invalid("stored original rows do not match the disk model"));
    }
    Ok(())
}
