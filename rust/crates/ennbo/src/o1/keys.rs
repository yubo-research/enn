use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use super::lsh::io_error;
use crate::error::ENNError;

pub(crate) fn open_rw(path: &Path, truncate: bool) -> Result<File, ENNError> {
    OpenOptions::new()
        .create(true)
        .truncate(truncate)
        .read(true)
        .write(true)
        .open(path)
        .map_err(io_error)
}

pub(crate) fn install(tmp: &Path, path: &Path) -> Result<(), ENNError> {
    match std::fs::remove_file(path) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => return Err(io_error(e)),
        _ => {}
    }
    std::fs::rename(tmp, path).map_err(io_error)
}

pub(crate) fn file_len(path: &Path) -> u64 {
    std::fs::metadata(path).map_or(0, |m| m.len())
}

pub(crate) struct KeyFile {
    path: PathBuf,
    file: File,
}

impl KeyFile {
    pub(crate) fn open(path: PathBuf, truncate: bool) -> Result<Self, ENNError> {
        let file = open_rw(&path, truncate)?;
        Ok(Self { path, file })
    }

    pub(crate) fn tmp_path(&self) -> PathBuf {
        self.path.with_extension("tmp")
    }

    pub(crate) fn len(&self) -> usize {
        (file_len(&self.path) / 8) as usize
    }

    pub(crate) fn write_at(&self, row: usize, keys: &[u64]) -> Result<(), ENNError> {
        let bytes: Vec<u8> = keys.iter().flat_map(|k| k.to_le_bytes()).collect();
        self.file.write_all_at(&bytes, row as u64 * 8).map_err(io_error)
    }

    pub(crate) fn read_at(&self, row: usize, out: &mut [u64]) -> Result<(), ENNError> {
        let mut bytes = vec![0u8; out.len() * 8];
        self.file.read_exact_at(&mut bytes, row as u64 * 8).map_err(io_error)?;
        for (k, b) in out.iter_mut().zip(bytes.chunks_exact(8)) {
            *k = u64::from_le_bytes(b.try_into().expect("8-byte chunk"));
        }
        Ok(())
    }

    pub(crate) fn truncate(&self, rows: usize) -> Result<(), ENNError> {
        self.file.set_len(rows as u64 * 8).map_err(io_error)
    }

    pub(crate) fn replace_with(&mut self, tmp: &Path) -> Result<(), ENNError> {
        install(tmp, &self.path)?;
        self.file = open_rw(&self.path, false)?;
        Ok(())
    }
}

pub(crate) struct KeyWriter {
    path: PathBuf,
    out: BufWriter<File>,
}

impl KeyWriter {
    pub(crate) fn create(path: PathBuf) -> Result<Self, ENNError> {
        let out = BufWriter::with_capacity(1 << 16, open_rw(&path, true)?);
        Ok(Self { path, out })
    }

    pub(crate) fn push(&mut self, keys: &[u64]) -> Result<(), ENNError> {
        for k in keys {
            self.out.write_all(&k.to_le_bytes()).map_err(io_error)?;
        }
        Ok(())
    }

    pub(crate) fn finish(mut self) -> Result<PathBuf, ENNError> {
        self.out.flush().map_err(io_error)?;
        Ok(self.path)
    }
}
