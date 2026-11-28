use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use super::hlsm_frame::TableFrame;
use super::lsh::io_error;
use crate::error::ENNError;

pub(crate) const ENTRY_HEAD: usize = 12;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct ERunMeta {
    pub id: u64,
    pub table: usize,
    pub level: u32,
    pub rows: u64,
    #[serde(default)]
    pub lo: u64,
    #[serde(default)]
    pub hi: u64,
}

pub(crate) fn run_path(dir: &Path, id: u64) -> PathBuf {
    dir.join(format!("o1e_run_{id}.bin"))
}

fn fence_path(dir: &Path, id: u64) -> PathBuf {
    dir.join(format!("o1e_fence_{id}.bin"))
}

fn frame_path(dir: &Path, id: u64) -> PathBuf {
    dir.join(format!("o1e_frame_{id}.bin"))
}

pub(crate) fn write_frame(dir: &Path, id: u64, frame: &TableFrame) -> Result<u64, ENNError> {
    let b = frame.to_bytes();
    std::fs::write(frame_path(dir, id), &b).map_err(io_error)?;
    Ok(b.len() as u64)
}

pub(crate) fn remove_run(dir: &Path, id: u64) {
    let _ = std::fs::remove_file(run_path(dir, id));
    let _ = std::fs::remove_file(fence_path(dir, id));
    let _ = std::fs::remove_file(frame_path(dir, id));
}

pub(crate) fn entry_hash(e: &[u8]) -> u64 {
    u64::from_le_bytes(e[..8].try_into().unwrap_or([0; 8]))
}

pub(crate) fn entry_id(e: &[u8]) -> u32 {
    u32::from_le_bytes(e[8..12].try_into().unwrap_or([0; 4]))
}

pub(crate) fn fence_stride(entry: usize) -> usize {
    (4096 / entry).max(1)
}

pub(crate) struct ERunWriter {
    out: BufWriter<File>,
    dir: PathBuf,
    id: u64,
    entry: usize,
    fences: Vec<u64>,
    rows: u64,
}

impl ERunWriter {
    pub(crate) fn create(dir: &Path, id: u64, entry: usize) -> Result<Self, ENNError> {
        let f = File::create(run_path(dir, id)).map_err(io_error)?;
        Ok(Self { out: BufWriter::with_capacity(1 << 20, f), dir: dir.to_path_buf(), id, entry, fences: Vec::new(), rows: 0 })
    }

    pub(crate) fn push(&mut self, e: &[u8]) -> Result<(), ENNError> {
        debug_assert_eq!(e.len(), self.entry);
        if self.rows.is_multiple_of(fence_stride(self.entry) as u64) {
            self.fences.push(entry_hash(e));
        }
        self.out.write_all(e).map_err(io_error)?;
        self.rows += 1;
        Ok(())
    }

    pub(crate) fn finish(self) -> Result<(u64, u64), ENNError> {
        let f = self.out.into_inner().map_err(|e| io_error(e.into_error()))?;
        f.sync_data().map_err(io_error)?;
        let fb: Vec<u8> = self.fences.iter().flat_map(|h| h.to_le_bytes()).collect();
        std::fs::write(fence_path(&self.dir, self.id), &fb).map_err(io_error)?;
        Ok((self.rows, self.rows * self.entry as u64 + fb.len() as u64))
    }
}

pub(crate) struct ERun {
    pub meta: ERunMeta,
    pub frame: TableFrame,
    file: File,
    fences: Vec<u64>,
}

impl ERun {
    pub(crate) fn open(dir: &Path, meta: ERunMeta) -> Result<Self, ENNError> {
        let file = File::open(run_path(dir, meta.id)).map_err(io_error)?;
        let fb = std::fs::read(fence_path(dir, meta.id)).map_err(io_error)?;
        let fences = fb.chunks_exact(8).map(entry_hash).collect();
        let frame = std::fs::read(frame_path(dir, meta.id)).ok().and_then(|b| TableFrame::from_bytes(&b));
        let frame = frame.ok_or_else(|| io_error(std::io::Error::other(format!("O1E run {} has no readable frame", meta.id))))?;
        Ok(Self { meta, frame, file, fences })
    }

    pub(crate) fn fence_bytes(&self) -> usize {
        8 * self.fences.len()
    }

    pub(crate) fn window(&self, h: u64, width: usize, entry: usize) -> (usize, usize) {
        let rows = self.meta.rows as usize;
        let stride = fence_stride(entry);
        let f = self.fences.partition_point(|&x| x < h);
        let centre = (f * stride).saturating_sub(stride / 2).min(rows);
        let lo = centre.saturating_sub(width / 2);
        let hi = (lo + width).min(rows);
        (hi.saturating_sub(width), hi)
    }

    pub(crate) fn read(&self, lo: usize, hi: usize, entry: usize, buf: &mut Vec<u8>) -> Result<(), ENNError> {
        buf.resize((hi - lo) * entry, 0);
        self.file.read_exact_at(buf, (lo * entry) as u64).map_err(io_error)
    }
}

struct Cursor {
    run: ERun,
    buf: Vec<u8>,
    at: usize,
    in_buf: usize,
    next: usize,
    chunk: usize,
}

impl Cursor {
    fn head(&self, entry: usize) -> &[u8] {
        &self.buf[self.at * entry..(self.at + 1) * entry]
    }

    fn load(&mut self, entry: usize, read: &mut u64) -> Result<bool, ENNError> {
        let rows = self.run.meta.rows as usize;
        if self.next >= rows {
            return Ok(false);
        }
        let hi = (self.next + self.chunk).min(rows);
        self.run.read(self.next, hi, entry, &mut self.buf)?;
        *read += self.buf.len() as u64;
        (self.in_buf, self.next, self.at) = (hi - self.next, hi, 0);
        Ok(true)
    }

    fn advance(&mut self, entry: usize, read: &mut u64) -> Result<bool, ENNError> {
        self.at += 1;
        match self.at < self.in_buf {
            true => Ok(true),
            false => self.load(entry, read),
        }
    }
}

pub(crate) fn merge_runs(inputs: Vec<ERun>, w: &mut ERunWriter, entry: usize) -> Result<u64, ENNError> {
    let chunk = (((4usize << 20) / inputs.len().max(1)).clamp(16 << 10, 256 << 10) / entry).max(1);
    let mut read = 0u64;
    let mut cur: Vec<Cursor> = Vec::with_capacity(inputs.len());
    let mut heap = BinaryHeap::new();
    for run in inputs {
        let mut c = Cursor { run, buf: Vec::new(), at: 0, in_buf: 0, next: 0, chunk };
        if c.load(entry, &mut read)? {
            let e = c.head(entry);
            heap.push(Reverse((entry_hash(e), entry_id(e), cur.len())));
        }
        cur.push(c);
    }
    while let Some(Reverse((_, _, s))) = heap.pop() {
        w.push(cur[s].head(entry))?;
        if cur[s].advance(entry, &mut read)? {
            let e = cur[s].head(entry);
            heap.push(Reverse((entry_hash(e), entry_id(e), s)));
        }
    }
    Ok(read)
}
