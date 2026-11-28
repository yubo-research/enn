use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use super::lsh::io_error;
use crate::error::ENNError;

pub(crate) type NearRows = HashMap<u32, Vec<f64>>;

const PAGE: usize = 4096;

pub(crate) struct LeafRows {
    path: PathBuf,
    file: File,
    slots: File,
    cap: usize,
    dim: usize,
    next: u32,
    free: Vec<u32>,
}

pub(crate) fn clustered() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ENN_CLUSTER_LEAVES").is_some())
}

pub(crate) fn lrows_path(tree: &Path) -> PathBuf {
    tree.with_extension("lrows")
}

fn slots_path(rows: &Path) -> PathBuf {
    rows.with_extension("lslot")
}

fn word(buf: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(buf[4 * i..4 * i + 4].try_into().unwrap_or([0; 4]))
}

fn open_rw(path: &Path, fresh: bool) -> Result<File, ENNError> {
    OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(fresh)
        .open(path)
        .map_err(io_error)
}

impl LeafRows {
    pub(crate) fn drop_cached(&self) {
        super::hygiene::drop_cached(&self.file);
    }

    pub(crate) fn start_writeback(&self) {
        super::hygiene::start_writeback(&self.file);
    }

    pub(crate) fn open(tree: &Path, cap: usize, dim: usize, fresh: bool) -> Result<Self, ENNError> {
        let path = lrows_path(tree);
        let file = open_rw(&path, fresh)?;
        let slots = open_rw(&slots_path(&path), fresh)?;
        let mut rows = Self {
            path,
            file,
            slots,
            cap,
            dim,
            next: 0,
            free: Vec::new(),
        };
        let len = rows.file.metadata().map_err(io_error)?.len();
        rows.next = len.div_ceil(rows.stride() as u64) as u32;
        Ok(rows)
    }

    fn rec(&self) -> usize {
        4 + 8 * self.dim
    }

    pub(crate) fn stride(&self) -> usize {
        (self.cap * self.rec()).div_ceil(PAGE) * PAGE
    }

    fn region(&self, slot: u32) -> u64 {
        u64::from(slot) * self.stride() as u64
    }

    fn slot(&self, page: u32) -> Result<Option<u32>, ENNError> {
        let mut b = [0u8; 4];
        let got = self
            .slots
            .read_at(&mut b, 4 * u64::from(page))
            .map_err(io_error)?;
        Ok(u32::from_le_bytes(b).checked_sub(1).filter(|_| got == 4))
    }

    fn offset(&mut self, page: u32) -> Result<u64, ENNError> {
        if let Some(s) = self.slot(page)? {
            return Ok(self.region(s));
        }
        let s = self.free.pop().unwrap_or_else(|| {
            self.next += 1;
            self.next - 1
        });
        self.slots
            .write_all_at(&(s + 1).to_le_bytes(), 4 * u64::from(page))
            .map_err(io_error)?;
        Ok(self.region(s))
    }

    #[cfg(test)]
    pub(crate) fn regions(&self) -> u32 {
        self.next - self.free.len() as u32
    }

    pub(crate) fn fits(&self, dim: usize) -> bool {
        dim == self.dim && dim > 0
    }

    pub(crate) fn moved(&mut self, tree: &Path) -> Result<(), ENNError> {
        let to = lrows_path(tree);
        if to != self.path {
            std::fs::rename(&self.path, &to).map_err(io_error)?;
            std::fs::rename(slots_path(&self.path), slots_path(&to)).map_err(io_error)?;
            self.path = to;
        }
        Ok(())
    }

    pub(crate) fn repack(&mut self, map: &[u32]) -> Result<(), ENNError> {
        let len = self.slots.metadata().map_err(io_error)?.len() as usize / 4;
        let mut old = vec![0u8; 4 * len];
        self.slots.read_exact_at(&mut old, 0).map_err(io_error)?;
        let top = map.iter().copied().max().unwrap_or(0) as usize + 1;
        let mut new = vec![0u8; 4 * top];
        let mut used = vec![false; self.next as usize];
        for (page, &p) in map.iter().enumerate().take(len) {
            let tag = word(&old, page);
            if tag == 0 || p == 0 {
                continue;
            }
            new[4 * p as usize..4 * p as usize + 4].copy_from_slice(&tag.to_le_bytes());
            if let Some(u) = used.get_mut(tag as usize - 1) {
                *u = true;
            }
        }
        self.free = (0..self.next).rev().filter(|&s| !used[s as usize]).collect();
        let tmp = slots_path(&self.path).with_extension("lscmp");
        open_rw(&tmp, true)?.write_all_at(&new, 0).map_err(io_error)?;
        std::fs::rename(&tmp, slots_path(&self.path)).map_err(io_error)?;
        self.slots = open_rw(&slots_path(&self.path), false)?;
        Ok(())
    }

    fn encode(&self, id: u32, row: &[f64], buf: &mut [u8]) {
        buf[..4].copy_from_slice(&(id + 1).to_le_bytes());
        for (i, v) in row.iter().enumerate() {
            buf[4 + 8 * i..12 + 8 * i].copy_from_slice(&v.to_le_bytes());
        }
    }

    pub(crate) fn put_all<'r>(
        &mut self,
        page: u32,
        rows: impl Iterator<Item = (u32, &'r [f64])>,
    ) -> Result<(), ENNError> {
        let mut buf = vec![0u8; self.stride()];
        let rec = self.rec();
        for (n, (id, row)) in rows.take(self.cap).enumerate() {
            if row.len() != self.dim {
                break;
            }
            self.encode(id, row, &mut buf[n * rec..(n + 1) * rec]);
        }
        let at = self.offset(page)?;
        super::io_streams::wrote(super::io_streams::Stream::LeafRows, at, buf.len());
        self.file.write_all_at(&buf, at).map_err(io_error)
    }

    pub(crate) fn get(
        &self,
        page: u32,
        want: Option<&[u32]>,
        out: &mut NearRows,
    ) -> Result<usize, ENNError> {
        let Some(slot) = self.slot(page)? else {
            return Ok(0);
        };
        let mut buf = vec![0u8; self.cap * self.rec()];
        let t0 = std::time::Instant::now();
        let got = self.file.read_at(&mut buf, self.region(slot)).map_err(io_error)?;
        super::io_streams::read(super::io_streams::Stream::LeafRows, self.region(slot), got, t0);
        let mut hits = 0;
        for rec in buf[..got - got % self.rec()].chunks_exact(self.rec()) {
            let tag = u32::from_le_bytes([rec[0], rec[1], rec[2], rec[3]]);
            let Some(id) = tag.checked_sub(1) else {
                continue;
            };
            if want.is_some_and(|w| !w.contains(&id)) || out.contains_key(&id) {
                continue;
            }
            out.insert(
                id,
                rec[4..]
                    .chunks_exact(8)
                    .map(|b| f64::from_le_bytes(b.try_into().unwrap_or([0; 8])))
                    .collect(),
            );
            hits += 1;
        }
        Ok(hits)
    }
}
