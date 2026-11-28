use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use super::lsh::io_error;
use crate::error::ENNError;

#[derive(Debug, Clone, Default, PartialEq)]
pub(crate) struct Items {
    pub ids: Vec<u32>,
    pub x: Vec<f64>,
}

impl Items {
    pub(crate) fn len(&self) -> usize {
        self.ids.len()
    }

    pub(crate) fn row(&self, i: usize, d: usize) -> &[f64] {
        &self.x[i * d..(i + 1) * d]
    }

    pub(crate) fn push(&mut self, id: u32, v: &[f64]) {
        self.ids.push(id);
        self.x.extend_from_slice(v);
    }

    pub(crate) fn upsert(&mut self, id: u32, v: &[f64], d: usize) {
        match self.ids.iter().position(|&c| c == id) {
            Some(i) => self.x[i * d..(i + 1) * d].copy_from_slice(v),
            None => self.push(id, v),
        }
    }

    pub(crate) fn mean(&self, d: usize) -> Vec<f64> {
        let mut c = vec![0.0; d];
        for r in self.x.chunks_exact(d) {
            c.iter_mut().zip(r).for_each(|(a, b)| *a += b);
        }
        let n = self.len().max(1) as f64;
        c.iter_mut().for_each(|a| *a /= n);
        c
    }
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Geo {
    pub d: usize,
    pub cap: usize,
    pub fan: usize,
    pub m: usize,
}

impl Geo {
    pub(crate) fn item_bytes(&self) -> usize {
        4 + 8 * self.d
    }
    pub(crate) fn leaf_bytes(&self) -> usize {
        (8 + self.cap * self.item_bytes()).div_ceil(4096) * 4096
    }
    pub(crate) fn up_bytes(&self) -> usize {
        8 + (self.fan + 1) * self.item_bytes()
    }
    fn link_bytes(&self) -> usize {
        4 * (1 + self.m)
    }
}

fn encode(it: &Items, d: usize, size: usize) -> Vec<u8> {
    let mut buf = vec![0u8; size];
    buf[..4].copy_from_slice(&(it.len() as u32).to_le_bytes());
    let mut at = 8;
    for (i, &id) in it.ids.iter().enumerate() {
        buf[at..at + 4].copy_from_slice(&id.to_le_bytes());
        at += 4;
        for v in it.row(i, d) {
            buf[at..at + 8].copy_from_slice(&v.to_le_bytes());
            at += 8;
        }
    }
    buf
}

pub(crate) fn decode(buf: &[u8], d: usize) -> Items {
    let n = u32::from_le_bytes(buf[..4].try_into().unwrap_or_default()) as usize;
    let mut it = Items { ids: Vec::with_capacity(n), x: Vec::with_capacity(n * d) };
    let mut at = 8;
    for _ in 0..n {
        it.ids.push(u32::from_le_bytes(buf[at..at + 4].try_into().unwrap_or_default()));
        at += 4;
        for _ in 0..d {
            it.x.push(f64::from_le_bytes(buf[at..at + 8].try_into().unwrap_or_default()));
            at += 8;
        }
    }
    it
}

pub(crate) fn open_rw(path: &Path) -> Result<File, ENNError> {
    OpenOptions::new().read(true).write(true).create(true).truncate(false).open(path).map_err(io_error)
}

pub(crate) fn read_at(f: &File, out: &mut [u8], id: u64) -> Result<(), ENNError> {
    let off = id * out.len() as u64;
    out.fill(0);
    let mut got = 0;
    while got < out.len() {
        let k = f.read_at(&mut out[got..], off + got as u64).map_err(io_error)?;
        if k == 0 {
            break;
        }
        got += k;
    }
    Ok(())
}

#[derive(Debug, Default)]
pub(crate) struct Counters {
    pub leaf_reads: AtomicU64,
    pub leaf_writes: AtomicU64,
    pub side_reads: AtomicU64,
    pub side_writes: AtomicU64,
}

fn bump(c: &AtomicU64) {
    c.fetch_add(1, Ordering::Relaxed);
}

pub(crate) struct GFiles {
    pub geo: Geo,
    dir: PathBuf,
    leaf: File,
    cent: File,
    link: File,
    up: Vec<File>,
    par: Vec<File>,
    pub ctr: Counters,
}

impl GFiles {
    pub(crate) fn open(dir: &Path, geo: Geo, height: usize) -> Result<Self, ENNError> {
        let mut f = Self {
            geo,
            dir: dir.to_path_buf(),
            leaf: open_rw(&dir.join("o1g_leaf.bin"))?,
            cent: open_rw(&dir.join("o1g_cent.bin"))?,
            link: open_rw(&dir.join("o1g_link.bin"))?,
            up: Vec::new(),
            par: Vec::new(),
            ctr: Counters::default(),
        };
        f.ensure_levels(height)?;
        Ok(f)
    }

    pub(crate) fn ensure_levels(&mut self, height: usize) -> Result<(), ENNError> {
        while self.up.len() + 1 < height {
            let l = self.up.len() + 1;
            self.up.push(open_rw(&self.dir.join(format!("o1g_up{l}.bin")))?);
        }
        while self.par.len() + 1 < height {
            let l = self.par.len();
            self.par.push(open_rw(&self.dir.join(format!("o1g_par{l}.bin")))?);
        }
        Ok(())
    }

    pub(crate) fn read_leaf(&self, id: u32) -> Result<Items, ENNError> {
        let mut buf = vec![0u8; self.geo.leaf_bytes()];
        read_at(&self.leaf, &mut buf, id as u64)?;
        bump(&self.ctr.leaf_reads);
        Ok(decode(&buf, self.geo.d))
    }

    pub(crate) fn write_leaf(&self, id: u32, it: &Items) -> Result<(), ENNError> {
        let size = self.geo.leaf_bytes();
        self.leaf.write_all_at(&encode(it, self.geo.d, size), id as u64 * size as u64).map_err(io_error)?;
        bump(&self.ctr.leaf_writes);
        Ok(())
    }

    pub(crate) fn read_up(&self, level: usize, id: u32) -> Result<Items, ENNError> {
        let mut buf = vec![0u8; self.geo.up_bytes()];
        read_at(&self.up[level - 1], &mut buf, id as u64)?;
        bump(&self.ctr.side_reads);
        Ok(decode(&buf, self.geo.d))
    }

    pub(crate) fn write_up(&self, level: usize, id: u32, it: &Items) -> Result<(), ENNError> {
        let size = self.geo.up_bytes();
        self.up[level - 1].write_all_at(&encode(it, self.geo.d, size), id as u64 * size as u64).map_err(io_error)?;
        bump(&self.ctr.side_writes);
        Ok(())
    }

    pub(crate) fn read_cent(&self, id: u32) -> Result<Vec<f64>, ENNError> {
        let mut buf = vec![0u8; 8 * self.geo.d];
        read_at(&self.cent, &mut buf, id as u64)?;
        bump(&self.ctr.side_reads);
        Ok(buf.chunks_exact(8).map(|b| f64::from_le_bytes(b.try_into().unwrap_or_default())).collect())
    }

    pub(crate) fn write_cent(&self, id: u32, c: &[f64]) -> Result<(), ENNError> {
        let buf: Vec<u8> = c.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.cent.write_all_at(&buf, id as u64 * buf.len() as u64).map_err(io_error)?;
        bump(&self.ctr.side_writes);
        Ok(())
    }

    pub(crate) fn read_links(&self, id: u32) -> Result<Vec<u32>, ENNError> {
        let mut buf = vec![0u8; self.geo.link_bytes()];
        read_at(&self.link, &mut buf, id as u64)?;
        bump(&self.ctr.side_reads);
        let w: Vec<u32> = buf.chunks_exact(4).map(|b| u32::from_le_bytes(b.try_into().unwrap_or_default())).collect();
        let n = (w[0] as usize).min(self.geo.m);
        Ok(w[1..1 + n].to_vec())
    }

    pub(crate) fn write_links(&self, id: u32, links: &[u32]) -> Result<(), ENNError> {
        let mut buf = vec![0u8; self.geo.link_bytes()];
        let n = links.len().min(self.geo.m);
        buf[..4].copy_from_slice(&(n as u32).to_le_bytes());
        for (k, l) in links[..n].iter().enumerate() {
            buf[4 + 4 * k..8 + 4 * k].copy_from_slice(&l.to_le_bytes());
        }
        self.link.write_all_at(&buf, id as u64 * buf.len() as u64).map_err(io_error)?;
        bump(&self.ctr.side_writes);
        Ok(())
    }

    pub(crate) fn read_par(&self, level: usize, id: u32) -> Result<u32, ENNError> {
        let mut buf = [0u8; 4];
        read_at(&self.par[level], &mut buf, id as u64)?;
        Ok(u32::from_le_bytes(buf))
    }

    pub(crate) fn write_par(&self, level: usize, id: u32, parent: u32) -> Result<(), ENNError> {
        self.par[level].write_all_at(&parent.to_le_bytes(), id as u64 * 4).map_err(io_error)
    }
}
