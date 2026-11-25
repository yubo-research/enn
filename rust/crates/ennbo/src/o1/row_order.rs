use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::thread::JoinHandle;

use ennbo_bpann::{bpann_read_json, bpann_write_json};
use memmap2::{Mmap, MmapMut};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

use super::forest_tree::Tree;
use super::row_codes::{CodeWriter, RowCodes};
use super::lsh::io_error;
use crate::backend::MmapColumnStore;
use crate::error::ENNError;

pub(crate) const ROW_ORDER_FILE: &str = "o1f_rows.json";
const PREFIX: &str = "o1f_xo_";
const CHUNK: usize = 256;
const UNSET: u32 = u32::MAX;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrderParams {
    pub rows_per_insert: usize,
    pub grow: f64,
    pub min_rows: usize,
}

impl Default for OrderParams {
    fn default() -> Self {
        Self { rows_per_insert: 10, grow: 1.3, min_rows: 4096 }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Serialize, Deserialize)]
struct OrderState {
    generation: u64,
    rows: usize,
}

pub(crate) struct RowOrder {
    copy: MmapColumnStore,
    map: Mmap,
    codes: Option<RowCodes>,
    rows: usize,
    generation: u64,
}

pub(crate) struct OrderBuild {
    generation: u64,
    order: Mmap,
    map: MmapMut,
    copy: MmapColumnStore,
    codes: CodeWriter,
    done: usize,
    rows: usize,
    flusher: Option<JoinHandle<std::io::Result<()>>>,
}

#[derive(Clone, Copy)]
pub(crate) struct Rows<'a> {
    pub x: &'a MmapColumnStore,
    pub order: Option<&'a RowOrder>,
}

fn word(bytes: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap_or([0xff; 4]))
}

fn file_path(dir: &Path, generation: u64, what: &str) -> PathBuf {
    dir.join(format!("{PREFIX}{what}_{generation}.bin"))
}

fn open_rw(path: &Path, create: bool) -> Result<File, ENNError> {
    OpenOptions::new().read(true).write(true).create(create).truncate(create).open(path).map_err(io_error)
}

fn remove_if_present(path: &Path) -> Result<(), ENNError> {
    match std::fs::remove_file(path) {
        Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(io_error(e)),
        _ => Ok(()),
    }
}

pub(crate) fn remove_other_generations(dir: &Path, keep: Option<u64>) -> Result<(), ENNError> {
    let kept: Vec<String> = keep.map_or_else(Vec::new, |g| ["copy", "map", "code"].iter().map(|w| format!("{PREFIX}{w}_{g}.bin")).collect());
    for entry in std::fs::read_dir(dir).map_err(io_error)? {
        let name = entry.map_err(io_error)?.file_name().to_string_lossy().into_owned();
        if name.starts_with(PREFIX) && !kept.contains(&name) {
            remove_if_present(&dir.join(name))?;
        }
    }
    Ok(())
}

impl RowOrder {
    pub(crate) fn open(dir: &Path, ncols: usize, nrows: usize) -> Result<Option<Self>, ENNError> {
        let state: Option<OrderState> = bpann_read_json(&dir.join(ROW_ORDER_FILE)).ok().flatten();
        let opened = state.filter(|s| s.rows <= nrows).map(|s| Self::open_files(dir, s, ncols));
        let order = match opened {
            Some(Ok(o)) => Some(o),
            _ => None,
        };
        remove_other_generations(dir, order.as_ref().map(|o| o.generation))?;
        Ok(order)
    }

    fn open_files(dir: &Path, s: OrderState, ncols: usize) -> Result<Self, ENNError> {
        let copy = MmapColumnStore::mmap_open_or_create(file_path(dir, s.generation, "copy"), ncols, Some(s.rows))?;
        let map = unsafe { Mmap::map(&open_rw(&file_path(dir, s.generation, "map"), false)?) }.map_err(io_error)?;
        if map.len() != 4 * s.rows {
            return Err(io_error("row order map has the wrong length"));
        }
        let codes = RowCodes::open(&file_path(dir, s.generation, "code"), ncols, s.rows);
        Ok(Self { copy, map, codes, rows: s.rows, generation: s.generation })
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn generation(&self) -> u64 {
        self.generation
    }

    fn slot(&self, id: usize) -> usize {
        word(&self.map, id) as usize
    }
}

impl<'a> Rows<'a> {
    pub(crate) fn plain(x: &'a MmapColumnStore) -> Self {
        Self { x, order: None }
    }

    pub(crate) fn code_dist2(&self, i: u32, query: &[f32], weights: &[f32]) -> Option<f64> {
        let o = self.order.filter(|o| (i as usize) < o.rows)?;
        o.codes.as_ref().map(|c| c.dist2(o.slot(i as usize), query, weights))
    }

    fn locate(&self, i: usize) -> (&'a MmapColumnStore, usize) {
        match self.order {
            Some(o) if i < o.rows => (&o.copy, o.slot(i)),
            _ => (self.x, i),
        }
    }

    pub(crate) fn mapped(&self, i: usize) -> Result<&'a [f64], ENNError> {
        let (store, at) = self.locate(i);
        store.mmap_row_slice(at)
    }

    pub(crate) fn pread(&self, i: usize) -> Result<Vec<f64>, ENNError> {
        let (store, at) = self.locate(i);
        store.pread_row(at)
    }

    pub(crate) fn position(&self, i: u32) -> u64 {
        match self.order {
            Some(o) if (i as usize) < o.rows => o.slot(i as usize) as u64,
            _ => (1u64 << 32) + u64::from(i),
        }
    }

    pub(crate) fn covers(&self, i: u32) -> bool {
        self.order.is_some_and(|o| (i as usize) < o.rows)
    }

    pub(crate) fn nrows(&self) -> usize {
        self.x.nrows
    }

    pub(crate) fn row_bytes(&self) -> usize {
        self.x.row_bytes()
    }
}

impl OrderBuild {
    pub(crate) fn start(dir: &Path, generation: u64, trees: &[&Tree], rows: usize, ncols: usize) -> Result<Option<Self>, ENNError> {
        let order_path = file_path(dir, generation, "order");
        let mut out = BufWriter::new(open_rw(&order_path, true)?);
        let (mut count, mut failed) = (0usize, None);
        for tree in trees {
            tree.walk(|leaf| {
                for &id in leaf.ids {
                    count += 1;
                    if let Err(e) = out.write_all(&id.to_le_bytes()) {
                        failed = Some(e);
                    }
                }
            })?;
        }
        out.flush().map_err(io_error)?;
        drop(out);
        if let Some(e) = failed {
            return Err(io_error(e));
        }
        if count != rows || rows == 0 {
            remove_if_present(&order_path)?;
            return Ok(None);
        }
        let order = unsafe { Mmap::map(&open_rw(&order_path, false)?) }.map_err(io_error)?;
        let map_file = open_rw(&file_path(dir, generation, "map"), true)?;
        map_file.set_len(4 * rows as u64).map_err(io_error)?;
        let mut map = unsafe { MmapMut::map_mut(&map_file) }.map_err(io_error)?;
        map.fill(0xff);
        let copy_path = file_path(dir, generation, "copy");
        remove_if_present(&copy_path)?;
        let copy = MmapColumnStore::mmap_open_or_create(copy_path, ncols, Some(0))?;
        let codes = CodeWriter::create(&file_path(dir, generation, "code"), ncols)?;
        Ok(Some(Self { generation, order, map, copy, codes, done: 0, rows, flusher: None }))
    }

    pub(crate) fn step(&mut self, src: Rows<'_>, budget: usize) -> Result<usize, ENNError> {
        let end = self.done.saturating_add(budget).min(self.rows);
        let ncols = self.copy.ncols;
        let mut at = self.done;
        while at < end {
            let stop = (at + CHUNK).min(end);
            let mut buf = Array2::zeros((stop - at, ncols));
            for (r, k) in (at..stop).enumerate() {
                let id = word(&self.order, k) as usize;
                if id >= self.rows || word(&self.map, id) != UNSET {
                    return Err(io_error("tree order lists a row twice or out of range"));
                }
                let row = src.mapped(id)?;
                self.codes.push(row)?;
                buf.row_mut(r).assign(&ndarray::ArrayView1::from(row));
                self.map[4 * id..4 * id + 4].copy_from_slice(&(k as u32).to_le_bytes());
            }
            self.copy.mmap_append(&buf.view())?;
            at = stop;
        }
        let moved = end - self.done;
        self.done = end;
        if self.flusher.as_ref().is_none_or(|f| f.is_finished()) {
            self.join_flusher()?;
            let file = self.copy.file_handle()?;
            self.flusher = Some(std::thread::spawn(move || file.sync_data()));
        }
        Ok(moved)
    }

    pub(crate) fn finished(&self) -> bool {
        self.done == self.rows
    }

    fn join_flusher(&mut self) -> Result<(), ENNError> {
        match self.flusher.take().map(JoinHandle::join) {
            Some(Ok(Err(e))) => Err(io_error(e)),
            Some(Err(_)) => Err(io_error("row order flusher panicked")),
            _ => Ok(()),
        }
    }

    pub(crate) fn finish(mut self, dir: &Path) -> Result<RowOrder, ENNError> {
        self.join_flusher()?;
        self.copy.flush()?;
        self.map.flush().map_err(io_error)?;
        self.codes.finish()?;
        bpann_write_json(&dir.join(ROW_ORDER_FILE), &OrderState { generation: self.generation, rows: self.rows }).map_err(io_error)?;
        let map = self.map.make_read_only().map_err(io_error)?;
        drop(self.order);
        remove_other_generations(dir, Some(self.generation))?;
        let codes = RowCodes::open(&file_path(dir, self.generation, "code"), self.copy.ncols, self.rows);
        Ok(RowOrder { copy: self.copy, map, codes, rows: self.rows, generation: self.generation })
    }
}
