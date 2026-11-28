use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

use super::hlsm_hash::{code_bytes, CodeBook, TableFrame, TableHash};
use super::hlsm_run::{merge_runs, remove_run, write_frame, ERun, ERunMeta, ERunWriter, ENTRY_HEAD};
use super::lsh::io_error;
use crate::backend::MmapColumnStore;
use crate::error::ENNError;

pub(crate) const E_META: &str = "o1e_meta.json";
const E_BOOKS: &str = "o1e_books.bin";

pub(crate) fn selected(work_dir: &Path) -> bool {
    work_dir.join(E_META).exists() || std::env::var("ENN_O1_KIND").is_ok_and(|v| v.eq_ignore_ascii_case("hlsm"))
}

pub(crate) fn env_f64(name: &str) -> Option<f64> {
    std::env::var(name).ok().and_then(|v| v.parse::<f64>().ok())
}

fn env_usize(name: &str) -> Option<usize> {
    env_f64(name).map(|v| v as usize)
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct EStats {
    pub flushes: u64,
    pub merges: u64,
    pub merged_rows: u64,
    pub max_merge_rows: u64,
    pub rebuilds: u64,
    pub rehashes: u64,
    pub bytes_written: u64,
    pub bytes_read: u64,
    #[serde(default)]
    pub row_bytes_read: u64,
    pub flush_ns: u64,
    pub merge_ns: u64,
    pub rebuild_ns: u64,
    pub max_rebuild_ns: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct EMeta {
    pub d: usize,
    pub k: usize,
    pub l0_rows: usize,
    #[serde(default)]
    pub book_rows: usize,
    pub ratio: usize,
    pub seed: u64,
    pub flushed: u64,
    pub next_id: u64,
    pub books: usize,
    pub target: Vec<f64>,
    pub frames: Vec<Option<TableFrame>>,
    pub runs: Vec<ERunMeta>,
    pub stats: EStats,
}

pub(crate) struct HlsmIndex {
    pub(crate) dir: PathBuf,
    pub(crate) meta: EMeta,
    pub(crate) hashes: Vec<TableHash>,
    pub(crate) books: Vec<CodeBook>,
    pub(crate) runs: Vec<ERun>,
    pub(crate) entry: usize,
    pub(crate) window: usize,
    pub(crate) rerank: usize,
    pub(crate) io: usize,
    pub(crate) n: usize,
}

const MIN_BOOK_ROWS: usize = 4096;

fn new_meta(d: usize, seed: u64, scale: &[f64]) -> EMeta {
    let l0 = env_usize("ENN_E_L0").unwrap_or((1usize << 20) / (8 * d.max(1))).max(64);
    let tables = env_usize("ENN_E_L").unwrap_or(8).max(1);
    EMeta {
        d,
        k: env_usize("ENN_E_K").unwrap_or(4).clamp(1, d.clamp(1, 64)),
        l0_rows: l0,
        book_rows: l0 * MIN_BOOK_ROWS.div_ceil(l0),
        ratio: env_usize("ENN_E_T").unwrap_or(4).max(2),
        seed,
        flushed: 0,
        next_id: 0,
        books: 0,
        target: scale.to_vec(),
        frames: vec![None; tables],
        runs: Vec::new(),
        stats: EStats::default(),
    }
}

impl HlsmIndex {
    pub(crate) fn open(work_dir: &Path, seed: u64, fallback_scale: &[f64], rows: &MmapColumnStore) -> Result<Self, ENNError> {
        let d = rows.ncols;
        let stored: Option<EMeta> = ennbo_bpann::bpann_read_json(&work_dir.join(E_META)).map_err(io_error)?;
        let meta = stored.unwrap_or_else(|| new_meta(d, seed, fallback_scale));
        let hashes = (0..meta.frames.len()).map(|t| TableHash::new(d, meta.k, meta.seed.wrapping_add(7919 * t as u64 + 1))).collect();
        let books = read_books(&work_dir.join(E_BOOKS), d, meta.books)?;
        let runs = meta.runs.iter().map(|r| ERun::open(work_dir, r.clone())).collect::<Result<_, _>>()?;
        let entry = ENTRY_HEAD + code_bytes(d);
        let mut index = Self {
            dir: work_dir.to_path_buf(),
            meta,
            hashes,
            books,
            runs,
            entry,
            window: (env_usize("ENN_E_WIN").unwrap_or(32 << 10) / entry).max(1),
            rerank: 100,
            io: env_usize("ENN_E_IO").unwrap_or(8).max(1),
            n: 0,
        };
        index.save()?;
        index.insert_rows(rows, 0)?;
        Ok(index)
    }

    pub(crate) fn save(&self) -> Result<(), ENNError> {
        ennbo_bpann::bpann_write_json(&self.dir.join(E_META), &self.meta).map_err(io_error)
    }

    pub(crate) fn hash_scale(&self) -> &[f64] {
        &self.meta.target
    }

    pub(crate) fn rehash(&mut self, scale: &[f64]) -> Result<(), ENNError> {
        if scale != self.meta.target.as_slice() {
            self.meta.target = scale.to_vec();
            self.meta.stats.rehashes += 1;
            self.save()?;
        }
        Ok(())
    }

    pub(crate) fn book_of(&self, id: u32) -> usize {
        (id as usize / self.meta.book_rows.max(self.meta.l0_rows)).min(self.books.len().saturating_sub(1))
    }

    pub(crate) fn tables(&self) -> usize {
        self.meta.frames.len()
    }

    pub(crate) fn insert_rows(&mut self, rows: &MmapColumnStore, _start: usize) -> Result<(), ENNError> {
        self.n = rows.nrows;
        while rows.nrows as u64 - self.meta.flushed >= self.meta.l0_rows as u64 {
            self.flush(rows)?;
        }
        if let Some(t) = self.stale_table() {
            self.rebuild_table(t, rows)?;
        }
        Ok(())
    }

    fn stale_table(&self) -> Option<usize> {
        if self.meta.flushed == 0 {
            return None;
        }
        self.meta.frames.iter().position(|f| f.as_ref().is_some_and(|f| f.scale != self.meta.target))
    }

    pub(crate) fn take_id(&mut self) -> u64 {
        self.meta.next_id += 1;
        self.meta.next_id - 1
    }

    fn flush(&mut self, rows: &MmapColumnStore) -> Result<(), ENNError> {
        let t0 = Instant::now();
        let (lo, hi) = (self.meta.flushed as usize, self.meta.flushed as usize + self.meta.l0_rows);
        let x: Vec<f64> = rows.mmap_row_range(lo, hi)?.iter().copied().collect();
        if self.books.is_empty() || lo.is_multiple_of(self.meta.book_rows.max(self.meta.l0_rows)) {
            self.push_book(CodeBook::fit(&x, self.meta.d))?;
        }
        let ids: Vec<u32> = (lo as u32..hi as u32).collect();
        let codes = self.encode(&ids, &x);
        for t in 0..self.tables() {
            let frame = self.hashes[t].fit(&x, &self.meta.target);
            self.meta.frames[t].get_or_insert_with(|| frame.clone());
            let entries = self.entries(&frame, t, &ids, &x, &codes);
            let id = self.take_id();
            let meta = self.write_run(id, (t, 0), &entries, &frame, (lo as u64, hi as u64))?;
            self.runs.push(ERun::open(&self.dir, meta.clone())?);
            self.meta.runs.push(meta);
        }
        self.meta.flushed = hi as u64;
        self.meta.stats.flushes += 1;
        self.meta.stats.flush_ns += t0.elapsed().as_nanos() as u64;
        self.compact_all(rows)?;
        self.save()
    }

    fn push_book(&mut self, book: CodeBook) -> Result<(), ENNError> {
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(self.dir.join(E_BOOKS)).map_err(io_error)?;
        f.write_all(&book.to_bytes()).map_err(io_error)?;
        f.sync_data().map_err(io_error)?;
        self.books.push(book);
        self.meta.books = self.books.len();
        Ok(())
    }

    pub(crate) fn encode(&self, ids: &[u32], x: &[f64]) -> Vec<u8> {
        let (d, cb) = (self.meta.d, code_bytes(self.meta.d));
        let mut out = vec![0u8; ids.len() * cb];
        for (i, &id) in ids.iter().enumerate() {
            self.books[self.book_of(id)].encode(&x[i * d..(i + 1) * d], &mut out[i * cb..(i + 1) * cb]);
        }
        out
    }

    pub(crate) fn entries(&self, frame: &TableFrame, t: usize, ids: &[u32], x: &[f64], codes: &[u8]) -> Vec<u8> {
        let (d, cb, e) = (self.meta.d, code_bytes(self.meta.d), self.entry);
        let mut z = vec![0.0; self.meta.k];
        let mut keyed: Vec<(u64, u32, usize)> = (0..ids.len()).map(|i| (self.hashes[t].key(&x[i * d..(i + 1) * d], frame, &mut z), ids[i], i)).collect();
        keyed.sort_unstable();
        let mut out = Vec::with_capacity(ids.len() * e);
        for (h, id, i) in keyed {
            out.extend_from_slice(&h.to_le_bytes());
            out.extend_from_slice(&id.to_le_bytes());
            out.extend_from_slice(&codes[i * cb..(i + 1) * cb]);
        }
        out
    }

    pub(crate) fn write_run(&mut self, id: u64, (table, level): (usize, u32), entries: &[u8], frame: &TableFrame, (lo, hi): (u64, u64)) -> Result<ERunMeta, ENNError> {
        let mut w = ERunWriter::create(&self.dir, id, self.entry)?;
        entries.chunks_exact(self.entry).try_for_each(|e| w.push(e))?;
        let (rows, bytes) = w.finish()?;
        self.meta.stats.bytes_written += bytes + write_frame(&self.dir, id, frame)?;
        Ok(ERunMeta { id, table, level, rows, lo, hi })
    }

    pub(crate) fn full_level(&self, t: usize) -> Option<u32> {
        let mine = || self.meta.runs.iter().filter(|r| r.table == t);
        let top = mine().map(|r| r.level).max()?;
        (0..=top).find(|&l| mine().filter(|r| r.level == l).count() >= self.meta.ratio)
    }

    pub(crate) fn take_level(&mut self, t: usize, level: u32) -> Vec<ERun> {
        let pick = |r: &ERunMeta| r.table == t && r.level == level;
        let (inputs, keep): (Vec<ERun>, Vec<ERun>) = std::mem::take(&mut self.runs).into_iter().partition(|r| pick(&r.meta));
        self.runs = keep;
        self.meta.runs.retain(|r| !pick(r));
        inputs
    }

    pub(crate) fn install(&mut self, metas: Vec<ERunMeta>, inputs: &[ERunMeta], t0: Instant) -> Result<(), ENNError> {
        let rows: u64 = inputs.iter().map(|r| r.rows).sum::<u64>() / metas.len().max(1) as u64;
        for meta in metas {
            self.runs.push(ERun::open(&self.dir, meta.clone())?);
            self.meta.runs.push(meta);
            let st = &mut self.meta.stats;
            st.merges += 1;
            st.merged_rows += rows;
            st.max_merge_rows = st.max_merge_rows.max(rows);
        }
        self.meta.stats.merge_ns += t0.elapsed().as_nanos() as u64;
        self.save()?;
        inputs.iter().for_each(|r| remove_run(&self.dir, r.id));
        Ok(())
    }

    pub(crate) fn merge_into(&mut self, inputs: Vec<ERun>, t: usize, level: u32) -> Result<ERunMeta, ENNError> {
        let id = self.take_id();
        let frame = inputs.first().map(|r| r.frame.clone()).ok_or_else(|| io_error("O1E merge with no input runs"))?;
        let (lo, hi) = (inputs.iter().map(|r| r.meta.lo).min().unwrap_or(0), inputs.iter().map(|r| r.meta.hi).max().unwrap_or(0));
        let mut w = ERunWriter::create(&self.dir, id, self.entry)?;
        let read = merge_runs(inputs, &mut w, self.entry)?;
        let (rows, bytes) = w.finish()?;
        self.meta.stats.bytes_read += read;
        self.meta.stats.bytes_written += bytes + write_frame(&self.dir, id, &frame)?;
        Ok(ERunMeta { id, table: t, level, rows, lo, hi })
    }

    pub(crate) fn memory_bytes(&self) -> usize {
        let fences: usize = self.runs.iter().map(ERun::fence_bytes).sum();
        std::mem::size_of::<Self>() + fences + self.books.len() * 88 * self.meta.d + self.runs.len() * 128
    }

    pub(crate) fn set_budget(&mut self, _max_leaves: usize, max_candidates: usize) {
        self.rerank = max_candidates.max(1);
    }
}

fn read_books(path: &Path, d: usize, count: usize) -> Result<Vec<CodeBook>, ENNError> {
    if count == 0 {
        return Ok(Vec::new());
    }
    let b = std::fs::read(path).map_err(io_error)?;
    let rec = 88 * d;
    assert!(b.len() >= count * rec, "o1e_books.bin is shorter than its meta says");
    Ok(b.chunks_exact(rec).take(count).map(|c| CodeBook::from_bytes(c, d)).collect())
}
