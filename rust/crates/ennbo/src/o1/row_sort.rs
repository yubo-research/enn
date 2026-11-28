use std::fs::File;
use std::io::{BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use super::lsh::io_error;
use super::row_codes::CodeWriter;
use super::row_order::Rows;
use crate::backend::MmapColumnStore;
use crate::error::ENNError;

const MIN_BUFFER: usize = 4096;
const UNSET: u32 = u32::MAX;

struct Buckets {
    paths: Vec<PathBuf>,
    out: Vec<BufWriter<File>>,
    rec: usize,
}

impl Buckets {
    fn create(
        dir: &Path,
        name: &str,
        count: usize,
        rec: usize,
        mem: usize,
    ) -> Result<Self, ENNError> {
        let cap = (mem / count.max(1)).max(MIN_BUFFER);
        let paths: Vec<PathBuf> = (0..count)
            .map(|b| dir.join(format!("{name}_{b}.bin")))
            .collect();
        let out = paths
            .iter()
            .map(|p| File::create(p).map(|f| BufWriter::with_capacity(cap, f)))
            .collect::<Result<_, _>>()
            .map_err(io_error)?;
        Ok(Self { paths, out, rec })
    }

    fn push(&mut self, bucket: usize, key: u32, body: &[u8]) -> Result<(), ENNError> {
        let w = &mut self.out[bucket];
        w.write_all(&key.to_le_bytes())
            .and_then(|()| w.write_all(body))
            .map_err(io_error)
    }

    fn close(&mut self) -> Result<(), ENNError> {
        self.out
            .drain(..)
            .try_for_each(|mut w| w.flush())
            .map_err(io_error)
    }

    fn take(&self, bucket: usize) -> Result<Vec<u8>, ENNError> {
        let mut bytes = Vec::new();
        File::open(&self.paths[bucket])
            .and_then(|mut f| f.read_to_end(&mut bytes))
            .map_err(io_error)?;
        std::fs::remove_file(&self.paths[bucket]).map_err(io_error)?;
        if bytes.len() % self.rec != 0 {
            return Err(io_error("row sort bucket has a torn record"));
        }
        Ok(bytes)
    }

    fn remove(&self) {
        self.paths.iter().for_each(|p| {
            let _ = std::fs::remove_file(p);
        });
    }
}

pub(crate) struct Sink<'a> {
    pub order: &'a [u8],
    pub map: &'a mut [u8],
    pub copy: &'a mut MmapColumnStore,
    pub codes: &'a mut CodeWriter,
}

pub(crate) struct OrderSort {
    src: Buckets,
    dst: Buckets,
    width: usize,
    rows: usize,
    ncols: usize,
    stage: u8,
    at: usize,
    credit: usize,
}

fn word(bytes: &[u8], i: usize) -> u32 {
    u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap_or([0xff; 4]))
}

impl OrderSort {
    pub(crate) fn start(
        dir: &Path,
        tag: &str,
        rows: usize,
        ncols: usize,
        mem: usize,
    ) -> Result<Self, ENNError> {
        let row_bytes = 8 * ncols.max(1);
        let width = (mem / row_bytes).max(1);
        let count = rows.div_ceil(width).max(1);
        let src = Buckets::create(dir, &format!("{tag}_s"), count, 8, mem)?;
        let dst = Buckets::create(dir, &format!("{tag}_d"), count, 4 + row_bytes, mem)?;
        Ok(Self {
            src,
            dst,
            width,
            rows,
            ncols,
            stage: 0,
            at: 0,
            credit: 0,
        })
    }

    pub(crate) fn finished(&self) -> bool {
        self.stage == 3
    }

    fn count(&self) -> usize {
        self.src.paths.len()
    }

    fn span(&self, bucket: usize) -> (usize, usize) {
        let lo = bucket * self.width;
        (lo, (lo + self.width).min(self.rows))
    }

    pub(crate) fn step(
        &mut self,
        src: Rows<'_>,
        budget: usize,
        sink: &mut Sink<'_>,
    ) -> Result<usize, ENNError> {
        self.credit = self.credit.saturating_add(budget.saturating_mul(3));
        let mut placed = 0;
        while !self.finished() {
            let need = if self.stage == 0 {
                1
            } else {
                self.span(self.at).1 - self.span(self.at).0
            };
            if self.credit < need {
                break;
            }
            match self.stage {
                0 => self.route(src, sink)?,
                1 => self.fetch(src)?,
                _ => placed += self.place(sink)?,
            }
        }
        Ok(placed)
    }

    fn route(&mut self, src: Rows<'_>, sink: &mut Sink<'_>) -> Result<(), ENNError> {
        let end = self.at.saturating_add(self.credit).min(self.rows);
        for k in self.at..end {
            let id = word(sink.order, k) as usize;
            if id >= self.rows || word(sink.map, id) != UNSET {
                return Err(io_error("tree order lists a row twice or out of range"));
            }
            sink.map[4 * id..4 * id + 4].copy_from_slice(&(k as u32).to_le_bytes());
            let p = src.source(id);
            self.src
                .push(p / self.width, p as u32, &(k as u32).to_le_bytes())?;
        }
        self.credit -= end - self.at;
        self.at = end;
        if self.at == self.rows {
            self.src.close()?;
            (self.stage, self.at) = (1, 0);
        }
        Ok(())
    }

    fn fetch(&mut self, src: Rows<'_>) -> Result<(), ENNError> {
        let (lo, hi) = self.span(self.at);
        let rb = 8 * self.ncols;
        let rows = src.read_sources(lo, hi)?;
        let pairs = self.src.take(self.at)?;
        for pair in pairs.chunks_exact(8) {
            let p = word(pair, 0) as usize - lo;
            let k = word(pair, 1);
            self.dst
                .push(k as usize / self.width, k, &rows[p * rb..(p + 1) * rb])?;
        }
        self.credit -= hi - lo;
        self.at += 1;
        if self.at == self.count() {
            self.dst.close()?;
            (self.stage, self.at) = (2, 0);
        }
        Ok(())
    }

    fn place(&mut self, sink: &mut Sink<'_>) -> Result<usize, ENNError> {
        let (lo, hi) = self.span(self.at);
        let rb = 8 * self.ncols;
        let recs = self.dst.take(self.at)?;
        if recs.len() != (hi - lo) * (4 + rb) {
            return Err(io_error("row sort bucket is missing rows"));
        }
        let mut buf = ndarray::Array2::<f64>::zeros((hi - lo, self.ncols));
        let vals = buf
            .as_slice_mut()
            .ok_or_else(|| io_error("row sort buffer is not contiguous"))?;
        for rec in recs.chunks_exact(4 + rb) {
            let r = word(rec, 0) as usize - lo;
            for (j, c) in rec[4..].chunks_exact(8).enumerate() {
                vals[r * self.ncols + j] = f64::from_ne_bytes(c.try_into().unwrap_or([0; 8]));
            }
        }
        for row in buf.rows() {
            sink.codes.push(row.as_slice().unwrap_or(&[]))?;
        }
        sink.copy.mmap_append(&buf.view())?;
        self.credit -= hi - lo;
        self.at += 1;
        if self.at == self.count() {
            self.stage = 3;
        }
        Ok(hi - lo)
    }
}

impl Drop for OrderSort {
    fn drop(&mut self) {
        self.src.remove();
        self.dst.remove();
    }
}
