use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use super::lsh::io_error;
use super::lsm_key::{empty_box, grow_box, union_box, Frame};
use crate::error::ENNError;

pub(crate) const FANOUT: usize = 16;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct RunMeta {
    pub id: u64,
    pub level: u32,
    pub rows: u64,
    pub frame: Frame,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct Shape {
    pub d: usize,
    pub m: usize,
    pub bm: usize,
    pub per: usize,
    pub block: usize,
}

impl Shape {
    pub(crate) fn new(d: usize, m: usize, bm: usize, block_bytes: usize) -> Self {
        let row = 4 + 8 * d;
        let block = block_bytes.max(row).div_ceil(4096) * 4096;
        Self {
            d,
            m,
            bm,
            per: (block / row).max(1),
            block,
        }
    }

    pub(crate) fn row_bytes(&self) -> usize {
        4 + 8 * self.d
    }

    pub(crate) fn box_bytes(&self) -> usize {
        16 * self.bm
    }

    pub(crate) fn blocks(&self, rows: u64) -> usize {
        (rows as usize).div_ceil(self.per)
    }

    pub(crate) fn levels(&self, blocks: usize) -> Vec<(usize, usize)> {
        let mut out = vec![(0, blocks)];
        while out.last().is_some_and(|l| l.1 > 1) {
            let (off, n) = *out.last().unwrap_or(&(0, 0));
            out.push((off + n, n.div_ceil(FANOUT)));
        }
        out
    }
}

pub(crate) fn run_path(dir: &Path, id: u64) -> PathBuf {
    dir.join(format!("o1l_run_{id}.bin"))
}

pub(crate) fn box_path(dir: &Path, id: u64) -> PathBuf {
    dir.join(format!("o1l_box_{id}.bin"))
}

fn put_f64s(out: &mut Vec<u8>, v: &[f64]) {
    v.iter().for_each(|x| out.extend_from_slice(&x.to_le_bytes()));
}

pub(crate) fn get_f64s(bytes: &[u8], out: &mut [f64]) {
    for (o, c) in out.iter_mut().zip(bytes.chunks_exact(8)) {
        let mut a = [0u8; 8];
        a.copy_from_slice(c);
        *o = f64::from_le_bytes(a);
    }
}

pub(crate) fn block_row(shape: &Shape, block: &[u8], i: usize, x: &mut [f64]) -> u32 {
    let off = i * shape.row_bytes();
    let mut a = [0u8; 4];
    a.copy_from_slice(&block[off..off + 4]);
    get_f64s(&block[off + 4..off + shape.row_bytes()], x);
    u32::from_le_bytes(a)
}

pub(crate) struct RunWriter {
    shape: Shape,
    run: BufWriter<File>,
    boxes: Option<BufWriter<File>>,
    box_file: PathBuf,
    block: Vec<u8>,
    in_block: usize,
    bx: Vec<f64>,
    rows: u64,
    blocks: usize,
    pub bytes: u64,
}

impl RunWriter {
    pub(crate) fn create(
        dir: &Path,
        id: u64,
        shape: Shape,
        with_boxes: bool,
    ) -> Result<Self, ENNError> {
        let cap = 1 << 20;
        let run = BufWriter::with_capacity(cap, File::create(run_path(dir, id)).map_err(io_error)?);
        let boxes = with_boxes
            .then(|| File::create(box_path(dir, id)).map(|f| BufWriter::with_capacity(1 << 16, f)))
            .transpose()
            .map_err(io_error)?;
        Ok(Self {
            shape,
            run,
            boxes,
            box_file: box_path(dir, id),
            block: Vec::with_capacity(shape.block),
            in_block: 0,
            bx: empty_box(shape.bm),
            rows: 0,
            blocks: 0,
            bytes: 0,
        })
    }

    pub(crate) fn push(&mut self, id: u32, x: &[f64], z: &[f64]) -> Result<(), ENNError> {
        self.block.extend_from_slice(&id.to_le_bytes());
        put_f64s(&mut self.block, x);
        grow_box(&mut self.bx, z);
        self.in_block += 1;
        self.rows += 1;
        if self.in_block == self.shape.per {
            self.end_block()?;
        }
        Ok(())
    }

    fn end_block(&mut self) -> Result<(), ENNError> {
        if self.in_block == 0 {
            return Ok(());
        }
        self.block.resize(self.shape.block, 0);
        self.run.write_all(&self.block).map_err(io_error)?;
        if let Some(b) = self.boxes.as_mut() {
            let mut bytes = Vec::with_capacity(self.shape.box_bytes());
            put_f64s(&mut bytes, &self.bx);
            b.write_all(&bytes).map_err(io_error)?;
        }
        self.bytes += (self.shape.block + self.boxes.as_ref().map_or(0, |_| self.shape.box_bytes())) as u64;
        self.block.clear();
        self.in_block = 0;
        self.blocks += 1;
        self.bx = empty_box(self.shape.bm);
        Ok(())
    }

    pub(crate) fn finish(mut self) -> Result<(u64, u64), ENNError> {
        self.end_block()?;
        let run = self.run.into_inner().map_err(|e| io_error(e.into_error()))?;
        run.sync_data().map_err(io_error)?;
        if let Some(b) = self.boxes.take() {
            b.into_inner().map_err(|e| io_error(e.into_error()))?;
            self.bytes += build_upper(&self.box_file, &self.shape, self.blocks)?;
        }
        Ok((self.rows, self.bytes))
    }
}

fn build_upper(path: &Path, shape: &Shape, blocks: usize) -> Result<u64, ENNError> {
    let f = OpenOptions::new().read(true).write(true).open(path).map_err(io_error)?;
    let bb = shape.box_bytes();
    let levels = shape.levels(blocks);
    let chunk = FANOUT * 256;
    let mut written = 0u64;
    let mut buf = Vec::new();
    let mut child = vec![0.0; 2 * shape.bm];
    for w in levels.windows(2) {
        let ((src, n), (dst, _)) = (w[0], w[1]);
        let mut out_at = dst;
        for start in (0..n).step_by(chunk) {
            let take = chunk.min(n - start);
            buf.resize(take * bb, 0);
            f.read_exact_at(&mut buf, ((src + start) * bb) as u64).map_err(io_error)?;
            let mut out = Vec::with_capacity(take.div_ceil(FANOUT) * bb);
            for group in buf.chunks(FANOUT * bb) {
                let mut acc = empty_box(shape.bm);
                for c in group.chunks_exact(bb) {
                    get_f64s(c, &mut child);
                    union_box(&mut acc, &child);
                }
                put_f64s(&mut out, &acc);
            }
            f.write_all_at(&out, (out_at * bb) as u64).map_err(io_error)?;
            out_at += out.len() / bb;
            written += out.len() as u64;
        }
    }
    f.sync_data().map_err(io_error)?;
    Ok(written)
}

pub(crate) struct Run {
    pub meta: RunMeta,
    file: File,
    boxes: Option<Mmap>,
    pub levels: Vec<(usize, usize)>,
}

impl Run {
    pub(crate) fn open(dir: &Path, shape: &Shape, meta: RunMeta) -> Result<Self, ENNError> {
        let file = File::open(run_path(dir, meta.id)).map_err(io_error)?;
        let bf = File::open(box_path(dir, meta.id)).map_err(io_error)?;
        let boxes = unsafe { Mmap::map(&bf) }.map_err(io_error)?;
        let levels = shape.levels(shape.blocks(meta.rows));
        Ok(Self {
            meta,
            file,
            boxes: Some(boxes),
            levels,
        })
    }

    pub(crate) fn open_bare(dir: &Path, meta: RunMeta) -> Result<Self, ENNError> {
        let file = File::open(run_path(dir, meta.id)).map_err(io_error)?;
        Ok(Self {
            meta,
            file,
            boxes: None,
            levels: Vec::new(),
        })
    }

    pub(crate) fn file(&self) -> &File {
        &self.file
    }

    pub(crate) fn rows_in(&self, shape: &Shape, b: usize) -> usize {
        (self.meta.rows as usize - b * shape.per).min(shape.per)
    }

    pub(crate) fn read_blocks(&self, shape: &Shape, lo: usize, hi: usize, out: &mut Vec<u8>) -> Result<usize, ENNError> {
        out.resize((hi - lo) * shape.block, 0);
        self.file
            .read_exact_at(out, (lo * shape.block) as u64)
            .map_err(io_error)?;
        Ok((lo..hi).map(|b| self.rows_in(shape, b)).sum())
    }

    pub(crate) fn bx(&self, shape: &Shape, level: usize, idx: usize, out: &mut [f64]) {
        let bb = shape.box_bytes();
        let at = (self.levels[level].0 + idx) * bb;
        if let Some(m) = self.boxes.as_ref() {
            get_f64s(&m[at..at + bb], out);
        }
    }
}

pub(crate) fn remove_run(dir: &Path, id: u64) {
    let _ = std::fs::remove_file(run_path(dir, id));
    let _ = std::fs::remove_file(box_path(dir, id));
}
