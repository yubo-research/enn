use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;
use std::os::unix::fs::FileExt;
use std::path::Path;

use super::lsh::io_error;
use super::lsm_key::{z_cmp, Frame, MAX_AXES};
use super::lsm_keying::Keying;
use super::lsm_run::{block_row, remove_run, Run, RunMeta, RunWriter, Shape};
use crate::error::ENNError;

pub(crate) fn sort_chunk_rows(shape: &Shape) -> usize {
    let env = std::env::var("ENN_LSM_CHUNK").ok().and_then(|v| v.parse::<usize>().ok());
    env.unwrap_or(((2usize << 20) / shape.row_bytes()).max(1024)).max(1)
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct MergeIo {
    pub read: u64,
    pub written: u64,
    pub resorted: u64,
}

pub(crate) fn sort_rows(keying: &Keying, frame: &Frame, ids: &[u32], x: &[f64]) -> (Vec<usize>, Vec<f64>) {
    let (m, d, kl) = (keying.proj.m, keying.proj.d, keying.klen());
    let n = ids.len();
    let mut keys = vec![0u64; n * kl];
    let mut zs = vec![0.0; n * m];
    for i in 0..n {
        let (z, k) = (&mut zs[i * m..(i + 1) * m], &mut keys[i * kl..(i + 1) * kl]);
        keying.key(&x[i * d..(i + 1) * d], frame, z, k);
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_unstable_by(|&a, &b| {
        z_cmp(&keys[a * kl..(a + 1) * kl], &keys[b * kl..(b + 1) * kl]).then(ids[a].cmp(&ids[b]))
    });
    (order, zs)
}

pub(crate) fn write_sorted(w: &mut RunWriter, keying: &Keying, order: &[usize], ids: &[u32], x: &[f64], zs: &[f64]) -> Result<(), ENNError> {
    let (d, m) = (keying.proj.d, keying.proj.m);
    for &i in order {
        let (xi, zi) = (&x[i * d..(i + 1) * d], &zs[i * m..(i + 1) * m]);
        w.push(ids[i], xi, keying.box_coords(xi, zi))?;
    }
    Ok(())
}

struct Stream {
    run: Run,
    buf: Vec<u8>,
    next_block: usize,
    blocks_per_read: usize,
    at: usize,
    in_buf: usize,
    left: u64,
    id: u32,
    x: Vec<f64>,
}

impl Stream {
    fn new(run: Run, shape: &Shape, buf_bytes: usize) -> Self {
        let left = run.meta.rows;
        Self {
            run,
            buf: Vec::new(),
            next_block: 0,
            blocks_per_read: (buf_bytes / shape.block).max(1),
            at: 0,
            in_buf: 0,
            left,
            id: 0,
            x: vec![0.0; shape.d],
        }
    }

    fn advance(&mut self, shape: &Shape, io: &mut MergeIo) -> Result<bool, ENNError> {
        if self.left == 0 {
            return Ok(false);
        }
        if self.at == self.in_buf {
            let total = shape.blocks(self.run.meta.rows);
            let nb = self.blocks_per_read.min(total - self.next_block);
            self.buf.resize(nb * shape.block, 0);
            self.run
                .file()
                .read_exact_at(&mut self.buf, (self.next_block * shape.block) as u64)
                .map_err(io_error)?;
            io.read += self.buf.len() as u64;
            self.next_block += nb;
            self.at = 0;
            self.in_buf = (nb * shape.per).min(self.left as usize);
        }
        let (b, i) = (self.at / shape.per, self.at % shape.per);
        let blk = &self.buf[b * shape.block..(b + 1) * shape.block];
        self.id = block_row(shape, blk, i, &mut self.x);
        self.at += 1;
        self.left -= 1;
        Ok(true)
    }
}

struct Head {
    key: [u64; MAX_AXES],
    m: usize,
    id: u32,
    s: usize,
}

impl PartialEq for Head {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Head {}
impl PartialOrd for Head {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Head {
    fn cmp(&self, o: &Self) -> Ordering {
        z_cmp(&self.key[..self.m], &o.key[..o.m]).then(self.id.cmp(&o.id))
    }
}

pub(crate) struct MergeJob<'a> {
    pub dir: &'a Path,
    pub shape: Shape,
    pub keying: &'a Keying,
    pub frame: &'a Frame,
}

impl MergeJob<'_> {
    fn resort(&self, run: Run, next_tmp: &mut u64, io: &mut MergeIo) -> Result<Vec<Run>, ENNError> {
        let chunk = sort_chunk_rows(&self.shape);
        let mut s = Stream::new(run, &self.shape, 1 << 20);
        let mut out = Vec::new();
        loop {
            let (mut ids, mut x) = (Vec::with_capacity(chunk), Vec::with_capacity(chunk * self.shape.d));
            while ids.len() < chunk && s.advance(&self.shape, io)? {
                ids.push(s.id);
                x.extend_from_slice(&s.x);
            }
            if ids.is_empty() {
                return Ok(out);
            }
            io.resorted += ids.len() as u64;
            let (order, zs) = sort_rows(self.keying, self.frame, &ids, &x);
            let id = *next_tmp;
            *next_tmp += 1;
            let mut w = RunWriter::create(self.dir, id, self.shape, false)?;
            write_sorted(&mut w, self.keying, &order, &ids, &x, &zs)?;
            let (rows, bytes) = w.finish()?;
            io.written += bytes;
            let meta = RunMeta { id, level: 0, rows, frame: self.frame.clone() };
            out.push(Run::open_bare(self.dir, meta)?);
        }
    }

    pub(crate) fn start(&self, inputs: Vec<Run>, out_id: u64, level: u32, next_tmp: &mut u64) -> Result<Merger, ENNError> {
        let mut io = MergeIo::default();
        let mut sources = Vec::new();
        let mut temps = Vec::new();
        let inputs_ids = inputs.iter().map(|r| r.meta.id).collect();
        for run in inputs {
            if &run.meta.frame == self.frame {
                sources.push(run);
            } else {
                let parts = self.resort(run, next_tmp, &mut io)?;
                temps.extend(parts.iter().map(|r| r.meta.id));
                sources.extend(parts);
            }
        }
        let total = sources.iter().map(|r| r.meta.rows).sum();
        let buf = ((4usize << 20) / sources.len().max(1)).clamp(16 << 10, 256 << 10);
        let mut streams: Vec<Stream> = sources.into_iter().map(|r| Stream::new(r, &self.shape, buf)).collect();
        let mut z = vec![0.0; self.shape.m];
        let mut heap = BinaryHeap::with_capacity(streams.len());
        for (s, st) in streams.iter_mut().enumerate() {
            if st.advance(&self.shape, &mut io)? {
                heap.push(Reverse(self.head(st, s, &mut z)));
            }
        }
        Ok(Merger {
            dir: self.dir.to_path_buf(),
            shape: self.shape,
            keying: self.keying.clone(),
            frame: self.frame.clone(),
            writer: Some(RunWriter::create(self.dir, out_id, self.shape, true)?),
            streams,
            heap,
            temps,
            inputs: inputs_ids,
            out_id,
            level,
            io,
            z,
            total,
            done: 0,
        })
    }

    pub(crate) fn merge(&self, inputs: Vec<Run>, out_id: u64, level: u32, next_tmp: &mut u64) -> Result<(RunMeta, MergeIo), ENNError> {
        let mut mg = self.start(inputs, out_id, level, next_tmp)?;
        mg.step(u64::MAX)?;
        mg.finish()
    }

    fn head(&self, st: &Stream, s: usize, z: &mut [f64]) -> Head {
        head_of(self.keying, self.frame, st, s, z)
    }
}

fn head_of(keying: &Keying, frame: &Frame, st: &Stream, s: usize, z: &mut [f64]) -> Head {
    let m = keying.klen();
    let mut key = [0u64; MAX_AXES];
    keying.key(&st.x, frame, z, &mut key[..m]);
    Head { key, m, id: st.id, s }
}

pub(crate) struct Merger {
    dir: std::path::PathBuf,
    shape: Shape,
    keying: Keying,
    frame: Frame,
    writer: Option<RunWriter>,
    streams: Vec<Stream>,
    heap: BinaryHeap<Reverse<Head>>,
    temps: Vec<u64>,
    pub inputs: Vec<u64>,
    pub out_id: u64,
    pub level: u32,
    io: MergeIo,
    z: Vec<f64>,
    pub total: u64,
    pub done: u64,
}

impl Merger {
    pub(crate) fn step(&mut self, rows: u64) -> Result<bool, ENNError> {
        let Some(w) = self.writer.as_mut() else {
            return Ok(true);
        };
        let mut moved = 0;
        while moved < rows {
            let Some(Reverse(h)) = self.heap.pop() else {
                return Ok(true);
            };
            let st = &mut self.streams[h.s];
            if !self.keying.raw {
                self.keying.proj.project(&st.x, &self.frame.scale, &mut self.z);
            }
            w.push(st.id, &st.x, self.keying.box_coords(&st.x, &self.z))?;
            if st.advance(&self.shape, &mut self.io)? {
                self.heap.push(Reverse(head_of(&self.keying, &self.frame, st, h.s, &mut self.z)));
            }
            moved += 1;
            self.done += 1;
        }
        Ok(self.heap.is_empty())
    }

    pub(crate) fn finish(mut self) -> Result<(RunMeta, MergeIo), ENNError> {
        self.streams.clear();
        let w = self.writer.take().ok_or_else(|| ENNError::InvalidParameter("merge already finished".into()))?;
        let (rows, bytes) = w.finish()?;
        self.io.written += bytes;
        self.temps.iter().for_each(|&t| remove_run(&self.dir, t));
        let meta = RunMeta { id: self.out_id, level: self.level, rows, frame: self.frame.clone() };
        Ok((meta, self.io))
    }
}
