use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::Path;

use super::forest_node::Cut;
use super::forest_split::Projector;
use super::lsh::{io_error, mix64};
use crate::error::ENNError;

pub(crate) const SAMPLE: usize = 2048;
pub(crate) const CHUNK: usize = 4096;

pub(crate) struct Scratch {
    files: [File; 2],
    pub dim: usize,
}

fn open_unlinked(path: &Path) -> Result<File, ENNError> {
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)
        .map_err(io_error)?;
    std::fs::remove_file(path).map_err(io_error)?;
    Ok(file)
}

impl Scratch {
    pub(crate) fn create(dir: &Path, tag: u64, dim: usize) -> Result<Self, ENNError> {
        let name = |k: u32| dir.join(format!("o1f_job_{tag:016x}_{k}.tmp"));
        Ok(Self {
            files: [open_unlinked(&name(0))?, open_unlinked(&name(1))?],
            dim,
        })
    }

    fn rec(&self) -> usize {
        4 + 8 * self.dim
    }

    pub(crate) fn write(
        &self,
        f: usize,
        at: usize,
        ids: &[u32],
        xs: &[f64],
    ) -> Result<(), ENNError> {
        let mut buf = Vec::with_capacity(ids.len() * self.rec());
        for (k, id) in ids.iter().enumerate() {
            buf.extend_from_slice(&id.to_le_bytes());
            xs[k * self.dim..(k + 1) * self.dim]
                .iter()
                .for_each(|v| buf.extend_from_slice(&v.to_le_bytes()));
        }
        super::io_streams::wrote(super::io_streams::Stream::Scratch, (at * self.rec()) as u64, buf.len());
        self.files[f]
            .write_all_at(&buf, (at * self.rec()) as u64)
            .map_err(io_error)
    }

    pub(crate) fn read(
        &self,
        f: usize,
        at: usize,
        n: usize,
    ) -> Result<(Vec<u32>, Vec<f64>), ENNError> {
        let mut buf = vec![0u8; n * self.rec()];
        let t0 = std::time::Instant::now();
        self.files[f]
            .read_exact_at(&mut buf, (at * self.rec()) as u64)
            .map_err(io_error)?;
        super::io_streams::read(super::io_streams::Stream::Scratch, (at * self.rec()) as u64, buf.len(), t0);
        let (mut ids, mut xs) = (Vec::with_capacity(n), Vec::with_capacity(n * self.dim));
        for rec in buf.chunks_exact(self.rec()) {
            ids.push(u32::from_le_bytes(rec[..4].try_into().expect("id field")));
            xs.extend(
                rec[4..]
                    .chunks_exact(8)
                    .map(|b| f64::from_le_bytes(b.try_into().expect("f64 field"))),
            );
        }
        Ok((ids, xs))
    }
}

#[derive(Default)]
pub(crate) struct Sample {
    pub ids: Vec<u32>,
    pub xs: Vec<f64>,
    seen: u64,
}

impl Sample {
    pub(crate) fn offer(&mut self, id: u32, x: &[f64], salt: u64) {
        self.seen += 1;
        let d = x.len();
        if self.ids.len() < SAMPLE {
            self.ids.push(id);
            self.xs.extend_from_slice(x);
            return;
        }
        let j = (mix64(salt ^ self.seen.wrapping_mul(0x9e37_79b9_7f4a_7c15)) % self.seen) as usize;
        if j < SAMPLE {
            self.ids[j] = id;
            self.xs[j * d..(j + 1) * d].copy_from_slice(x);
        }
    }

    pub(crate) fn row(&self, id: u32, d: usize) -> Option<&[f64]> {
        let k = self.ids.iter().position(|&i| i == id)?;
        Some(&self.xs[k * d..(k + 1) * d])
    }
}

pub(crate) struct Part {
    pub cut: Cut,
    pub thr: f64,
    pub pivots: Option<(Vec<f64>, Vec<f64>)>,
    pub done: usize,
    pub nl: usize,
    pub nr: usize,
    pub samples: [Sample; 2],
}

pub(crate) struct Task {
    pub slot: u32,
    pub file: usize,
    pub off: usize,
    pub n: usize,
    pub key: u32,
    pub sample: Sample,
    pub part: Option<Part>,
}

impl Part {
    pub(crate) fn goes_left(&self, projector: &Projector<'_>, x: &[f64]) -> bool {
        let p = match (&self.cut, &self.pivots) {
            (Cut::Seed(seed), _) => projector.project(x, *seed),
            (Cut::Pivot(..), Some((a, b))) => projector.project_pivot(x, a, b),
            (Cut::Dir(dir), _) => dir.dot(x),
            (Cut::Pivot(..), None) => 0.0,
        };
        p < self.thr
    }

    pub(crate) fn step(
        &mut self,
        scratch: &Scratch,
        task: &Task,
        projector: &Projector<'_>,
    ) -> Result<usize, ENNError> {
        let d = scratch.dim;
        let k = CHUNK.min(task.n - self.done);
        let (ids, xs) = scratch.read(task.file, task.off + self.done, k)?;
        let (mut side, mut out) = ([Vec::new(), Vec::new()], [Vec::new(), Vec::new()]);
        for (j, &id) in ids.iter().enumerate() {
            let x = &xs[j * d..(j + 1) * d];
            let s = usize::from(!self.goes_left(projector, x));
            side[s].push(id);
            out[s].extend_from_slice(x);
            self.samples[s].offer(id, x, u64::from(task.key) ^ ((s as u64) << 40));
        }
        let dst = 1 - task.file;
        scratch.write(dst, task.off + self.nl, &side[0], &out[0])?;
        self.nl += side[0].len();
        self.nr += side[1].len();
        scratch.write(dst, task.off + task.n - self.nr, &side[1], &out[1])?;
        self.done += k;
        Ok(k)
    }
}
