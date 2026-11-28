use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::time::Instant;

use super::lsh::io_error;
use super::lsm_key::Frame;
use super::lsm_keying::{axis_keys, Keying};
use super::lsm_merge::{sort_rows, write_sorted, MergeIo, MergeJob, Merger};
use super::lsm_run::{remove_run, Run, RunMeta, RunWriter, Shape};
use crate::backend::MmapColumnStore;
use crate::error::ENNError;

pub(crate) const LSM_META: &str = "o1l_meta.json";

pub(crate) fn selected(work_dir: &Path) -> bool {
    work_dir.join(LSM_META).exists()
        || std::env::var("ENN_O1_KIND").is_ok_and(|v| v.eq_ignore_ascii_case("lsm"))
}

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|v| v.parse::<f64>().ok()).map(|v| v as usize)
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct LsmStats {
    pub flushes: u64,
    pub merges: u64,
    pub flush_ns: u64,
    pub merge_ns: u64,
    pub max_merge_ns: u64,
    pub max_merge_rows: u64,
    pub merged_rows: u64,
    pub resorted_rows: u64,
    pub bytes_written: u64,
    pub bytes_read: u64,
    pub rehashes: u64,
    #[serde(default)]
    pub backlog: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct LsmMeta {
    pub d: usize,
    pub block: usize,
    pub l0_rows: usize,
    pub ratio: usize,
    pub seed: u64,
    pub flushed: u64,
    pub next_id: u64,
    pub frame: Frame,
    #[serde(default)]
    pub float_keys: bool,
    #[serde(default)]
    pub raw_boxes: bool,
    #[serde(default)]
    pub no_rekey: bool,
    #[serde(default)]
    pub axis_proj: bool,
    pub runs: Vec<RunMeta>,
    pub stats: LsmStats,
}

pub(crate) struct LsmIndex {
    pub(crate) dir: PathBuf,
    pub(crate) shape: Shape,
    pub(crate) keying: Keying,
    pub(crate) meta: LsmMeta,
    pub(crate) runs: Vec<Run>,
    pub(crate) max_rows: usize,
    pub(crate) span: usize,
    pub(crate) n: usize,
    step: u64,
    active: Vec<Merger>,
}

impl LsmIndex {
    pub(crate) fn open(work_dir: &Path, seed: u64, fallback_scale: &[f64], rows: &MmapColumnStore) -> Result<Self, ENNError> {
        let d = rows.ncols;
        let stored: Option<LsmMeta> = ennbo_bpann::bpann_read_json(&work_dir.join(LSM_META)).map_err(io_error)?;
        let meta = stored.unwrap_or_else(|| {
            let block = env_usize("ENN_LSM_BLOCK").unwrap_or(4096);
            let l0 = env_usize("ENN_LSM_L0").unwrap_or((1usize << 20) / (8 * d.max(1)));
            LsmMeta {
                d,
                block,
                l0_rows: l0.max(256),
                ratio: env_usize("ENN_LSM_T").unwrap_or(4).max(2),
                seed,
                flushed: 0,
                next_id: 0,
                frame: Frame { scale: fallback_scale.to_vec(), ..Frame::default() },
                float_keys: std::env::var("ENN_LSM_KEY").is_ok_and(|v| v == "float"),
                raw_boxes: std::env::var("ENN_LSM_BOX").is_ok_and(|v| v == "raw"),
                no_rekey: std::env::var("ENN_LSM_REKEY").is_ok_and(|v| v == "off"),
                axis_proj: axis_keys(d),
                runs: Vec::new(),
                stats: LsmStats::default(),
            }
        });
        let keying = Keying::open(d, meta.seed, meta.axis_proj, meta.raw_boxes);
        let shape = Shape::new(d, keying.proj.m, keying.bdim(), meta.block);
        let runs = meta.runs.iter().map(|r| Run::open(work_dir, &shape, r.clone())).collect::<Result<_, _>>()?;
        let mut index = Self {
            dir: work_dir.to_path_buf(),
            shape,
            keying,
            meta,
            runs,
            max_rows: 3200,
            span: env_usize("ENN_LSM_SPAN").unwrap_or(1).max(1),
            n: 0,
            step: env_usize("ENN_LSM_STEP").unwrap_or(0) as u64,
            active: Vec::new(),
        };
        index.save()?;
        index.insert_rows(rows, 0)?;
        Ok(index)
    }

    fn save(&self) -> Result<(), ENNError> {
        ennbo_bpann::bpann_write_json(&self.dir.join(LSM_META), &self.meta).map_err(io_error)
    }

    pub(crate) fn hash_scale(&self) -> &[f64] {
        &self.meta.frame.scale
    }

    pub(crate) fn rehash(&mut self, scale: &[f64]) -> Result<(), ENNError> {
        if !self.meta.no_rekey && scale != self.meta.frame.scale.as_slice() {
            self.meta.frame = Frame { scale: scale.to_vec(), ..Frame::default() };
            self.meta.stats.rehashes += 1;
            self.save()?;
        }
        Ok(())
    }

    pub(crate) fn insert_rows(&mut self, rows: &MmapColumnStore, _start: usize) -> Result<(), ENNError> {
        let new = rows.nrows.saturating_sub(self.n) as u64;
        self.n = rows.nrows;
        while rows.nrows as u64 - self.meta.flushed >= self.meta.l0_rows as u64 {
            self.flush(rows)?;
            self.compact()?;
        }
        if self.step > 0 {
            self.advance(self.step.saturating_mul(new))?;
        }
        Ok(())
    }

    fn advance(&mut self, rows: u64) -> Result<(), ENNError> {
        let t0 = Instant::now();
        let mut i = 0;
        while i < self.active.len() {
            if self.active[i].step(rows)? {
                let mg = self.active.swap_remove(i);
                self.publish(mg, &t0)?;
            } else {
                i += 1;
            }
        }
        self.meta.stats.merge_ns += t0.elapsed().as_nanos() as u64;
        self.compact()?;
        self.meta.stats.backlog = self.active.iter().map(|m| m.total - m.done).sum();
        Ok(())
    }

    fn publish(&mut self, mg: Merger, t0: &Instant) -> Result<(), ENNError> {
        let inputs = mg.inputs.clone();
        let (meta, io) = mg.finish()?;
        self.runs.retain(|r| !inputs.contains(&r.meta.id));
        self.meta.runs.retain(|r| !inputs.contains(&r.id));
        self.runs.push(Run::open(&self.dir, &self.shape, meta.clone())?);
        self.meta.runs.push(meta);
        self.account(io, t0.elapsed().as_nanos() as u64, 0);
        self.save()?;
        inputs.iter().for_each(|&i| remove_run(&self.dir, i));
        Ok(())
    }

    fn account(&mut self, io: MergeIo, ns: u64, rows: u64) {
        let st = &mut self.meta.stats;
        st.max_merge_ns = st.max_merge_ns.max(ns);
        st.max_merge_rows = st.max_merge_rows.max(rows);
        st.merged_rows += rows;
        st.resorted_rows += io.resorted;
        st.bytes_written += io.written;
        st.bytes_read += io.read;
    }

    fn take_id(&mut self) -> u64 {
        self.meta.next_id += 1;
        self.meta.next_id - 1
    }

    fn flush(&mut self, rows: &MmapColumnStore) -> Result<(), ENNError> {
        let t0 = Instant::now();
        let lo = self.meta.flushed as usize;
        let hi = lo + self.meta.l0_rows;
        let x = rows.mmap_row_range(lo, hi)?;
        let x: Vec<f64> = x.iter().copied().collect();
        let ids: Vec<u32> = (lo as u32..hi as u32).collect();
        if self.meta.frame.center.is_empty() && !self.meta.float_keys {
            let mut zs = vec![0.0; ids.len() * self.shape.m];
            for (z, r) in zs.chunks_exact_mut(self.shape.m).zip(x.chunks_exact(self.shape.d)) {
                self.keying.proj.project(r, &self.meta.frame.scale, z);
            }
            self.meta.frame.fit(&zs, self.shape.m);
        }
        let (order, zs) = sort_rows(&self.keying, &self.meta.frame, &ids, &x);
        let id = self.take_id();
        let mut w = RunWriter::create(&self.dir, id, self.shape, true)?;
        write_sorted(&mut w, &self.keying, &order, &ids, &x, &zs)?;
        let (n, bytes) = w.finish()?;
        let meta = RunMeta { id, level: 0, rows: n, frame: self.meta.frame.clone() };
        self.runs.push(Run::open(&self.dir, &self.shape, meta.clone())?);
        self.meta.runs.push(meta);
        self.meta.flushed = hi as u64;
        let st = &mut self.meta.stats;
        st.flushes += 1;
        st.bytes_written += bytes;
        st.flush_ns += t0.elapsed().as_nanos() as u64;
        self.save()
    }

    fn busy(&self, id: u64) -> bool {
        self.active.iter().any(|m| m.inputs.contains(&id))
    }

    fn full_level(&self) -> Option<u32> {
        let top = self.meta.runs.iter().map(|r| r.level).max()?;
        (0..=top)
            .filter(|&l| !self.active.iter().any(|m| m.level == l + 1))
            .find(|&l| self.meta.runs.iter().filter(|r| r.level == l && !self.busy(r.id)).count() >= self.meta.ratio)
    }

    fn compact(&mut self) -> Result<(), ENNError> {
        while let Some(level) = self.full_level() {
            match self.step {
                0 => self.merge_level(level)?,
                _ => self.start_level(level)?,
            }
        }
        Ok(())
    }

    fn start_level(&mut self, level: u32) -> Result<(), ENNError> {
        let inputs: Vec<Run> = self
            .meta
            .runs
            .iter()
            .filter(|r| r.level == level && !self.busy(r.id))
            .map(|r| Run::open_bare(&self.dir, r.clone()))
            .collect::<Result<_, _>>()?;
        let rows = inputs.iter().map(|r| r.meta.rows).sum();
        let out_id = self.take_id();
        let mut next_tmp = (1u64 << 62) | (out_id << 20);
        let job = MergeJob { dir: &self.dir, shape: self.shape, keying: &self.keying, frame: &self.meta.frame };
        let mg = job.start(inputs, out_id, level + 1, &mut next_tmp)?;
        self.meta.stats.merges += 1;
        self.meta.stats.max_merge_rows = self.meta.stats.max_merge_rows.max(rows);
        self.meta.stats.merged_rows += rows;
        self.active.push(mg);
        self.save()
    }

    fn merge_level(&mut self, level: u32) -> Result<(), ENNError> {
        let t0 = Instant::now();
        let (inputs, keep): (Vec<Run>, Vec<Run>) = std::mem::take(&mut self.runs).into_iter().partition(|r| r.meta.level == level);
        self.runs = keep;
        let ids: Vec<u64> = inputs.iter().map(|r| r.meta.id).collect();
        let rows: u64 = inputs.iter().map(|r| r.meta.rows).sum();
        let out_id = self.take_id();
        let mut next_tmp = (1u64 << 62) | (out_id << 20);
        let job = MergeJob { dir: &self.dir, shape: self.shape, keying: &self.keying, frame: &self.meta.frame };
        let (meta, io) = job.merge(inputs, out_id, level + 1, &mut next_tmp)?;
        self.runs.push(Run::open(&self.dir, &self.shape, meta.clone())?);
        self.meta.runs.retain(|r| r.level != level);
        self.meta.runs.push(meta);
        let ns = t0.elapsed().as_nanos() as u64;
        self.meta.stats.merges += 1;
        self.meta.stats.merge_ns += ns;
        self.account(io, ns, rows);
        self.save()?;
        ids.iter().for_each(|&i| remove_run(&self.dir, i));
        Ok(())
    }

    pub(crate) fn blocks(&self) -> usize {
        self.runs.iter().map(|r| self.shape.blocks(r.meta.rows)).sum()
    }

    pub(crate) fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + self.runs.len() * 256 + self.meta.frame.scale.len() * 8 * (1 + self.runs.len())
    }

    pub(crate) fn set_budget(&mut self, _max_leaves: usize, max_candidates: usize) {
        self.max_rows = max_candidates.max(1);
    }
}
