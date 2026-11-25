use std::path::{Path, PathBuf};
use std::time::Instant;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::forest::{tree_path, ForestIndex};
use super::forest_pack::PackJob;
use super::forest_repair::inverse;
use super::forest_rules::SplitConfig;
use super::forest_tree::Tree;
use super::lsh::{io_error, table_seed};
use crate::backend::MmapColumnStore;
use crate::error::ENNError;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct Seg {
    pub lo: usize,
    pub files: Vec<u64>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SegParams {
    pub rows: usize,
    pub work: usize,
}

struct TreeBuild {
    tree: Tree,
    work: Vec<(u32, Vec<u32>)>,
    pack: Option<PackJob>,
    path: PathBuf,
    done: bool,
}

pub(crate) struct SegJob {
    first: usize,
    count: usize,
    lo: usize,
    hi: usize,
    slots: Vec<usize>,
    fids: Vec<u64>,
    scale: Vec<f64>,
    inv: Vec<f64>,
    split: SplitConfig,
    built: Vec<Tree>,
    builds: Vec<TreeBuild>,
}

fn build_path(dir: &Path, fid: u64) -> PathBuf {
    tree_path(dir, fid).with_extension("build")
}

pub(crate) fn file_id(name: &str) -> Option<u64> {
    let rest = name.strip_prefix("o1f_tree_")?;
    let (num, ext) = rest.split_once('.')?;
    matches!(ext, "bin" | "build" | "pack").then(|| num.parse().ok()).flatten()
}

impl TreeBuild {
    fn step(&mut self, rows: &MmapColumnStore, inv: &[f64], left: &mut usize) -> Result<bool, ENNError> {
        while *left > 0 {
            if let Some((slot, ids)) = self.work.pop() {
                *left = left.saturating_sub(ids.len().max(1));
                if let Some(p) = self.tree.part(slot, ids, rows, inv)? {
                    self.tree.nodes.write(slot, &p.node)?;
                    self.tree.stats.splits += 1;
                    self.work.extend([p.left, p.right]);
                }
                continue;
            }
            let pack = self.pack.get_or_insert_with(|| PackJob::new(&self.tree.nodes, &self.path.with_extension("bin")));
            if let Some(nodes) = pack.step(&self.tree.nodes, left)? {
                nodes.absorb_counters(&self.tree.nodes);
                self.tree.nodes = nodes;
                self.tree.packed_at = self.tree.nodes.num_nodes();
                std::fs::remove_file(&self.path).map_err(io_error)?;
                self.done = true;
                return Ok(true);
            }
        }
        Ok(false)
    }
}

impl SegJob {
    fn start(&mut self, dir: &Path, seed: u64, cap: usize, ncols: usize) -> Result<(), ENNError> {
        for &fid in &self.fids {
            let path = build_path(dir, fid);
            let tree = Tree::create(&path, cap, table_seed(seed, fid as usize), self.split, ncols)?;
            let ids = (self.lo as u32..self.hi as u32).collect();
            self.builds.push(TreeBuild { work: vec![(tree.nodes.root(), ids)], tree, pack: None, path, done: false });
        }
        Ok(())
    }

    fn step(&mut self, dir: &Path, seed: u64, cap: usize, rows: &MmapColumnStore, budget: usize) -> Result<(bool, usize), ENNError> {
        if self.builds.is_empty() {
            self.start(dir, seed, cap, rows.ncols)?;
        }
        let share = budget.div_ceil(self.builds.len().max(1));
        let inv = &self.inv;
        let used: usize = self
            .builds
            .par_iter_mut()
            .filter(|b| !b.done)
            .map(|b| {
                let mut left = share;
                b.step(rows, inv, &mut left).map(|_| share - left)
            })
            .collect::<Result<Vec<_>, ENNError>>()?
            .into_iter()
            .sum();
        let finished = self.builds.iter().all(|b| b.done);
        if finished {
            self.built = std::mem::take(&mut self.builds).into_iter().map(|b| b.tree).collect();
        }
        Ok((finished, used.min(budget)))
    }

    fn discard(self, dir: &Path) {
        for &fid in &self.fids {
            for path in [build_path(dir, fid), tree_path(dir, fid)] {
                let _ = std::fs::remove_file(path);
            }
        }
    }
}

impl ForestIndex {
    pub(crate) fn segmented(&self) -> bool {
        self.params.seg.rows > 0 || self.metrics.segs.len() > 1
    }

    pub(crate) fn seg_bounds(&self, s: usize) -> (usize, usize) {
        let hi = self.metrics.segs.get(s + 1).map_or(self.num_rows, |n| n.lo);
        (self.metrics.segs[s].lo, hi)
    }

    pub(crate) fn active_trees(&self) -> std::ops::Range<usize> {
        let t = self.params.num_trees;
        let s = self.metrics.segs.len().saturating_sub(1);
        s * t..(s + 1) * t
    }

    pub(crate) fn set_segments(&mut self, params: SegParams) {
        self.params.seg = params;
    }

    fn new_fids(&mut self, n: usize) -> Vec<u64> {
        let first = self.metrics.next_fid.max(self.params.num_trees as u64);
        self.metrics.next_fid = first + n as u64;
        (first..first + n as u64).collect()
    }

    fn seal_if_full(&mut self) -> Result<(), ENNError> {
        let rows = self.params.seg.rows;
        let lo = self.metrics.segs.last().map_or(0, |s| s.lo);
        if rows == 0 || self.num_rows - lo < rows {
            return Ok(());
        }
        let n = self.params.num_trees;
        let files = self.new_fids(n);
        let (cap, dim, split) = (self.params.leaf_cap, self.metrics.target.len(), self.params.rules.split);
        for &fid in &files {
            let mut tree = Tree::create(&tree_path(&self.work_dir, fid), cap, table_seed(self.seed, fid as usize), split, dim)?;
            tree.nodes.set_cache(self.cache.clone());
            tree.base = self.num_rows as u32;
            self.trees.push(tree);
            self.invs.push(inverse(&self.metrics.target));
            self.metrics.tree_scales.push(self.metrics.target.clone());
        }
        self.metrics.segs.push(Seg { lo: self.num_rows, files });
        self.save_metrics()
    }

    fn merge_pair(&self) -> Option<usize> {
        let sealed = self.metrics.segs.len().saturating_sub(1);
        let len = |s: usize| self.seg_bounds(s).1 - self.seg_bounds(s).0;
        (0..sealed.saturating_sub(1)).rev().find(|&s| len(s) < 2 * len(s + 1))
    }

    fn stale_sealed(&self) -> Option<usize> {
        let sealed = self.active_trees().start;
        (0..sealed)
            .filter(|&t| self.tree_drift(t) > self.params.stale_drift)
            .max_by(|&a, &b| self.tree_drift(a).total_cmp(&self.tree_drift(b)).then(b.cmp(&a)))
    }

    fn pick_job(&mut self) -> Option<SegJob> {
        let n = self.params.num_trees;
        let (first, count, slots) = match (self.merge_pair(), self.stale_sealed()) {
            (Some(s), _) => (s, 2, (0..n).collect::<Vec<_>>()),
            (None, Some(t)) => (t / n, 1, vec![t % n]),
            (None, None) => return None,
        };
        let (lo, hi) = (self.seg_bounds(first).0, self.seg_bounds(first + count - 1).1);
        let fids = self.new_fids(slots.len());
        let scale = self.metrics.target.clone();
        let split = SplitConfig { par_above: 0, stage_above: 0, ..self.params.rules.split };
        Some(SegJob { first, count, lo, hi, slots, fids, inv: inverse(&scale), scale, split, built: Vec::new(), builds: Vec::new() })
    }

    pub(crate) fn step_segments(&mut self, rows: &MmapColumnStore, budget: usize) -> Result<usize, ENNError> {
        let t0 = Instant::now();
        self.seal_if_full()?;
        let mut done = 0;
        while done < budget {
            if self.seg_job.is_none() {
                self.seg_job = self.pick_job();
            }
            let (dir, seed, cap) = (self.work_dir.clone(), self.seed, self.params.leaf_cap);
            let Some(job) = self.seg_job.as_mut() else { break };
            let (finished, used) = job.step(&dir, seed, cap, rows, budget - done)?;
            done += used;
            if !finished {
                break;
            }
            self.finish_job()?;
        }
        self.maint.seg_ns += t0.elapsed().as_nanos() as u64;
        self.maint.seg_rows += done as u64;
        Ok(done)
    }

    fn finish_job(&mut self) -> Result<(), ENNError> {
        let Some(mut job) = self.seg_job.take() else { return Ok(()) };
        let n = self.params.num_trees;
        let mut built = std::mem::take(&mut job.built);
        for tree in &mut built {
            tree.nodes.set_cache(self.cache.clone());
            tree.split = self.params.rules.split;
        }
        let old_fids: Vec<u64>;
        let old: Vec<Tree> = if job.count == 2 {
            let r = job.first * n..(job.first + 2) * n;
            self.invs.splice(r.clone(), std::iter::repeat_n(job.inv.clone(), n));
            self.metrics.tree_scales.splice(r.clone(), std::iter::repeat_n(job.scale.clone(), n));
            let merged = Seg { lo: job.lo, files: job.fids.clone() };
            old_fids = self.metrics.segs.splice(job.first..job.first + 2, [merged]).flat_map(|s| s.files).collect();
            self.maint.merges += 1;
            self.trees.splice(r, built).collect()
        } else {
            let mut gone = Vec::new();
            let mut fids = Vec::new();
            for ((&k, tree), &fid) in job.slots.iter().zip(built).zip(&job.fids) {
                let t = job.first * n + k;
                gone.push(std::mem::replace(&mut self.trees[t], tree));
                self.invs[t] = job.inv.clone();
                self.metrics.tree_scales[t] = job.scale.clone();
                fids.push(std::mem::replace(&mut self.metrics.segs[job.first].files[k], fid));
            }
            old_fids = fids;
            self.repair_stats.finished += 1;
            gone
        };
        for t in &old {
            self.retire_io(t);
            self.maint.rebuilt_rows += t.stats.rebuilt_rows;
            self.maint.splits += t.stats.splits;
        }
        self.save_metrics()?;
        drop(old);
        for fid in old_fids {
            std::fs::remove_file(tree_path(&self.work_dir, fid)).map_err(io_error)?;
        }
        Ok(())
    }

    pub(crate) fn drop_seg_job(&mut self) {
        if let Some(job) = self.seg_job.take() {
            job.discard(&self.work_dir);
        }
    }

    pub(crate) fn remove_orphans(&self) -> Result<(), ENNError> {
        let live: std::collections::HashSet<u64> = self.metrics.segs.iter().flat_map(|s| s.files.iter().copied()).collect();
        for entry in std::fs::read_dir(&self.work_dir).map_err(io_error)? {
            let path = entry.map_err(io_error)?.path();
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or_default();
            if file_id(name).is_some_and(|fid| !live.contains(&fid) || !name.ends_with(".bin")) {
                std::fs::remove_file(&path).map_err(io_error)?;
            }
        }
        Ok(())
    }
}
