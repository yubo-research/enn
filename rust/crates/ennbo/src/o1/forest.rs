use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering as AtomicOrdering;

use ennbo_bpann::{bpann_read_json, bpann_write_json};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use super::cache::SharedCache;
use super::forest_repair::{inverse, ForestMetrics, Repair, RepairStats, FOREST_METRICS_FILE};
use super::forest_rules::ForestRules;
use super::forest_tree::{Tree, TreeStats};
use super::lsh::{io_error, table_seed};
use crate::backend::MmapColumnStore;
use crate::error::ENNError;

pub(crate) const FOREST_STATE_FILE: &str = "o1f_index.json";

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ForestParams {
    pub num_trees: usize,
    pub leaf_cap: usize,
    pub max_leaves: usize,
    pub max_candidates: usize,
    pub alpha: f64,
    pub repair_rows: usize,
    pub stale_drift: f64,
    pub prefer_fresh: bool,
    pub rebuild_below: usize,
    pub io_width: usize,
    pub io_slow_ns: u64,
    pub pread_pivots: bool,
    pub rules: ForestRules,
    pub order: super::row_order::OrderParams,
    pub buffer: usize,
}

impl Default for ForestParams {
    fn default() -> Self {
        Self {
            num_trees: 1,
            leaf_cap: 32,
            max_leaves: 100,
            max_candidates: 3200,
            alpha: 0.7,
            repair_rows: 30,
            stale_drift: 0.25,
            prefer_fresh: true,
            rebuild_below: 3000,
            io_width: 10,
            io_slow_ns: 30_000,
            pread_pivots: true,
            rules: ForestRules::default(),
            order: super::row_order::OrderParams::default(),
            buffer: super::forest_batch::buffer_rows(),
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct MaintStats {
    pub insert_ns: u64,
    pub repair_ns: u64,
    pub repack_ns: u64,
    pub repacks: u64,
    pub repacked_nodes: u64,
    pub rebuilt_rows: u64,
    pub splits: u64,
    pub order_ns: u64,
    pub order_rows: u64,
    pub order_builds: u64,
    pub ordered_rows: u64,
    pub gathered: u64,
    pub rebuild_gathered: u64,
    pub route_rows: u64,
    pub deferred: u64,
    pub skipped: u64,
    pub aborted: u64,
    pub late_rows: u64,
    pub scan_nodes: u64,
    pub max_rebuild_rows: u64,
    pub job_active: u64,
    pub phase_ns: [u64; 5],
    pub phase_flt: [u64; 5],
    pub phase_rd: [u64; 5],
    pub majflt: [u64; 4],
    pub read_bytes: [u64; 4],
    pub store_rows: [u64; 2],
    pub gather_flt: [u64; 2],
    pub gather_ns: [u64; 5],
    pub split_age: [u64; 4],
    pub near_hits: u64,
    pub split_diag: [u64; 12],
    pub pack_hits: u64,
    pub pack_miss: u64,
    pub flushes: u64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
struct ForestState {
    num_trees: usize,
    leaf_cap: usize,
    indexed_rows: usize,
    metrics_version: u64,
    repair_cursor: Option<usize>,
}

pub(crate) struct ForestIndex {
    pub(super) params: ForestParams,
    pub(super) trees: Vec<Tree>,
    pub(super) invs: Vec<Vec<f64>>,
    pub(super) metrics: ForestMetrics,
    pub(super) repair: Option<Repair>,
    pub(super) repair_stats: RepairStats,
    pub(super) num_rows: usize,
    pub(super) seed: u64,
    pub(super) work_dir: PathBuf,
    pub(super) retired_io: (u64, u64),
    pub(super) cache: SharedCache,
    pub(super) maint: MaintStats,
    pub(super) order: Option<super::row_order::RowOrder>,
    pub(super) order_build: Option<super::row_order::OrderBuild>,
}

pub(crate) fn tree_path(work_dir: &Path, t: usize) -> PathBuf {
    work_dir.join(format!("o1f_tree_{t}.bin"))
}

fn file_id(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("o1f_tree_")?;
    let (num, ext) = rest.split_once('.')?;
    matches!(ext, "bin" | "build" | "pack")
        .then(|| num.parse().ok())
        .flatten()
}

const AUTO_WINDOW: u64 = 20_000;
const AUTO_STREAK: u32 = 3;

impl ForestIndex {
    pub(crate) fn open(
        work_dir: &Path,
        seed: u64,
        params: ForestParams,
        fallback_scale: &[f64],
        rows: &MmapColumnStore,
    ) -> Result<Self, ENNError> {
        let mut params = params;
        params.rules = params.rules.for_shape(rows.ncols, params.leaf_cap);
        let saved: Option<ForestState> = bpann_read_json(&work_dir.join(FOREST_STATE_FILE))
            .ok()
            .flatten();
        let metrics: Option<ForestMetrics> = bpann_read_json(&work_dir.join(FOREST_METRICS_FILE))
            .ok()
            .flatten();
        let recorded = metrics.as_ref().map(|m| m.tree_scales.len());
        params.num_trees = saved
            .as_ref()
            .map(|s| s.num_trees)
            .or(recorded)
            .filter(|&n| n > 0)
            .unwrap_or(params.num_trees);
        let mut index = Self {
            params,
            trees: Vec::new(),
            invs: Vec::new(),
            metrics: ForestMetrics {
                target: fallback_scale.to_vec(),
                ..ForestMetrics::default()
            },
            repair: None,
            repair_stats: RepairStats::default(),
            num_rows: 0,
            seed,
            work_dir: work_dir.to_path_buf(),
            retired_io: (0, 0),
            cache: None,
            maint: MaintStats::default(),
            order: super::row_order::RowOrder::open(work_dir, rows.ncols, rows.nrows)?,
            order_build: None,
        };
        if let (Some(state), Some(m)) = (saved, metrics) {
            if index.can_restore(&state, &m, rows) && index.restore(&state, m, rows)? {
                return Ok(index);
            }
        }
        index.rebuild_all(rows)?;
        Ok(index)
    }

    fn restore(
        &mut self,
        state: &ForestState,
        metrics: ForestMetrics,
        rows: &MmapColumnStore,
    ) -> Result<bool, ENNError> {
        let dirty = state.indexed_rows != rows.nrows || state.metrics_version != metrics.version;
        let trees = (0..self.params.num_trees)
            .map(|t| {
                Tree::open(
                    &tree_path(&self.work_dir, t),
                    self.params.leaf_cap,
                    table_seed(self.seed, t),
                )
            })
            .collect::<Result<Vec<_>, _>>();
        let Ok(trees) = trees else {
            return Ok(false);
        };
        let trees: Vec<Tree> = trees
            .into_iter()
            .map(|mut t| {
                t.nodes.set_cache(self.cache.clone());
                t.split = self.params.rules.split;
                t
            })
            .collect();
        self.invs = metrics.tree_scales.iter().map(|s| inverse(s)).collect();
        self.metrics = metrics;
        self.trees = trees;
        self.num_rows = state.indexed_rows;
        self.remove_orphans()?;
        if dirty {
            self.recover(rows, state.indexed_rows)?;
        } else {
            self.resume_repair(state.repair_cursor, rows.nrows)?;
        }
        Ok(true)
    }

    fn recover(&mut self, rows: &MmapColumnStore, from: usize) -> Result<(), ENNError> {
        if let Some((t, scale)) = self.metrics.repair.take() {
            if !super::forest_repair::shadow_path(&self.work_dir, t).exists()
                && t < self.trees.len()
            {
                self.invs[t] = inverse(&scale);
                self.metrics.tree_scales[t] = scale;
            }
        }
        self.remove_stale_files()?;
        self.save_metrics()?;
        let alpha = self.params.alpha;
        self.num_rows = rows.nrows;
        self.trees
            .par_iter_mut()
            .zip(self.invs.par_iter())
            .try_for_each(|(tree, inv)| {
                (from..rows.nrows).try_for_each(|id| {
                    tree.insert_absent(id as u32, super::row_order::Rows::plain(rows), inv, alpha)
                })
            })?;
        self.commit()
    }

    fn can_restore(&self, s: &ForestState, m: &ForestMetrics, rows: &MmapColumnStore) -> bool {
        s.num_trees == self.params.num_trees
            && s.leaf_cap == self.params.leaf_cap
            && s.indexed_rows <= rows.nrows
            && s.metrics_version <= m.version
            && m.target.len() == rows.ncols
            && m.tree_scales.len() == self.params.num_trees
            && m.tree_scales.iter().all(|t| t.len() == rows.ncols)
    }

    pub(super) fn tree_file(&self, t: usize) -> PathBuf {
        tree_path(&self.work_dir, t)
    }

    pub(crate) fn remove_orphans(&self) -> Result<(), ENNError> {
        for entry in std::fs::read_dir(&self.work_dir).map_err(io_error)? {
            let path = entry.map_err(io_error)?.path();
            let name = path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or_default();
            if file_id(name).is_some_and(|t| t >= self.params.num_trees || !name.ends_with(".bin"))
            {
                std::fs::remove_file(&path).map_err(io_error)?;
            }
        }
        Ok(())
    }

    pub(super) fn retire_io(&mut self, tree: &Tree) {
        self.retired_io.0 += tree.nodes.counters.reads.load(AtomicOrdering::Relaxed);
        self.retired_io.1 += tree.nodes.counters.writes.load(AtomicOrdering::Relaxed);
    }

    pub(super) fn save_metrics(&mut self) -> Result<(), ENNError> {
        self.metrics.version += 1;
        bpann_write_json(&self.work_dir.join(FOREST_METRICS_FILE), &self.metrics).map_err(io_error)
    }

    fn state_path(&self) -> PathBuf {
        self.work_dir.join(FOREST_STATE_FILE)
    }

    fn invalidate(&self) -> Result<(), ENNError> {
        match std::fs::remove_file(self.state_path()) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(io_error(e)),
            _ => Ok(()),
        }
    }

    pub(super) fn commit(&self) -> Result<(), ENNError> {
        let state = ForestState {
            num_trees: self.params.num_trees,
            leaf_cap: self.params.leaf_cap,
            indexed_rows: self.num_rows,
            metrics_version: self.metrics.version,
            repair_cursor: self.repair.as_ref().map(|r| r.cursor),
        };
        bpann_write_json(&self.state_path(), &state).map_err(io_error)
    }

    pub(super) fn rebuild_all(&mut self, rows: &MmapColumnStore) -> Result<(), ENNError> {
        if u32::try_from(rows.nrows).is_err() {
            return Err(ENNError::InvalidParameter(
                "O1F store holds at most 2^32 rows".into(),
            ));
        }
        self.invalidate()?;
        self.abandon_repair()?;
        self.remove_stale_files()?;
        self.remove_orphans()?;
        let old = std::mem::take(&mut self.trees);
        old.iter().for_each(|t| self.retire_io(t));
        let (cap, seed, dir, cache) = (
            self.params.leaf_cap,
            self.seed,
            self.work_dir.clone(),
            self.cache.clone(),
        );
        let (n, split) = (self.params.num_trees, self.params.rules.split);
        self.metrics.tree_scales = vec![self.metrics.target.clone(); n];
        self.invs = vec![inverse(&self.metrics.target); n];
        self.save_metrics()?;
        let inv = self.invs[0].clone();
        let replaces = u64::from(!old.is_empty());
        let mut prior: Vec<TreeStats> = old.iter().map(|t| t.stats).collect();
        prior.resize(n, TreeStats::default());
        drop(old);
        self.trees = (0..n)
            .into_par_iter()
            .map(|t| {
                let mut tree = Tree::create(
                    &tree_path(&dir, t),
                    cap,
                    table_seed(seed, t),
                    split,
                    rows.ncols,
                )?;
                tree.nodes.set_cache(cache.clone());
                tree.stats = TreeStats {
                    global_rebuilds: prior[t].global_rebuilds + replaces,
                    ..prior[t]
                };
                let root = tree.nodes.root();
                tree.build(
                    root,
                    (0..rows.nrows as u32).collect(),
                    super::row_order::Rows::plain(rows),
                    &inv,
                )?;
                tree.compact(&tree_path(&dir, t))?;
                Ok(tree)
            })
            .collect::<Result<Vec<_>, ENNError>>()?;
        self.num_rows = rows.nrows;
        self.commit()
    }

    pub(crate) fn insert_rows(
        &mut self,
        rows: &MmapColumnStore,
        start: usize,
    ) -> Result<(), ENNError> {
        if u32::try_from(rows.nrows).is_err() {
            return Err(ENNError::InvalidParameter(
                "O1F store holds at most 2^32 rows".into(),
            ));
        }
        let fresh = rows.nrows.saturating_sub(start);
        if self.params.buffer == 0 {
            return self.insert_from(rows, start, fresh);
        }
        if self.holds_back(rows) {
            return self.maintain_held(rows, fresh);
        }
        self.maint.flushes += 1;
        let start = self.num_rows.min(start);
        self.insert_from(rows, start, fresh)
    }

    pub(super) fn insert_from(
        &mut self,
        rows: &MmapColumnStore,
        start: usize,
        fresh: usize,
    ) -> Result<(), ENNError> {
        let alpha = self.params.alpha;
        let f0 = super::forest_order::io_marks();
        let t0 = std::time::Instant::now();
        let src = super::row_order::Rows {
            x: rows,
            order: self.order.as_ref(),
            near: None,
        };
        let ids: Vec<u32> = (start as u32..rows.nrows as u32).collect();
        let group = self.params.buffer > 0;
        self.trees
            .par_iter_mut()
            .zip(self.invs.par_iter())
            .try_for_each(|(tree, inv)| tree.insert_batch(&ids, src, inv, alpha, group))?;
        self.num_rows = rows.nrows;
        if let Some(us) = crate::backend::auto_us() {
            self.auto_advise(rows, us);
        }
        let (t1, f1) = (std::time::Instant::now(), super::forest_order::io_marks());
        let budget = fresh.saturating_mul(self.params.repair_rows);
        self.maintain(rows, budget)?;
        let (t2, f2) = (std::time::Instant::now(), super::forest_order::io_marks());
        self.repack_one()?;
        let f3 = super::forest_order::io_marks();
        self.maint.insert_ns += (t1 - t0).as_nanos() as u64;
        self.maint.repair_ns += (t2 - t1).as_nanos() as u64;
        self.maint.repack_ns += t2.elapsed().as_nanos() as u64;
        self.advance_order(rows, rows.nrows - start.min(rows.nrows))?;
        let f4 = super::forest_order::io_marks();
        let marks = [f0, f1, f2, f3, f4];
        for (k, w) in marks.windows(2).enumerate() {
            self.maint.majflt[k] += w[1][0].saturating_sub(w[0][0]);
            self.maint.read_bytes[k] += w[1][1].saturating_sub(w[0][1]);
        }
        self.batch_hygiene();
        self.commit()
    }

    fn auto_advise(&self, rows: &MmapColumnStore, us: f64) {
        static MARK: std::sync::Mutex<(u64, u64, u32)> = std::sync::Mutex::new((0, 0, 0));
        let (mut reads, mut ns) = (0, 0);
        for tree in &self.trees {
            reads += tree.nodes.counters.reads.load(std::sync::atomic::Ordering::Relaxed);
            ns += tree.nodes.counters.read_ns.load(std::sync::atomic::Ordering::Relaxed);
        }
        let Ok(mut mark) = MARK.lock() else {
            return;
        };
        if reads < mark.0 || ns < mark.1 {
            *mark = (reads, ns, 0);
            return;
        }
        let dr = reads - mark.0;
        if dr < AUTO_WINDOW {
            return;
        }
        let mean_us = (ns - mark.1) as f64 / dr as f64 / 1e3;
        let hot = if mean_us > us { mark.2 + 1 } else { 0 };
        *mark = (reads, ns, hot);
        if hot >= AUTO_STREAK && crate::backend::latch() {
            rows.advise_now();
            if let Some(o) = &self.order {
                o.advise_now();
            }
        }
    }

    pub(crate) fn maintain_now(
        &mut self,
        rows: &MmapColumnStore,
        max_rows: usize,
    ) -> Result<usize, ENNError> {
        self.flush_buffer(rows)?;
        self.invalidate()?;
        let done = self.maintain(rows, max_rows)?;
        self.commit()?;
        Ok(done)
    }

    pub(crate) fn rehash(
        &mut self,
        rows: &MmapColumnStore,
        hash_scale: &[f64],
    ) -> Result<(), ENNError> {
        self.flush_buffer(rows)?;
        if self.params.repair_rows == 0
            || self.num_rows <= self.params.rebuild_below.max(self.params.leaf_cap)
        {
            self.metrics.target = hash_scale.to_vec();
            return self.rebuild_all(rows);
        }
        self.invalidate()?;
        self.retarget(hash_scale)?;
        self.commit()
    }

    pub(super) fn tier(&self, tree: usize) -> f64 {
        let drift = self.tree_drift(tree);
        if self.params.prefer_fresh && drift > self.params.stale_drift {
            drift
        } else {
            0.0
        }
    }
}
