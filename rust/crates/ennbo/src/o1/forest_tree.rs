use std::path::{Path, PathBuf};

use super::cache::ObjCache;
use super::forest_defer::Job;
use super::forest_lrows::{lrows_path, LeafRows, NearRows};
use super::forest_node::{Cut, Node, NodeFile, NIL};
use super::forest_rules::{SplitConfig, SplitRule};
use super::forest_split::{Placed, Projector};
use super::lsh::io_error;
use super::row_order::Rows;
use super::search::{fetch_row, RowRef, X_COLUMN};
use crate::error::ENNError;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct TreeStats {
    pub splits: u64,
    pub chain_pages: u64,
    pub rebuilds: u64,
    pub rebuilt_rows: u64,
    pub max_rebuild_rows: u64,
    pub global_rebuilds: u64,
    pub gathered: u64,
    pub rebuild_gathered: u64,
    pub route_rows: u64,
    pub deferred: u64,
    pub skipped: u64,
    pub aborted: u64,
    pub late_rows: u64,
    pub scan_nodes: u64,
    pub phase_ns: [u64; 5],
    pub phase_flt: [u64; 5],
    pub phase_rd: [u64; 5],
    pub split_store: u64,
    pub rebuild_store: u64,
    pub split_flt: u64,
    pub rebuild_flt: u64,
    pub split_ns: u64,
    pub rebuild_ns: u64,
    pub split_age: [u64; 4],
    pub near_hits: u64,
    pub split_diag: [u64; 12],
}

impl TreeStats {
    pub(crate) fn absorb(&mut self, old: &TreeStats) {
        self.splits += old.splits;
        self.chain_pages += old.chain_pages;
        self.rebuilds += old.rebuilds;
        self.rebuilt_rows += old.rebuilt_rows;
        self.max_rebuild_rows = self.max_rebuild_rows.max(old.max_rebuild_rows);
        self.global_rebuilds += old.global_rebuilds;
        self.gathered += old.gathered;
        self.rebuild_gathered += old.rebuild_gathered;
        self.route_rows += old.route_rows;
        self.deferred += old.deferred;
        self.skipped += old.skipped;
        self.aborted += old.aborted;
        self.late_rows += old.late_rows;
        self.scan_nodes += old.scan_nodes;
        self.phase_ns
            .iter_mut()
            .zip(old.phase_ns)
            .for_each(|(a, b)| *a += b);
        for k in 0..5 {
            self.phase_flt[k] += old.phase_flt[k];
            self.phase_rd[k] += old.phase_rd[k];
        }
        self.split_store += old.split_store;
        self.rebuild_store += old.rebuild_store;
        self.split_flt += old.split_flt;
        self.rebuild_flt += old.rebuild_flt;
        self.split_ns += old.split_ns;
        self.rebuild_ns += old.rebuild_ns;
        self.split_age
            .iter_mut()
            .zip(old.split_age)
            .for_each(|(a, b)| *a += b);
        self.near_hits += old.near_hits;
        for k in 0..12 {
            self.split_diag[k] += old.split_diag[k];
        }
    }
}

pub(crate) struct Tree {
    pub(crate) nodes: NodeFile,
    pub(crate) seed: u64,
    pub(crate) stats: TreeStats,
    pub(crate) split: SplitConfig,
    pub(crate) packed_at: u32,
    pub(crate) dir: PathBuf,
    pub(crate) job: Option<Box<Job>>,
    pub(crate) path: PathBuf,
    pub(crate) lrows: Option<LeafRows>,
    pub(crate) mixed: bool,
    pub(crate) warm: super::forest_prefetch::Gate,
}

#[derive(Clone, Copy)]
pub(crate) struct RowSource<'a> {
    pub store: Rows<'a>,
    pub cache: Option<&'a ObjCache>,
    pub pread: bool,
}

pub(crate) struct SplitRef {
    pub cut: Cut,
    pub thr: f64,
    pub left: u32,
    pub right: u32,
}

pub(crate) struct LeafVisit<'a> {
    pub depth: usize,
    pub pages: usize,
    pub ids: &'a [u32],
}

pub(super) fn row(rows: Rows<'_>, id: u32) -> Result<std::borrow::Cow<'_, [f64]>, ENNError> {
    rows.mapped(id as usize)
}

pub(super) fn uncovered(rows: Rows<'_>, ids: &[u32]) -> u64 {
    let covered = rows.order.map_or(0, |o| o.rows());
    ids.iter().filter(|&&i| i as usize >= covered).count() as u64
}

fn parent(path: &Path) -> PathBuf {
    path.parent().map_or_else(PathBuf::new, Path::to_path_buf)
}

pub(crate) fn depth_limit(n: usize, cap: usize, alpha: f64) -> usize {
    let ratio = (n.max(cap) as f64) / (cap as f64);
    (ratio.ln() / (1.0 / alpha).ln()).ceil() as usize + 1
}

impl Tree {
    pub(crate) fn create(
        path: &Path,
        cap: usize,
        seed: u64,
        split: SplitConfig,
        dim: usize,
    ) -> Result<Self, ENNError> {
        let _ = std::fs::remove_file(lrows_path(path));
        let _ = std::fs::remove_file(lrows_path(path).with_extension("lslot"));
        Ok(Self {
            nodes: NodeFile::create(path, cap, dim, split.rule.payload(dim))?,
            seed,
            stats: TreeStats::default(),
            split,
            packed_at: 0,
            dir: parent(path),
            job: None,
            path: path.to_path_buf(),
            lrows: None,
            mixed: false,
            warm: super::forest_prefetch::Gate::default(),
        })
    }

    pub(crate) fn open(path: &Path, cap: usize, seed: u64) -> Result<Self, ENNError> {
        let nodes = NodeFile::open(path, cap)?;
        let packed_at = nodes.num_nodes();
        let dim = nodes.dim_payload().0;
        let lrows = if super::forest_lrows::clustered() && dim > 0 && lrows_path(path).exists() {
            Some(LeafRows::open(path, cap, dim, false)?)
        } else {
            None
        };
        Ok(Self {
            nodes,
            seed,
            stats: TreeStats::default(),
            split: SplitConfig::default(),
            packed_at,
            dir: parent(path),
            job: None,
            path: path.to_path_buf(),
            lrows,
            mixed: false,
            warm: super::forest_prefetch::Gate::default(),
        })
    }

    pub(super) fn projector<'a>(&self, inv_scale: &'a [f64]) -> Projector<'a> {
        Projector {
            tree_seed: self.seed,
            inv_scale,
        }
    }

    pub(crate) fn branch(
        &self,
        x: &[f64],
        inv_scale: &[f64],
        split: &SplitRef,
        rows: RowSource<'_>,
        weights: Option<&[f64]>,
    ) -> Result<(f64, f64, u32, u32), ENNError> {
        let projector = self.projector(inv_scale);
        let (p, norm) = match split.cut {
            Cut::Seed(seed) => (
                projector.project(x, seed),
                weights.map(|w| projector.seed_dual_norm(w)),
            ),
            Cut::Pivot(a, b) => {
                let get = |i: u32| match (rows.pread && !rows.store.covers(i), rows.cache) {
                    (true, None) => rows
                        .store
                        .pread(i as usize)
                        .map(|r| RowRef::Shared(r.into())),
                    _ => fetch_row(rows.store, rows.cache, X_COLUMN, i as usize),
                };
                let (ra, rb) = (get(a)?, get(b)?);
                (
                    projector.project_pivot(x, &ra, &rb),
                    weights.map(|w| projector.pivot_dual_norm(&ra, &rb, w)),
                )
            }
            Cut::Dir(ref dir) => (
                dir.dot(x),
                weights.map(|w| super::forest_bound::dual_norm(dir, w)),
            ),
        };
        let raw = p - split.thr;
        let bound = match norm {
            Some(n) if n > 0.0 && n.is_finite() => raw.abs() / n,
            Some(_) => 0.0,
            None => raw.abs(),
        };
        Ok(if raw < 0.0 {
            (-raw, bound, split.left, split.right)
        } else {
            (raw, bound, split.right, split.left)
        })
    }

    pub(crate) fn step(
        &self,
        x: &[f64],
        inv_scale: &[f64],
        node: u32,
        rows: RowSource<'_>,
    ) -> Result<Option<(f64, u32, u32)>, ENNError> {
        match self.nodes.read(node)? {
            Node::Split {
                cut,
                thr,
                left,
                right,
            } => self
                .branch(
                    x,
                    inv_scale,
                    &SplitRef {
                        cut,
                        thr,
                        left,
                        right,
                    },
                    rows,
                    None,
                )
                .map(|(m, _, near, far)| Some((m, near, far))),
            _ => Ok(None),
        }
    }

    pub(crate) fn leaf_ids(&self, leaf: u32, out: &mut Vec<u32>) -> Result<usize, ENNError> {
        let mut at = leaf;
        let mut pages = 0;
        while at != NIL {
            match self.nodes.read(at)? {
                Node::Leaf { ids, next } => {
                    out.extend_from_slice(&ids);
                    pages += 1;
                    at = next;
                }
                _ => return Err(io_error("forest leaf chain is corrupt")),
            }
        }
        Ok(pages)
    }

    pub(super) fn descend_from(
        &self,
        start: u32,
        x: &[f64],
        inv_scale: &[f64],
        rows: Rows<'_>,
    ) -> Result<(Vec<u32>, u32), ENNError> {
        let mut path = Vec::new();
        let mut at = start;
        let src = RowSource {
            store: rows,
            cache: None,
            pread: false,
        };
        while let Some((_, near, _)) = self.step(x, inv_scale, at, src)? {
            path.push(at);
            at = near;
        }
        Ok((path, at))
    }

    pub(crate) fn insert(
        &mut self,
        id: u32,
        rows: Rows<'_>,
        inv_scale: &[f64],
        alpha: f64,
    ) -> Result<(), ENNError> {
        self.insert_row(id, rows, inv_scale, alpha, false)
    }

    pub(crate) fn insert_absent(
        &mut self,
        id: u32,
        rows: Rows<'_>,
        inv_scale: &[f64],
        alpha: f64,
    ) -> Result<(), ENNError> {
        self.insert_row(id, rows, inv_scale, alpha, true)
    }

    fn insert_row(
        &mut self,
        id: u32,
        rows: Rows<'_>,
        inv_scale: &[f64],
        alpha: f64,
        skip_present: bool,
    ) -> Result<(), ENNError> {
        let x = row(rows, id)?;
        let (mut path, leaf) = self.descend_from(self.nodes.root(), &x, inv_scale, rows)?;
        self.stats.route_rows += if self.split.rule == SplitRule::Pivot {
            2 * path.len() as u64
        } else {
            0
        };
        let shared = self.job.as_ref().map_or(0, |j| j.shared(&path));
        let frozen = self.job.as_ref().is_some_and(|j| j.freezes(shared));
        let placed = self.place(id, leaf, rows, inv_scale, skip_present, !frozen)?;
        if placed == Placed::Present {
            return Ok(());
        }
        if let Some(job) = self.job.as_mut() {
            job.log(id, shared);
        }
        if let Placed::Split(made) = placed {
            path.push(leaf);
            let deep = path.len() + made > depth_limit(id as usize + 1, self.nodes.cap(), alpha);
            if deep && self.job.as_ref().is_some_and(|j| j.covers(shared)) {
                self.stats.skipped += 1;
            } else if deep {
                self.rebalance_any(&path, id, rows, inv_scale, alpha)?;
            }
        }
        self.advance(rows)
    }

    fn rebalance_any(
        &mut self,
        path: &[u32],
        id: u32,
        rows: Rows<'_>,
        inv_scale: &[f64],
        alpha: f64,
    ) -> Result<(), ENNError> {
        if self.split.defer_rows == 0 {
            return self.rebalance(path, rows, inv_scale, alpha);
        }
        self.rebalance_deferred(path, id, rows, inv_scale, alpha)
    }

    pub(super) fn ensure_lrows(&mut self, dim: usize) -> Result<(), ENNError> {
        if self.lrows.is_none() && super::forest_lrows::clustered() && dim > 0 {
            self.lrows = Some(LeafRows::open(&self.path, self.nodes.cap(), dim, false)?);
        }
        if self.lrows.as_ref().is_some_and(|l| !l.fits(dim)) {
            self.lrows = None;
        }
        Ok(())
    }

    pub(super) fn near_rows(
        &mut self,
        pages: &[u32],
        ids: &[u32],
    ) -> Result<Option<NearRows>, ENNError> {
        let Some(l) = &self.lrows else {
            return Ok(None);
        };
        let mut near = NearRows::new();
        for &page in pages {
            self.stats.near_hits += l.get(page, Some(ids), &mut near)? as u64;
        }
        Ok(Some(near))
    }

    pub(crate) fn chain_near(&self, leaf: u32) -> Result<Option<NearRows>, ENNError> {
        let Some(l) = self.lrows.as_ref().filter(|_| super::forest_lrows::clustered()) else {
            return Ok(None);
        };
        let (mut near, mut at) = (NearRows::new(), leaf);
        while at != NIL {
            l.get(at, None, &mut near)?;
            at = match self.nodes.read(at)? {
                Node::Leaf { next, .. } => next,
                _ => return Err(io_error("forest leaf chain is corrupt")),
            };
        }
        Ok(Some(near))
    }

    pub(super) fn subtree_near(&mut self, slot: u32) -> Result<Option<NearRows>, ENNError> {
        if self.lrows.is_none() || !super::forest_lrows::clustered() {
            return Ok(None);
        }
        let (mut stack, mut pages) = (vec![slot], Vec::new());
        while let Some(at) = stack.pop() {
            match self.nodes.read(at)? {
                Node::Leaf { next, .. } => {
                    pages.push(at);
                    stack.extend((next != NIL).then_some(next));
                }
                Node::Split { left, right, .. } => stack.extend([left, right]),
                Node::Free { .. } => return Err(io_error("forest tree reaches a free node")),
            }
        }
        let mut near = NearRows::new();
        if let Some(l) = &self.lrows {
            for page in pages {
                self.stats.near_hits += l.get(page, None, &mut near)? as u64;
            }
        }
        Ok(Some(near))
    }

    pub(crate) fn effective_split(&self, dim: usize) -> SplitConfig {
        if self.nodes.holds(self.split.rule, dim) {
            self.split
        } else {
            SplitConfig {
                rule: SplitRule::Pivot,
                ..self.split
            }
        }
    }

    pub(crate) fn walk(&self, mut visit: impl FnMut(LeafVisit<'_>)) -> Result<(), ENNError> {
        let mut stack = vec![(self.nodes.root(), 0usize)];
        let mut ids = Vec::new();
        while let Some((at, depth)) = stack.pop() {
            match self.nodes.read(at)? {
                Node::Split { left, right, .. } => {
                    stack.extend([(left, depth + 1), (right, depth + 1)])
                }
                Node::Leaf { .. } => {
                    ids.clear();
                    let pages = self.leaf_ids(at, &mut ids)?;
                    visit(LeafVisit {
                        depth,
                        pages,
                        ids: &ids,
                    });
                }
                Node::Free { .. } => return Err(io_error("forest tree reaches a free node")),
            }
        }
        Ok(())
    }
}
