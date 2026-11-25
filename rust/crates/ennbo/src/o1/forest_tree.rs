use std::path::Path;

use super::cache::ObjCache;
use super::forest_node::{Cut, Node, NodeFile, NIL};
use super::forest_rules::{SplitConfig, SplitRule};
use super::forest_split::{choose_split, Parted, Placed, Projector};
use super::lsh::io_error;
use super::row_order::Rows;
use super::search::{fetch_row, RowRef, X_COLUMN};
use crate::backend::MmapColumnStore;
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
    }
}

pub(crate) struct Tree {
    pub(crate) nodes: NodeFile,
    pub(crate) seed: u64,
    pub(crate) stats: TreeStats,
    pub(crate) split: SplitConfig,
    pub(crate) packed_at: u32,
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

pub(super) fn row(rows: &MmapColumnStore, id: u32) -> Result<&[f64], ENNError> {
    rows.mmap_row_slice(id as usize)
}

pub(crate) fn depth_limit(n: usize, cap: usize, alpha: f64) -> usize {
    let ratio = (n.max(cap) as f64) / (cap as f64);
    (ratio.ln() / (1.0 / alpha).ln()).ceil() as usize + 1
}

impl Tree {
    pub(crate) fn create(path: &Path, cap: usize, seed: u64, split: SplitConfig, dim: usize) -> Result<Self, ENNError> {
        Ok(Self { nodes: NodeFile::create(path, cap, dim, split.rule.payload(dim))?, seed, stats: TreeStats::default(), split, packed_at: 0 })
    }

    pub(crate) fn open(path: &Path, cap: usize, seed: u64) -> Result<Self, ENNError> {
        let nodes = NodeFile::open(path, cap)?;
        let packed_at = nodes.num_nodes();
        Ok(Self { nodes, seed, stats: TreeStats::default(), split: SplitConfig::default(), packed_at })
    }

    fn projector<'a>(&self, inv_scale: &'a [f64]) -> Projector<'a> {
        Projector { tree_seed: self.seed, inv_scale }
    }

    pub(crate) fn branch(&self, x: &[f64], inv_scale: &[f64], split: &SplitRef, rows: RowSource<'_>, weights: Option<&[f64]>) -> Result<(f64, f64, u32, u32), ENNError> {
        let projector = self.projector(inv_scale);
        let (p, norm) = match split.cut {
            Cut::Seed(seed) => (projector.project(x, seed), weights.map(|w| projector.seed_dual_norm(w))),
            Cut::Pivot(a, b) => {
                let get = |i: u32| match (rows.pread && !rows.store.covers(i), rows.cache) {
                    (true, None) => rows.store.pread(i as usize).map(|r| RowRef::Shared(r.into())),
                    _ => fetch_row(rows.store, rows.cache, X_COLUMN, i as usize),
                };
                let (ra, rb) = (get(a)?, get(b)?);
                (projector.project_pivot(x, &ra, &rb), weights.map(|w| projector.pivot_dual_norm(&ra, &rb, w)))
            }
            Cut::Dir(ref dir) => (dir.dot(x), weights.map(|w| super::forest_bound::dual_norm(dir, w))),
        };
        let raw = p - split.thr;
        let bound = match norm {
            Some(n) if n > 0.0 && n.is_finite() => raw.abs() / n,
            Some(_) => 0.0,
            None => raw.abs(),
        };
        Ok(if raw < 0.0 { (-raw, bound, split.left, split.right) } else { (raw, bound, split.right, split.left) })
    }

    pub(crate) fn step(&self, x: &[f64], inv_scale: &[f64], node: u32, rows: RowSource<'_>) -> Result<Option<(f64, u32, u32)>, ENNError> {
        match self.nodes.read(node)? {
            Node::Split { cut, thr, left, right } => self.branch(x, inv_scale, &SplitRef { cut, thr, left, right }, rows, None).map(|(m, _, near, far)| Some((m, near, far))),
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

    pub(super) fn descend_from(&self, start: u32, x: &[f64], inv_scale: &[f64], rows: &MmapColumnStore) -> Result<(Vec<u32>, u32), ENNError> {
        let mut path = Vec::new();
        let mut at = start;
        let src = RowSource { store: Rows::plain(rows), cache: None, pread: false };
        while let Some((_, near, _)) = self.step(x, inv_scale, at, src)? {
            path.push(at);
            at = near;
        }
        Ok((path, at))
    }

    pub(crate) fn insert(&mut self, id: u32, rows: &MmapColumnStore, inv_scale: &[f64], alpha: f64) -> Result<(), ENNError> {
        self.insert_row(id, rows, inv_scale, alpha, false)
    }

    pub(crate) fn insert_absent(&mut self, id: u32, rows: &MmapColumnStore, inv_scale: &[f64], alpha: f64) -> Result<(), ENNError> {
        self.insert_row(id, rows, inv_scale, alpha, true)
    }

    fn insert_row(&mut self, id: u32, rows: &MmapColumnStore, inv_scale: &[f64], alpha: f64, skip_present: bool) -> Result<(), ENNError> {
        let x = row(rows, id)?;
        let (mut path, leaf) = self.descend_from(self.nodes.root(), x, inv_scale, rows)?;
        self.stats.route_rows += if self.split.rule == SplitRule::Pivot { 2 * path.len() as u64 } else { 0 };
        let placed = self.place(id, leaf, rows, inv_scale, skip_present)?;
        if placed == Placed::Present {
            return Ok(());
        }
        let Placed::Split(made) = placed else { return Ok(()) };
        path.push(leaf);
        if path.len() + made > depth_limit(id as usize + 1, self.nodes.cap(), alpha) {
            self.rebalance(&path, rows, inv_scale, alpha)?;
        }
        Ok(())
    }

    pub(super) fn place(&mut self, id: u32, leaf: u32, rows: &MmapColumnStore, inv_scale: &[f64], skip_present: bool) -> Result<Placed, ENNError> {
        let mut pages = Vec::new();
        let mut at = leaf;
        let mut last = Vec::new();
        while at != NIL {
            let Node::Leaf { ids, next } = self.nodes.read(at)? else {
                return Err(io_error("forest leaf chain is corrupt"));
            };
            if skip_present && ids.contains(&id) {
                return Ok(Placed::Present);
            }
            pages.push(at);
            (last, at) = (ids, next);
        }
        let tail = *pages.last().ok_or_else(|| io_error("forest leaf is missing"))?;
        if last.len() < self.nodes.cap() {
            last.push(id);
            self.nodes.write(tail, &Node::Leaf { ids: last, next: NIL })?;
            return Ok(Placed::Appended);
        }
        if pages.len() > 1 && !pages.len().is_power_of_two() {
            let page = self.nodes.alloc()?;
            self.nodes.write(page, &Node::Leaf { ids: vec![id], next: NIL })?;
            self.stats.chain_pages += 1;
            self.nodes.write(tail, &Node::Leaf { ids: last, next: page })?;
            return Ok(Placed::Appended);
        }
        let mut ids = Vec::new();
        self.leaf_ids(leaf, &mut ids)?;
        ids.push(id);
        let made = self.build(leaf, ids, rows, inv_scale)?;
        self.nodes.free(&pages[1..])?;
        Ok(Placed::Split(made))
    }

    pub(super) fn part(&mut self, slot: u32, mut ids: Vec<u32>, rows: &MmapColumnStore, inv_scale: &[f64]) -> Result<Option<Parted>, ENNError> {
        if ids.len() <= self.nodes.cap() {
            self.nodes.write(slot, &Node::Leaf { ids, next: NIL })?;
            return Ok(None);
        }
        self.stats.gathered += ids.len() as u64;
        let xs = ids.iter().map(|&i| row(rows, i)).collect::<Result<Vec<_>, _>>()?;
        let split = if self.nodes.holds(self.split.rule, rows.ncols) { self.split } else { SplitConfig { rule: SplitRule::Pivot, ..self.split } };
        let Some((cut, thr, left)) = choose_split(&self.projector(inv_scale), &xs, &ids, slot, self.seed, split) else {
            self.write_chain(slot, ids)?;
            return Ok(None);
        };
        drop(xs);
        let (mut l_ids, mut r_ids) = (Vec::new(), Vec::new());
        for (id, is_left) in ids.drain(..).zip(left) {
            if is_left { l_ids.push(id) } else { r_ids.push(id) }
        }
        let (l, r) = (self.nodes.alloc()?, self.nodes.alloc()?);
        Ok(Some(Parted { node: Node::Split { cut, thr, left: l, right: r }, left: (l, l_ids), right: (r, r_ids) }))
    }

    pub(crate) fn build(&mut self, slot: u32, ids: Vec<u32>, rows: &MmapColumnStore, inv_scale: &[f64]) -> Result<usize, ENNError> {
        let Some(p) = self.part(slot, ids, rows, inv_scale)? else { return Ok(0) };
        let dl = self.build(p.left.0, p.left.1, rows, inv_scale)?;
        let dr = self.build(p.right.0, p.right.1, rows, inv_scale)?;
        self.nodes.write(slot, &p.node)?;
        self.stats.splits += 1;
        Ok(1 + dl.max(dr))
    }

    pub(super) fn write_chain(&mut self, slot: u32, ids: Vec<u32>) -> Result<usize, ENNError> {
        let chunks: Vec<Vec<u32>> = ids.chunks(self.nodes.cap()).map(<[u32]>::to_vec).collect();
        let mut next = NIL;
        for chunk in chunks[1..].iter().rev() {
            let page = self.nodes.alloc()?;
            self.nodes.write(page, &Node::Leaf { ids: chunk.clone(), next })?;
            next = page;
            self.stats.chain_pages += 1;
        }
        self.nodes.write(slot, &Node::Leaf { ids: chunks[0].clone(), next })?;
        Ok(0)
    }

    pub(super) fn gather(&self, node: u32, ids: &mut Vec<u32>, nodes: &mut Vec<u32>) -> Result<(), ENNError> {
        let mut stack = vec![node];
        while let Some(at) = stack.pop() {
            nodes.push(at);
            match self.nodes.read(at)? {
                Node::Leaf { ids: page, next } => {
                    ids.extend_from_slice(&page);
                    stack.extend((next != NIL).then_some(next));
                }
                Node::Split { left, right, .. } => stack.extend([left, right]),
                Node::Free { .. } => return Err(io_error("forest tree reaches a free node")),
            }
        }
        Ok(())
    }

    fn count(&self, node: u32) -> Result<usize, ENNError> {
        let (mut stack, mut n) = (vec![node], 0);
        while let Some(at) = stack.pop() {
            match self.nodes.read(at)? {
                Node::Leaf { ids, next } => {
                    n += ids.len();
                    stack.extend((next != NIL).then_some(next));
                }
                Node::Split { left, right, .. } => stack.extend([left, right]),
                Node::Free { .. } => return Err(io_error("forest tree reaches a free node")),
            }
        }
        Ok(n)
    }

    fn rebalance(&mut self, path: &[u32], rows: &MmapColumnStore, inv_scale: &[f64], alpha: f64) -> Result<(), ENNError> {
        let mut below = self.count(path[path.len() - 1])?;
        for (i, w) in path.windows(2).enumerate().rev() {
            let Node::Split { left, right, .. } = self.nodes.read(w[0])? else {
                return Err(io_error("forest path is corrupt"));
            };
            let sibling = if left == w[1] { right } else { left };
            let size = below + self.count(sibling)?;
            if below.max(size - below) as f64 > alpha * size as f64 {
                return self.rebuild(path[i], rows, inv_scale);
            }
            below = size;
        }
        Ok(())
    }

    pub(crate) fn rebuild(&mut self, slot: u32, rows: &MmapColumnStore, inv_scale: &[f64]) -> Result<(), ENNError> {
        let (mut ids, mut old) = (Vec::new(), Vec::new());
        self.gather(slot, &mut ids, &mut old)?;
        let n = ids.len() as u64;
        let before = self.stats.gathered;
        if self.split.par_above > 0 && ids.len() >= self.split.par_above as usize {
            let plan = self.plan_subtree(slot, ids, rows, inv_scale)?;
            self.write_plan(slot, plan)?;
        } else {
            self.build(slot, ids, rows, inv_scale)?;
        }
        self.nodes.free(&old[1..])?;
        self.stats.rebuild_gathered += self.stats.gathered - before;
        self.stats.rebuilds += 1;
        self.stats.rebuilt_rows += n;
        self.stats.max_rebuild_rows = self.stats.max_rebuild_rows.max(n);
        Ok(())
    }

    pub(crate) fn walk(&self, mut visit: impl FnMut(LeafVisit<'_>)) -> Result<(), ENNError> {
        let mut stack = vec![(self.nodes.root(), 0usize)];
        let mut ids = Vec::new();
        while let Some((at, depth)) = stack.pop() {
            match self.nodes.read(at)? {
                Node::Split { left, right, .. } => stack.extend([(left, depth + 1), (right, depth + 1)]),
                Node::Leaf { .. } => {
                    ids.clear();
                    let pages = self.leaf_ids(at, &mut ids)?;
                    visit(LeafVisit { depth, pages, ids: &ids });
                }
                Node::Free { .. } => return Err(io_error("forest tree reaches a free node")),
            }
        }
        Ok(())
    }
}
