use std::sync::atomic::{AtomicU64, Ordering};

use super::forest_node::{Cut, Node, NIL};
use super::forest_rules::{SplitConfig, SplitRule};
use super::forest_split::{choose_split, Projector};
use super::forest_tree::{row, Tree};
use crate::backend::MmapColumnStore;
use crate::error::ENNError;

pub(crate) enum Plan {
    Leaf(Vec<u32>),
    Split { cut: Cut, thr: f64, left: Box<Plan>, right: Box<Plan> },
}

struct Planner<'a> {
    projector: Projector<'a>,
    rows: &'a MmapColumnStore,
    cap: usize,
    salt: u64,
    split: SplitConfig,
    par_above: usize,
    local_rows: usize,
    gathered: AtomicU64,
}

fn child_key(key: u32, side: u32) -> u32 {
    key.wrapping_mul(0x9e37_79b1).rotate_left(7).wrapping_add(1 + side)
}

impl Planner<'_> {
    fn plan(&self, mut ids: Vec<u32>, key: u32) -> Result<Plan, ENNError> {
        if ids.len() <= self.cap {
            return Ok(Plan::Leaf(ids));
        }
        if ids.len() <= self.local_rows {
            let buf = self.load(&ids)?;
            let xs: Vec<&[f64]> = buf.chunks_exact(self.rows.ncols).collect();
            return Ok(self.plan_local(ids, xs, key));
        }
        self.gathered.fetch_add(ids.len() as u64, Ordering::Relaxed);
        let xs = ids.iter().map(|&i| row(self.rows, i)).collect::<Result<Vec<_>, _>>()?;
        let Some((cut, thr, left)) = choose_split(&self.projector, &xs, &ids, key, self.salt, self.split) else {
            return Ok(Plan::Leaf(ids));
        };
        drop(xs);
        let n = ids.len();
        let (mut l_ids, mut r_ids) = (Vec::new(), Vec::new());
        for (id, is_left) in ids.drain(..).zip(left) {
            if is_left { l_ids.push(id) } else { r_ids.push(id) }
        }
        let (kl, kr) = (child_key(key, 0), child_key(key, 1));
        let (l, r) = if n >= self.par_above {
            rayon::join(|| self.plan(l_ids, kl), || self.plan(r_ids, kr))
        } else {
            (self.plan(l_ids, kl), self.plan(r_ids, kr))
        };
        Ok(Plan::Split { cut, thr, left: Box::new(l?), right: Box::new(r?) })
    }

    fn load(&self, ids: &[u32]) -> Result<Vec<f64>, ENNError> {
        let d = self.rows.ncols;
        let mut buf = vec![0.0; ids.len() * d];
        let mut order: Vec<usize> = (0..ids.len()).collect();
        order.sort_unstable_by_key(|&p| ids[p]);
        for p in order {
            let dst = &mut buf[p * d..(p + 1) * d];
            if self.split.pread & 1 == 1 {
                dst.copy_from_slice(&self.rows.pread_row(ids[p] as usize)?);
            } else {
                dst.copy_from_slice(row(self.rows, ids[p])?);
            }
        }
        self.gathered.fetch_add(ids.len() as u64, Ordering::Relaxed);
        Ok(buf)
    }

    fn plan_local(&self, ids: Vec<u32>, xs: Vec<&[f64]>, key: u32) -> Plan {
        if ids.len() <= self.cap {
            return Plan::Leaf(ids);
        }
        let Some((cut, thr, left)) = choose_split(&self.projector, &xs, &ids, key, self.salt, self.split) else {
            return Plan::Leaf(ids);
        };
        let (mut l, mut r) = ((Vec::new(), Vec::new()), (Vec::new(), Vec::new()));
        for ((id, x), is_left) in ids.into_iter().zip(xs).zip(left) {
            let side = if is_left { &mut l } else { &mut r };
            side.0.push(id);
            side.1.push(x);
        }
        let (pl, pr) = (self.plan_local(l.0, l.1, child_key(key, 0)), self.plan_local(r.0, r.1, child_key(key, 1)));
        Plan::Split { cut, thr, left: Box::new(pl), right: Box::new(pr) }
    }
}

pub(crate) fn local_rows(kib: u32, dim: usize) -> usize {
    (kib as usize * 1024) / (8 * dim.max(1))
}

impl Tree {
    pub(crate) fn plan_subtree(&mut self, slot: u32, ids: Vec<u32>, rows: &MmapColumnStore, inv_scale: &[f64]) -> Result<Plan, ENNError> {
        let split = if self.nodes.holds(self.split.rule, rows.ncols) { self.split } else { SplitConfig { rule: SplitRule::Pivot, ..self.split } };
        let planner = Planner {
            projector: Projector { tree_seed: self.seed, inv_scale },
            rows,
            cap: self.nodes.cap(),
            salt: self.seed,
            split,
            par_above: if self.split.par_above == 0 { usize::MAX } else { (self.split.par_above as usize).max(2 * self.nodes.cap()) },
            local_rows: if rows.ncols == 0 { 0 } else { local_rows(split.local_kib, rows.ncols) },
            gathered: AtomicU64::new(0),
        };
        let plan = planner.plan(ids, slot);
        self.stats.gathered += planner.gathered.into_inner();
        plan
    }

    pub(crate) fn write_plan(&mut self, slot: u32, plan: Plan) -> Result<usize, ENNError> {
        match plan {
            Plan::Leaf(ids) if ids.len() <= self.nodes.cap() => {
                self.nodes.write(slot, &Node::Leaf { ids, next: NIL })?;
                Ok(0)
            }
            Plan::Leaf(ids) => self.write_chain(slot, ids),
            Plan::Split { cut, thr, left, right } => {
                let (l, r) = (self.nodes.alloc()?, self.nodes.alloc()?);
                let dl = self.write_plan(l, *left)?;
                let dr = self.write_plan(r, *right)?;
                self.nodes.write(slot, &Node::Split { cut, thr, left: l, right: r })?;
                self.stats.splits += 1;
                Ok(1 + dl.max(dr))
            }
        }
    }
}
