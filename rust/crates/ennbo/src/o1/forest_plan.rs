use std::sync::atomic::{AtomicU64, Ordering};

use super::forest_node::{Cut, Node, NIL};
use super::forest_rules::SplitConfig;
use super::forest_split::{choose_split, Projector};
use super::forest_tree::{row, Tree};
use super::row_order::Rows;
use crate::error::ENNError;

pub(crate) enum Plan {
    Leaf(Vec<u32>),
    Split {
        cut: Cut,
        thr: f64,
        left: Box<Plan>,
        right: Box<Plan>,
    },
}

#[derive(Clone, Copy)]
pub(crate) enum PlanRows<'a> {
    Store(Rows<'a>),
    Local {
        ids: &'a [u32],
        buf: &'a [f64],
        dim: usize,
    },
}

impl PlanRows<'_> {
    fn x(&self, p: u32) -> Result<std::borrow::Cow<'_, [f64]>, ENNError> {
        match *self {
            Self::Store(rows) => row(rows, p),
            Self::Local { buf, dim, .. } => Ok(std::borrow::Cow::Borrowed(
                &buf[p as usize * dim..(p as usize + 1) * dim],
            )),
        }
    }

    fn id(&self, p: u32) -> u32 {
        match *self {
            Self::Store(_) => p,
            Self::Local { ids, .. } => ids[p as usize],
        }
    }

    fn ncols(&self) -> usize {
        match *self {
            Self::Store(rows) => rows.x.ncols,
            Self::Local { dim, .. } => dim,
        }
    }
}

struct Planner<'a> {
    projector: Projector<'a>,
    rows: PlanRows<'a>,
    cap: usize,
    salt: u64,
    split: SplitConfig,
    par_above: usize,
    gathered: AtomicU64,
}

pub(crate) fn child_key(key: u32, side: u32) -> u32 {
    key.wrapping_mul(0x9e37_79b1)
        .rotate_left(7)
        .wrapping_add(1 + side)
}

impl Planner<'_> {
    fn plan(&self, mut ids: Vec<u32>, key: u32) -> Result<Plan, ENNError> {
        if ids.len() <= self.cap {
            return Ok(Plan::Leaf(self.real(ids)));
        }
        self.gathered.fetch_add(ids.len() as u64, Ordering::Relaxed);
        let held = ids
            .iter()
            .map(|&i| self.rows.x(i))
            .collect::<Result<Vec<_>, _>>()?;
        let xs: Vec<&[f64]> = held.iter().map(|r| &r[..]).collect();
        let real: Vec<u32> = ids.iter().map(|&p| self.rows.id(p)).collect();
        let Some((cut, thr, left)) =
            choose_split(&self.projector, &xs, &real, key, self.salt, self.split)
        else {
            return Ok(Plan::Leaf(real));
        };
        drop(xs);
        drop(held);
        let n = ids.len();
        let (mut l_ids, mut r_ids) = (Vec::new(), Vec::new());
        for (id, is_left) in ids.drain(..).zip(left) {
            if is_left {
                l_ids.push(id)
            } else {
                r_ids.push(id)
            }
        }
        let (kl, kr) = (child_key(key, 0), child_key(key, 1));
        let (l, r) = if n >= self.par_above {
            rayon::join(|| self.plan(l_ids, kl), || self.plan(r_ids, kr))
        } else {
            (self.plan(l_ids, kl), self.plan(r_ids, kr))
        };
        Ok(Plan::Split {
            cut,
            thr,
            left: Box::new(l?),
            right: Box::new(r?),
        })
    }

    fn real(&self, ids: Vec<u32>) -> Vec<u32> {
        match self.rows {
            PlanRows::Store(_) => ids,
            PlanRows::Local { .. } => ids.into_iter().map(|p| self.rows.id(p)).collect(),
        }
    }
}

impl Tree {
    pub(crate) fn plan_subtree(
        &mut self,
        slot: u32,
        ids: Vec<u32>,
        rows: Rows<'_>,
        inv_scale: &[f64],
    ) -> Result<Plan, ENNError> {
        self.plan_rows(slot, ids, PlanRows::Store(rows), inv_scale)
    }

    pub(crate) fn plan_local(
        &mut self,
        key: u32,
        ids: &[u32],
        buf: &[f64],
        inv_scale: &[f64],
    ) -> Result<Plan, ENNError> {
        let dim = inv_scale.len();
        self.plan_rows(
            key,
            (0..ids.len() as u32).collect(),
            PlanRows::Local { ids, buf, dim },
            inv_scale,
        )
    }

    fn plan_rows(
        &mut self,
        slot: u32,
        ids: Vec<u32>,
        rows: PlanRows<'_>,
        inv_scale: &[f64],
    ) -> Result<Plan, ENNError> {
        let split = self.effective_split(rows.ncols());
        let planner = Planner {
            projector: Projector {
                tree_seed: self.seed,
                inv_scale,
            },
            rows,
            cap: self.nodes.cap(),
            salt: self.seed,
            split,
            par_above: if self.split.par_above == 0 {
                usize::MAX
            } else {
                (self.split.par_above as usize).max(2 * self.nodes.cap())
            },
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
            Plan::Split {
                cut,
                thr,
                left,
                right,
            } => {
                let (l, r) = (self.nodes.alloc()?, self.nodes.alloc()?);
                let dl = self.write_plan(l, *left)?;
                let dr = self.write_plan(r, *right)?;
                self.nodes.write(
                    slot,
                    &Node::Split {
                        cut,
                        thr,
                        left: l,
                        right: r,
                    },
                )?;
                self.stats.splits += 1;
                Ok(1 + dl.max(dr))
            }
        }
    }
}
