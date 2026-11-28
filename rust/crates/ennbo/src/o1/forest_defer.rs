use super::forest_extern::{Sample, Scratch, Task, CHUNK};
use super::forest_jobbuild::levels;
use super::forest_node::NIL;
use super::forest_pack::children;
use super::forest_scan::{Scan, ScanStep};
use super::forest_split::Placed;
use super::forest_tree::{depth_limit, row, Tree};
use super::row_order::Rows;
use crate::error::ENNError;

pub(crate) const REPLAY_COST: usize = 32;
pub(crate) const NODE_COST: usize = 32;

pub(crate) enum Phase {
    Scan(Scan),
    Load {
        ids: Vec<u32>,
        at: usize,
        sample: Sample,
    },
    Build(Vec<Task>),
    Replay(usize),
    Free(Vec<u32>),
}

pub(crate) struct Job {
    pub anc: Vec<u32>,
    pub k: usize,
    pub late: Vec<(u32, usize)>,
    pub credit: i64,
    pub phase: Phase,
    pub inv: Vec<f64>,
    pub alpha: f64,
    pub scratch: Option<Scratch>,
    pub root: u32,
    pub rows: usize,
    pub dedupe: bool,
}

impl Job {
    fn new(anc: &[u32], scan: Scan, inv: &[f64], alpha: f64) -> Self {
        let dedupe = scan.start == u32::MAX;
        Self {
            anc: anc.to_vec(),
            k: 0,
            late: Vec::new(),
            credit: 0,
            phase: Phase::Scan(scan),
            inv: inv.to_vec(),
            alpha,
            scratch: None,
            root: NIL,
            rows: 0,
            dedupe,
        }
    }

    fn bulk(root: u32, n: usize, inv: &[f64], alpha: f64) -> Self {
        let mut job = Self::new(&[root], Scan::new(&[root], 0), inv, alpha);
        job.phase = Phase::Load {
            ids: (0..n as u32).collect(),
            at: 0,
            sample: Sample::default(),
        };
        job.rows = n;
        job
    }

    pub(crate) fn shared(&self, path: &[u32]) -> usize {
        path.iter()
            .zip(&self.anc)
            .take_while(|(a, b)| a == b)
            .count()
    }

    pub(crate) fn freezes(&self, shared: usize) -> bool {
        matches!(&self.phase, Phase::Scan(s) if s.i >= 1 && shared >= s.i)
    }

    pub(crate) fn covers(&self, shared: usize) -> bool {
        !matches!(self.phase, Phase::Scan(_) | Phase::Free(_)) && shared > self.k
    }

    pub(crate) fn log(&mut self, id: u32, shared: usize) {
        match self.phase {
            Phase::Scan(_) => self.late.push((id, shared)),
            Phase::Free(_) => {}
            _ if shared > self.k => self.late.push((id, shared)),
            _ => {}
        }
    }

    fn phase_index(&self) -> usize {
        match self.phase {
            Phase::Scan(_) => 0,
            Phase::Load { .. } => 1,
            Phase::Build(_) => 2,
            Phase::Replay(_) => 3,
            Phase::Free(_) => 4,
        }
    }

    fn size(&self) -> usize {
        match &self.phase {
            Phase::Scan(s) => s.ids.len(),
            _ => self.rows,
        }
    }

    fn found(&mut self, k: usize) {
        let mut ids = match std::mem::replace(&mut self.phase, Phase::Free(Vec::new())) {
            Phase::Scan(s) => s.ids,
            _ => Vec::new(),
        };
        self.late.retain(|&(_, m)| m > k);
        if self.dedupe && !self.late.is_empty() {
            let replayed: std::collections::HashSet<u32> = self.late.iter().map(|l| l.0).collect();
            ids.retain(|id| !replayed.contains(id));
        }
        ids.sort_unstable();
        self.rows = ids.len();
        self.k = k;
        self.phase = Phase::Load {
            ids,
            at: 0,
            sample: Sample::default(),
        };
    }
}

impl Tree {
    pub(crate) fn local_rows(&self, dim: usize) -> usize {
        (self.split.local_bytes as usize / (4 + 8 * dim)).max(2 * self.nodes.cap())
    }

    fn rebalance_small(
        &mut self,
        path: &[u32],
        id: u32,
        rows: Rows<'_>,
        inv_scale: &[f64],
        alpha: f64,
    ) -> Result<Option<(Scan, ScanStep)>, ENNError> {
        let limit = self.local_rows(rows.x.ncols);
        let start = if self.mixed { u32::MAX } else { id + 1 };
        let mut scan = Scan::new(path, start);
        let (_, step) = self.scan(&mut scan, path, alpha, usize::MAX, limit)?;
        match step {
            ScanStep::Found(k) if scan.ids.len() <= limit => self
                .rebuild_local(path[k], scan.ids, rows, inv_scale)
                .map(|()| None),
            ScanStep::Balanced | ScanStep::Broken => Ok(None),
            step => Ok(Some((scan, step))),
        }
    }

    pub(crate) fn rebalance_deferred(
        &mut self,
        path: &[u32],
        id: u32,
        rows: Rows<'_>,
        inv_scale: &[f64],
        alpha: f64,
    ) -> Result<(), ENNError> {
        let Some((scan, step)) = self.rebalance_small(path, id, rows, inv_scale, alpha)? else {
            return Ok(());
        };
        if self.job.is_some() {
            self.stats.skipped += 1;
            return Ok(());
        }
        self.stats.deferred += 1;
        let mut job = Job::new(path, scan, inv_scale, alpha);
        if let ScanStep::Found(k) = step {
            job.found(k);
        }
        self.job = Some(Box::new(job));
        Ok(())
    }

    pub(crate) fn rebuild_local(
        &mut self,
        slot: u32,
        mut ids: Vec<u32>,
        rows: Rows<'_>,
        inv_scale: &[f64],
    ) -> Result<(), ENNError> {
        ids.sort_unstable();
        let near = self.subtree_near(slot)?;
        let rows = Rows {
            near: near.as_ref().or(rows.near),
            ..rows
        };
        let mut buf = Vec::with_capacity(ids.len() * rows.x.ncols);
        for &i in &ids {
            buf.extend_from_slice(&row(rows, i)?);
        }
        let before = self.stats.gathered;
        let plan = self.plan_local(slot, &ids, &buf, inv_scale)?;
        let old = self.nodes.read(slot)?;
        self.write_plan(slot, plan)?;
        self.seed_leaves(slot, &ids, &buf, rows.x.ncols)?;
        self.free_some(&mut children(&old), usize::MAX)?;
        self.stats.rebuild_gathered += self.stats.gathered - before;
        self.note_rebuild(ids.len());
        Ok(())
    }

    pub(crate) fn note_rebuild(&mut self, n: usize) {
        self.stats.rebuilds += 1;
        self.stats.rebuilt_rows += n as u64;
        self.stats.max_rebuild_rows = self.stats.max_rebuild_rows.max(n as u64);
    }

    pub(crate) fn start_bulk(&mut self, n: usize, inv_scale: &[f64], alpha: f64) {
        self.stats.deferred += 1;
        self.job = Some(Box::new(Job::bulk(self.nodes.root(), n, inv_scale, alpha)));
    }

    pub(crate) fn bulk_rows(&self) -> Option<usize> {
        self.job.as_ref().map(|j| j.rows)
    }

    pub(crate) fn advance(&mut self, rows: Rows<'_>) -> Result<(), ENNError> {
        let Some(size) = self.job.as_ref().map(|j| j.size()) else {
            return Ok(());
        };
        self.drive(
            rows,
            self.split.defer_rows as usize * levels(size, self.nodes.cap()),
        )
        .map(|_| ())
    }

    pub(crate) fn drive(&mut self, rows: Rows<'_>, credit: usize) -> Result<bool, ENNError> {
        let Some(mut job) = self.job.take() else {
            return Ok(true);
        };
        job.credit += credit as i64;
        while job.credit > 0 {
            let (phase, t0) = (job.phase_index(), std::time::Instant::now());
            let m0 = super::forest_order::diag_marks();
            let step = self.job_step(&mut job, rows)?;
            self.stats.phase_ns[phase] += t0.elapsed().as_nanos() as u64;
            let m1 = super::forest_order::diag_marks();
            self.stats.phase_flt[phase] += m1[0].saturating_sub(m0[0]);
            self.stats.phase_rd[phase] += m1[1].saturating_sub(m0[1]);
            match step {
                Some(cost) => job.credit -= cost as i64,
                None => return Ok(true),
            }
        }
        self.job = Some(job);
        Ok(false)
    }

    fn job_step(&mut self, job: &mut Job, rows: Rows<'_>) -> Result<Option<usize>, ENNError> {
        match &mut job.phase {
            Phase::Scan(s) => {
                let (nodes, step) =
                    self.scan(s, &job.anc, job.alpha, CHUNK / NODE_COST, usize::MAX)?;
                let cost = (nodes * NODE_COST).max(1);
                match step {
                    ScanStep::Busy => Ok(Some(cost)),
                    ScanStep::Found(k) => {
                        job.found(k);
                        Ok(Some(cost))
                    }
                    ScanStep::Balanced | ScanStep::Broken => {
                        self.stats.aborted += 1;
                        Ok(None)
                    }
                }
            }
            Phase::Load { .. } => self.load_step(job, rows).map(Some),
            Phase::Build(_) => self.build_step(job, rows).map(Some),
            Phase::Replay(_) => self.replay_step(job, rows).map(Some),
            Phase::Free(stack) if stack.is_empty() => Ok(None),
            Phase::Free(stack) => self
                .free_some(stack, CHUNK / NODE_COST)
                .map(|n| Some(n * NODE_COST)),
        }
    }

    fn replay_step(&mut self, job: &mut Job, rows: Rows<'_>) -> Result<usize, ENNError> {
        let Phase::Replay(at) = job.phase else {
            return Ok(1);
        };
        let hi = job.late.len().min(at + CHUNK / REPLAY_COST);
        for k in at..hi {
            let id = job.late[k].0;
            let (mut path, leaf) = self.descend_from(job.root, &row(rows, id)?, &job.inv, rows)?;
            let Placed::Split(made) = self.place(id, leaf, rows, &job.inv, false, true)? else {
                continue;
            };
            path.push(leaf);
            if path.len() + made > depth_limit(job.rows + k + 1, self.nodes.cap(), job.alpha)
                && self
                    .rebalance_small(&path, id, rows, &job.inv, job.alpha)?
                    .is_some()
            {
                self.stats.skipped += 1;
            }
        }
        self.stats.late_rows += (hi - at) as u64;
        job.phase = Phase::Replay(hi);
        if hi == job.late.len() {
            self.swap(job)?;
        }
        Ok((REPLAY_COST * (hi - at)).max(1))
    }

    fn swap(&mut self, job: &mut Job) -> Result<(), ENNError> {
        let z = job.anc[job.k];
        let old = self.nodes.read(z)?;
        let fresh = self.nodes.read(job.root)?;
        self.nodes.write(z, &fresh)?;
        self.nodes.free(&[job.root])?;
        self.note_rebuild(job.rows + job.late.len());
        job.late = Vec::new();
        job.scratch = None;
        job.phase = Phase::Free(children(&old));
        Ok(())
    }
}
