use super::forest_defer::{Job, Phase};
use super::forest_extern::{Part, Sample, Scratch, Task, CHUNK};
use super::forest_node::{Cut, Node};
use super::forest_plan::child_key;
use super::forest_split::{choose_split, Projector};
use super::forest_tree::Tree;
use super::lsh::io_error;
use super::row_order::Rows;
use crate::error::ENNError;

pub(crate) fn levels(n: usize, cap: usize) -> usize {
    (usize::BITS - (n / cap.max(1)).leading_zeros()).max(1) as usize
}

const LOAD_GAP: usize = 100 << 10;

impl Tree {
    pub(crate) fn load_step(&mut self, job: &mut Job, rows: Rows<'_>) -> Result<usize, ENNError> {
        let d = rows.x.ncols;
        if job.scratch.is_none() {
            job.scratch = Some(Scratch::create(
                &self.dir,
                self.seed ^ u64::from(job.anc[job.k]),
                d,
            )?);
        }
        let scratch = job
            .scratch
            .as_ref()
            .ok_or_else(|| io_error("forest job scratch is missing"))?;
        let Phase::Load { ids, at, sample } = &mut job.phase else {
            return Ok(1);
        };
        let hi = ids.len().min(*at + CHUNK);
        let mut xs = Vec::with_capacity((hi - *at) * d);
        rows.x.pread_runs(&ids[*at..hi], LOAD_GAP, &mut xs)?;
        for (&id, x) in ids[*at..hi].iter().zip(xs.chunks_exact(d)) {
            sample.offer(id, x, self.seed);
        }
        scratch.write(0, *at, &ids[*at..hi], &xs)?;
        let cost = hi - *at;
        *at = hi;
        if hi == ids.len() {
            let root = self.nodes.alloc()?;
            let task = Task {
                slot: root,
                file: 0,
                off: 0,
                n: ids.len(),
                key: job.anc[job.k],
                sample: std::mem::take(sample),
                part: None,
            };
            job.root = root;
            job.phase = Phase::Build(vec![task]);
        }
        Ok(cost.max(1))
    }

    pub(crate) fn build_step(&mut self, job: &mut Job, rows: Rows<'_>) -> Result<usize, ENNError> {
        let limit = self.local_rows(rows.x.ncols);
        let scratch = job
            .scratch
            .as_ref()
            .ok_or_else(|| io_error("forest job scratch is missing"))?;
        let Phase::Build(tasks) = &mut job.phase else {
            return Ok(1);
        };
        let Some(mut task) = tasks.pop() else {
            job.phase = Phase::Replay(0);
            return Ok(1);
        };
        if task.n <= limit && task.part.is_none() {
            let (ids, xs) = scratch.read(task.file, task.off, task.n)?;
            let plan = self.plan_local(task.key, &ids, &xs, &job.inv)?;
            self.write_plan(task.slot, plan)?;
            self.seed_leaves(task.slot, &ids, &xs, rows.x.ncols)?;
            return Ok(task.n * levels(task.n, self.nodes.cap()));
        }
        let projector = Projector {
            tree_seed: self.seed,
            inv_scale: &job.inv,
        };
        let Some(mut part) = task.part.take() else {
            let cost = task.sample.ids.len();
            task.part = self.choose_part(&task.sample, task.key, &projector, rows.x.ncols);
            if task.part.is_none() {
                let (ids, xs) = scratch.read(task.file, task.off, task.n)?;
                self.write_chain(task.slot, ids.clone())?;
                self.seed_leaves(task.slot, &ids, &xs, rows.x.ncols)?;
            } else {
                tasks.push(task);
            }
            return Ok(cost.max(1));
        };
        let cost = part.step(scratch, &task, &projector)?;
        if part.done < task.n {
            task.part = Some(part);
            tasks.push(task);
            return Ok(cost);
        }
        if part.nl == 0 || part.nr == 0 {
            let (ids, xs) = scratch.read(1 - task.file, task.off, task.n)?;
            self.write_chain(task.slot, ids.clone())?;
            self.seed_leaves(task.slot, &ids, &xs, rows.x.ncols)?;
            return Ok(cost);
        }
        let (l, r) = (self.nodes.alloc()?, self.nodes.alloc()?);
        self.nodes.write(
            task.slot,
            &Node::Split {
                cut: part.cut,
                thr: part.thr,
                left: l,
                right: r,
            },
        )?;
        self.stats.splits += 1;
        let [ls, rs] = part.samples;
        let file = 1 - task.file;
        tasks.push(Task {
            slot: r,
            file,
            off: task.off + part.nl,
            n: part.nr,
            key: child_key(task.key, 1),
            sample: rs,
            part: None,
        });
        tasks.push(Task {
            slot: l,
            file,
            off: task.off,
            n: part.nl,
            key: child_key(task.key, 0),
            sample: ls,
            part: None,
        });
        Ok(cost)
    }

    fn choose_part(
        &self,
        sample: &Sample,
        key: u32,
        projector: &Projector<'_>,
        d: usize,
    ) -> Option<Part> {
        let xs: Vec<&[f64]> = sample.xs.chunks_exact(d).collect();
        let (cut, thr, _) = choose_split(
            projector,
            &xs,
            &sample.ids,
            key,
            self.seed,
            self.effective_split(d),
        )?;
        let pivots = match cut {
            Cut::Pivot(a, b) => Some((sample.row(a, d)?.to_vec(), sample.row(b, d)?.to_vec())),
            _ => None,
        };
        Some(Part {
            cut,
            thr,
            pivots,
            done: 0,
            nl: 0,
            nr: 0,
            samples: [Sample::default(), Sample::default()],
        })
    }
}
