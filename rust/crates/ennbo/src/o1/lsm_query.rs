use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use super::disk_index::O1Occupancy;
use super::forest_query::Scored;
use super::lsm::LsmIndex;
use super::lsm_key::{box_dist2, metric_factor, wbox_dist2};
use super::lsm_run::{block_row, FANOUT};
use super::profile::QueryProfile;
use super::row_order::Rows;
use super::search::weighted_dist2;
use crate::error::ENNError;

#[derive(Clone, Copy)]
pub(crate) struct Cand {
    pub lb: f64,
    pub run: u32,
    pub level: u32,
    pub idx: u32,
}

impl PartialEq for Cand {
    fn eq(&self, o: &Self) -> bool {
        self.cmp(o) == Ordering::Equal
    }
}
impl Eq for Cand {}
impl PartialOrd for Cand {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Cand {
    fn cmp(&self, o: &Self) -> Ordering {
        self.lb.total_cmp(&o.lb)
    }
}

#[derive(Clone, Copy, PartialEq)]
pub(crate) struct Hit(pub f64, pub u32);
impl Eq for Hit {}
impl PartialOrd for Hit {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Hit {
    fn cmp(&self, o: &Self) -> Ordering {
        self.0.total_cmp(&o.0).then(self.1.cmp(&o.1))
    }
}

pub(crate) struct TopK {
    pub need: usize,
    pub heap: BinaryHeap<Hit>,
}

impl TopK {
    pub(crate) fn offer(&mut self, d: f64, id: u32) {
        if self.heap.len() < self.need {
            self.heap.push(Hit(d, id));
        } else if self.heap.peek().is_some_and(|h| d < h.0) {
            self.heap.pop();
            self.heap.push(Hit(d, id));
        }
    }

    pub(crate) fn sorted(self) -> Vec<(f64, u32)> {
        let mut out: Vec<(f64, u32)> = self.heap.into_iter().map(|h| (h.0, h.1)).collect();
        out.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        out
    }

    pub(crate) fn bound(&self) -> f64 {
        match self.heap.len() < self.need {
            true => f64::INFINITY,
            false => self.heap.peek().map_or(f64::INFINITY, |h| h.0),
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub(crate) struct QueryIo {
    pub blocks: u64,
    pub nodes: u64,
    pub reads: u64,
}

pub(crate) fn profiled((exact, io): (Vec<(f64, u32)>, QueryIo), prof: &QueryProfile) -> Scored {
    prof.count_visits(io.nodes, io.blocks, io.reads);
    Scored { exact, coded: Vec::new(), rerank: 0 }
}

impl LsmIndex {
    fn span_of(&self, r: usize, b: usize) -> (usize, usize) {
        let lo = b - b % self.span;
        (lo, (lo + self.span).min(self.shape.blocks(self.runs[r].meta.rows)))
    }

    pub(crate) fn scored(&self, query: &[f64], rows: Rows<'_>, weights: &[f64], need: usize) -> Result<(Vec<(f64, u32)>, QueryIo), ENNError> {
        let mut top = TopK { need: need.max(1), heap: BinaryHeap::new() };
        for i in self.meta.flushed as usize..rows.nrows() {
            let x = rows.fetch(i, false)?;
            top.offer(weighted_dist2(query, &x, weights), i as u32);
        }
        let io = self.search_runs(query, weights, &mut top)?;
        Ok((top.sorted(), io))
    }

    fn box_lb(&self, zq: &[f64], cr: f64, query: &[f64], weights: &[f64], bx: &[f64]) -> f64 {
        match self.keying.raw {
            true => (1.0 - 1e-9) * wbox_dist2(query, bx, weights),
            false => cr * box_dist2(zq, bx),
        }
    }

    fn search_runs(&self, query: &[f64], weights: &[f64], top: &mut TopK) -> Result<QueryIo, ENNError> {
        let (m, d) = (self.shape.m, self.shape.d);
        let mut zq = vec![0.0; self.runs.len() * m];
        let mut cr = vec![0.0; self.runs.len()];
        let mut heap = BinaryHeap::new();
        let mut bx = vec![0.0; 2 * self.shape.bm];
        for (r, run) in self.runs.iter().enumerate() {
            let z = &mut zq[r * m..(r + 1) * m];
            self.keying.proj.project(query, &run.meta.frame.scale, z);
            cr[r] = metric_factor(&run.meta.frame.scale, weights) * (1.0 - 1e-9);
            let level = run.levels.len() - 1;
            for idx in 0..run.levels[level].1 {
                run.bx(&self.shape, level, idx, &mut bx);
                let lb = self.box_lb(z, cr[r], query, weights, &bx);
                heap.push(Reverse(Cand { lb, run: r as u32, level: level as u32, idx: idx as u32 }));
            }
        }
        let mut io = QueryIo::default();
        let mut scanned = 0usize;
        let mut block = Vec::new();
        let mut seen = std::collections::HashSet::new();
        let mut x = vec![0.0; d];
        while let Some(Reverse(c)) = heap.pop() {
            if c.lb >= top.bound() {
                break;
            }
            let (r, run) = (c.run as usize, &self.runs[c.run as usize]);
            if c.level == 0 {
                if scanned >= self.max_rows && top.bound().is_finite() {
                    break;
                }
                if !seen.insert((c.run, c.idx)) {
                    continue;
                }
                let (lo, hi) = self.span_of(r, c.idx as usize);
                (lo..hi).for_each(|b| {
                    seen.insert((c.run, b as u32));
                });
                let n = run.read_blocks(&self.shape, lo, hi, &mut block)?;
                for i in 0..n {
                    let at = (i / self.shape.per) * self.shape.block;
                    let id = block_row(&self.shape, &block[at..], i % self.shape.per, &mut x);
                    top.offer(weighted_dist2(query, &x, weights), id);
                }
                scanned += n;
                io.blocks += (hi - lo) as u64;
                io.reads += 1;
                continue;
            }
            io.nodes += 1;
            let child = c.level as usize - 1;
            let lo = c.idx as usize * FANOUT;
            let hi = (lo + FANOUT).min(run.levels[child].1);
            for idx in lo..hi {
                run.bx(&self.shape, child, idx, &mut bx);
                let lb = self.box_lb(&zq[r * m..(r + 1) * m], cr[r], query, weights, &bx);
                if lb < top.bound() {
                    heap.push(Reverse(Cand { lb, run: c.run, level: child as u32, idx: idx as u32 }));
                }
            }
        }
        Ok(io)
    }
}

impl LsmIndex {
    pub(crate) fn scored_at(&self, query: &[f64], prof: &QueryProfile, rows: Rows<'_>, weights: &[f64], need: usize) -> Result<Scored, ENNError> {
        Ok(profiled(self.scored(query, rows, weights, need)?, prof))
    }

    pub(crate) fn candidate_ids(&self, query: &[f64], rows: Rows<'_>) -> Result<Vec<u32>, ENNError> {
        let w: Vec<f64> = self.meta.frame.scale.iter().map(|s| 1.0 / (s * s)).collect();
        let (hits, _) = self.scored(query, rows, &w, 32)?;
        let mut ids: Vec<u32> = hits.into_iter().map(|h| h.1).collect();
        ids.sort_unstable();
        Ok(ids)
    }

    pub(crate) fn occupancy(&self) -> O1Occupancy {
        O1Occupancy {
            num_rows: self.n,
            reachable_rows: self.n,
            disk_nodes: self.blocks(),
            node_bytes: self.shape.block,
            splits: self.meta.stats.merges,
            rebuilt_rows: self.meta.stats.merged_rows,
            max_rebuild_rows: self.meta.stats.max_merge_rows,
            ..O1Occupancy::default()
        }
    }
}
