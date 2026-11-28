use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};

use super::disk_index::O1Occupancy;
use super::forest_query::Scored;
use super::graph::GraphIndex;
use super::graph_io::Items;
use super::graph_box::box_lb;
use super::lsm_query::{QueryIo, TopK};
use super::profile::QueryProfile;
use super::search::weighted_dist2;
use crate::error::ENNError;

#[derive(Clone, Copy, PartialEq)]
struct GCand {
    key: f64,
    lb: f64,
    level: u32,
    idx: u32,
}

impl Eq for GCand {}
impl PartialOrd for GCand {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for GCand {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.key.total_cmp(&o.key).then(self.idx.cmp(&o.idx))
    }
}

type Queue = BinaryHeap<Reverse<GCand>>;

struct Ctx<'a> {
    q: &'a [f64],
    w: &'a [f64],
}

impl GraphIndex {
    pub(crate) fn scored(&self, query: &[f64], weights: &[f64], need: usize) -> Result<(Vec<(f64, u32)>, QueryIo), ENNError> {
        let cx = Ctx { q: query, w: weights };
        let mut top = TopK { need: need.max(1), heap: BinaryHeap::new() };
        self.score_buffer(&cx, &mut top);
        let mut heap = Queue::new();
        heap.push(Reverse(GCand { key: 0.0, lb: 0.0, level: self.meta.height as u32 - 1, idx: self.meta.root }));
        let mut seen: HashSet<u32> = HashSet::new();
        let (mut io, mut scanned) = (QueryIo::default(), 0usize);
        loop {
            if scanned >= self.max_rows && top.bound().is_finite() {
                break;
            }
            let batch = self.next_leaves(&cx, &mut heap, &mut seen, &mut io, top.bound())?;
            if batch.is_empty() {
                break;
            }
            for (id, leaf) in batch.iter().zip(self.read_leaves(&batch)?) {
                (0..leaf.len()).for_each(|i| top.offer(weighted_dist2(query, leaf.row(i, self.meta.d), weights), leaf.ids[i]));
                scanned += leaf.len();
                for l in self.files.read_links(*id)? {
                    if seen.insert(l) {
                        heap.push(Reverse(self.leaf_cand(&cx, l)?));
                    }
                }
            }
            io.blocks += batch.len() as u64;
            io.reads += 1;
        }
        Ok((top.sorted(), io))
    }

    fn score_buffer(&self, cx: &Ctx<'_>, top: &mut TopK) {
        for b in self.buf.values() {
            (0..b.len()).for_each(|i| top.offer(weighted_dist2(cx.q, b.row(i, self.meta.d), cx.w), b.ids[i]));
        }
    }

    fn lb(&self, cx: &Ctx<'_>, level: usize, id: u32) -> Result<f64, ENNError> {
        match self.bound {
            true => Ok(box_lb(cx.q, &self.boxes.read(level, id)?, cx.w)),
            false => Ok(0.0),
        }
    }

    fn leaf_cand(&self, cx: &Ctx<'_>, l: u32) -> Result<GCand, ENNError> {
        let key = weighted_dist2(cx.q, &self.files.read_cent(l)?, cx.w);
        Ok(GCand { key, lb: self.lb(cx, 0, l)?, level: 0, idx: l })
    }

    fn next_leaves(&self, cx: &Ctx<'_>, heap: &mut Queue, seen: &mut HashSet<u32>, io: &mut QueryIo, bound: f64) -> Result<Vec<u32>, ENNError> {
        let d = self.meta.d;
        let mut batch = Vec::new();
        while batch.len() < self.width {
            let Some(Reverse(c)) = heap.pop() else { break };
            if c.lb >= bound {
                continue;
            }
            if c.level == 0 {
                batch.push(c.idx);
                continue;
            }
            let rec = self.files.read_up(c.level as usize, c.idx)?;
            io.nodes += 1;
            for i in 0..rec.len() {
                if c.level > 1 || seen.insert(rec.ids[i]) {
                    let key = weighted_dist2(cx.q, rec.row(i, d), cx.w);
                    let lb = self.lb(cx, c.level as usize - 1, rec.ids[i])?;
                    heap.push(Reverse(GCand { key, lb, level: c.level - 1, idx: rec.ids[i] }));
                }
            }
        }
        Ok(batch)
    }

    fn read_leaves(&self, ids: &[u32]) -> Result<Vec<Items>, ENNError> {
        if ids.len() == 1 {
            return Ok(vec![self.files.read_leaf(ids[0])?]);
        }
        std::thread::scope(|sc| {
            let hs: Vec<_> = ids.iter().map(|&id| sc.spawn(move || self.files.read_leaf(id))).collect();
            hs.into_iter().map(|h| h.join().unwrap_or_else(|_| Err(ENNError::InvalidParameter("leaf read panicked".into())))).collect()
        })
    }

    pub(crate) fn scored_at(&self, query: &[f64], prof: &QueryProfile, weights: &[f64], need: usize) -> Result<Scored, ENNError> {
        let (exact, io) = self.scored(query, weights, need)?;
        prof.count_visits(io.nodes, io.blocks, io.reads);
        Ok(Scored { exact, coded: Vec::new(), rerank: 0 })
    }

    pub(crate) fn candidate_ids(&self, query: &[f64]) -> Result<Vec<u32>, ENNError> {
        let (hits, _) = self.scored(query, &self.weights(), 32)?;
        let mut ids: Vec<u32> = hits.into_iter().map(|h| h.1).collect();
        ids.sort_unstable();
        Ok(ids)
    }

    pub(crate) fn occupancy(&self) -> O1Occupancy {
        O1Occupancy {
            num_rows: self.meta.n as usize,
            reachable_rows: self.meta.n as usize,
            disk_nodes: self.leaves(),
            node_bytes: self.files.geo.leaf_bytes(),
            splits: self.meta.stats.leaf_splits,
            mean_depth: self.meta.height as f64,
            ..O1Occupancy::default()
        }
    }
}
