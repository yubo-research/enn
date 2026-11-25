use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::forest::ForestIndex;
use super::forest_bound::Scorer;
use super::forest_node::Node;
use super::lsh::io_error;
use super::profile::{QueryProfile, Stage};
use super::forest_rules::SelectRule;
use super::forest_tree::{RowSource, SplitRef};
use super::row_order::Rows;
use crate::error::ENNError;

pub(crate) fn select_by_votes(raw: &[u32], keep: usize) -> Vec<u32> {
    let mut order: Vec<usize> = (0..raw.len()).collect();
    order.sort_by_key(|&i| raw[i]);
    let mut tallies: Vec<(u32, usize, u32)> = Vec::new();
    for i in order {
        match tallies.last_mut() {
            Some(t) if t.0 == raw[i] => t.2 += 1,
            _ => tallies.push((raw[i], i, 1)),
        }
    }
    tallies.sort_by(|a, b| b.2.cmp(&a.2).then(a.1.cmp(&b.1)));
    let mut ids: Vec<u32> = tallies.into_iter().take(keep).map(|t| t.0).collect();
    ids.sort_unstable();
    ids
}

struct Probe {
    tier: f64,
    cost: f64,
    bound: f64,
    tree: usize,
    node: u32,
}

impl PartialEq for Probe {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Probe {}

impl PartialOrd for Probe {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Probe {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .tier
            .total_cmp(&self.tier)
            .then_with(|| other.cost.total_cmp(&self.cost))
            .then_with(|| other.tree.cmp(&self.tree))
            .then_with(|| other.node.cmp(&self.node))
    }
}

enum Expansion {
    Split { margin: f64, bound: f64, near: u32, far: u32 },
    Leaf { ids: Vec<u32>, pages: usize },
}

pub(crate) struct Scored {
    pub exact: Vec<(f64, u32)>,
    pub coded: Vec<(f64, u32)>,
    pub rerank: usize,
}

type Memo = HashMap<(usize, u32), Result<Expansion, ENNError>>;

type Done = ((usize, u32), Result<Expansion, ENNError>);

struct IoWorkers {
    jobs: Sender<(usize, u32)>,
    done: Receiver<Done>,
}

const LOOKAHEAD_POPS: usize = 4;

impl ForestIndex {
    fn expand(&self, query: &[f64], tree: usize, node: u32, prof: &QueryProfile, src: RowSource<'_>, weights: Option<&[f64]>) -> Result<Expansion, ENNError> {
        let t0 = prof.start();
        let read = self.trees[tree].nodes.read(node)?;
        self.finish(query, tree, read, t0, prof, src, weights)
    }

    #[allow(clippy::too_many_arguments)]
    fn finish(&self, query: &[f64], tree: usize, read: Node, t0: Option<Instant>, prof: &QueryProfile, src: RowSource<'_>, weights: Option<&[f64]>) -> Result<Expansion, ENNError> {
        let t = &self.trees[tree];
        match read {
            Node::Split { cut, thr, left, right } => {
                prof.stop(Stage::Route, t0);
                let t1 = prof.start();
                let (margin, bound, near, far) = t.branch(query, &self.invs[tree], &SplitRef { cut, thr, left, right }, src, weights)?;
                prof.stop(Stage::Hash, t1);
                Ok(Expansion::Split { margin, bound, near, far })
            }
            Node::Leaf { mut ids, next } => {
                let pages = 1 + t.leaf_ids(next, &mut ids)?;
                prof.stop(Stage::Leaf, t0);
                Ok(Expansion::Leaf { ids, pages })
            }
            Node::Free { .. } => Err(io_error("forest tree reaches a free node")),
        }
    }

    fn wave(&self, heap: &mut BinaryHeap<Probe>, memo: &Memo) -> Vec<(usize, u32)> {
        let width = self.params.io_width;
        let (mut top, mut wave) = (Vec::new(), Vec::new());
        while wave.len() < width && top.len() < LOOKAHEAD_POPS * width {
            let Some(p) = heap.pop() else { break };
            let mut key = (p.tree, p.node);
            while let Some(Ok(Expansion::Split { near, .. })) = memo.get(&key) {
                key = (p.tree, *near);
            }
            if !memo.contains_key(&key) && !wave.contains(&key) {
                wave.push(key);
            }
            top.push(p);
        }
        heap.extend(top);
        wave
    }

    fn expand_wave(&self, heap: &mut BinaryHeap<Probe>, memo: &mut Memo, io: &IoWorkers) -> Result<(), ENNError> {
        let wave = self.wave(heap, memo);
        for &key in &wave {
            io.jobs.send(key).map_err(io_error)?;
        }
        for _ in 0..wave.len() {
            let (key, done) = io.done.recv().map_err(io_error)?;
            memo.insert(key, done);
        }
        Ok(())
    }

    fn io_worker(&self, query: &[f64], prof: &QueryProfile, src: RowSource<'_>, weights: Option<&[f64]>, jobs: &Mutex<Receiver<(usize, u32)>>, done: &Sender<Done>) {
        loop {
            let Some((t, n)) = jobs.lock().ok().and_then(|rx| rx.recv().ok()) else { return };
            if done.send(((t, n), self.expand(query, t, n, prof, src, weights))).is_err() {
                return;
            }
        }
    }

    pub(crate) fn candidate_ids_at(&self, query: &[f64], prof: &QueryProfile, rows: Rows<'_>, overlap: bool) -> Result<Vec<u32>, ENNError> {
        self.probe_at(query, prof, rows, overlap, None)
    }

    pub(crate) fn scored_at(&self, query: &[f64], prof: &QueryProfile, rows: Rows<'_>, overlap: bool, weights: &[f64], need: usize) -> Result<Option<Scored>, ENNError> {
        let rules = &self.params.rules;
        if rules.bound <= 0.0 && rules.patience == 0 {
            return Ok(None);
        }
        let mut scorer = Scorer::new(weights, need, &self.invs, rules);
        self.probe_at(query, prof, rows, overlap, Some(&mut scorer))?;
        Ok(Some(Scored { exact: scorer.scored, coded: scorer.coded, rerank: rules.rerank as usize }))
    }

    fn probe_at(&self, query: &[f64], prof: &QueryProfile, rows: Rows<'_>, overlap: bool, mut score: Option<&mut Scorer<'_>>) -> Result<Vec<u32>, ENNError> {
        let src = RowSource { store: Rows { x: rows.x, order: self.order.as_ref() }, cache: self.cache.as_deref(), pread: false };
        if !overlap || self.params.io_width <= 1 {
            return self.search(query, prof, src, &mut || None, score);
        }
        let (jobs, job_rx) = mpsc::channel::<(usize, u32)>();
        let (done_tx, done) = mpsc::channel();
        let job_rx = Mutex::new(job_rx);
        let weights = score.as_deref().map(|s| s.weights);
        std::thread::scope(|scope| {
            let mut parts = Some((jobs, done, done_tx));
            let mut start = || {
                let (jobs, done, done_tx) = parts.take()?;
                for _ in 0..self.params.io_width {
                    let (job_rx, done_tx) = (&job_rx, done_tx.clone());
                    let pread = self.params.pread_pivots;
                    scope.spawn(move || self.io_worker(query, prof, RowSource { pread, ..src }, weights, job_rx, &done_tx));
                }
                Some(IoWorkers { jobs, done })
            };
            self.search(query, prof, src, &mut start, score.take())
        })
    }

    fn search(&self, query: &[f64], prof: &QueryProfile, src: RowSource<'_>, start_io: &mut dyn FnMut() -> Option<IoWorkers>, mut score: Option<&mut Scorer<'_>>) -> Result<Vec<u32>, ENNError> {
        let slow = Duration::from_nanos(self.params.io_slow_ns);
        let mut io: Option<IoWorkers> = None;
        let mut heap: BinaryHeap<Probe> = (0..self.trees.len())
            .map(|tree| Probe { tier: self.tier(tree), cost: 0.0, bound: 0.0, tree, node: self.trees[tree].nodes.root() })
            .collect();
        let mut memo = Memo::new();
        let weights = score.as_deref().map(|s| s.weights);
        let (mut ids, mut pages, mut splits, mut leaves) = (Vec::new(), 0, 0, 0);
        while let Some(top) = heap.peek() {
            let full = self.params.rules.select == SelectRule::FirstSeen && ids.len() >= self.params.max_candidates;
            if pages >= self.params.max_leaves || full {
                break;
            }
            let key = (top.tree, top.node);
            if let Some(sc) = score.as_deref_mut() {
                if sc.settled() {
                    break;
                }
                if sc.prunes(top.bound) {
                    heap.pop();
                    continue;
                }
            }
            if let Some(io) = io.as_ref().filter(|_| !memo.contains_key(&key)) {
                self.expand_wave(&mut heap, &mut memo, io)?;
            }
            let p = heap.pop().expect("peeked probe");
            let step = match memo.remove(&key) {
                Some(done) => done?,
                None => {
                    let t0 = Instant::now();
                    let step = self.expand(query, p.tree, p.node, prof, src, weights)?;
                    if io.is_none() && t0.elapsed() >= slow {
                        io = start_io();
                    }
                    step
                }
            };
            match step {
                Expansion::Split { margin, bound, near, far } => {
                    splits += 1;
                    let rule = self.params.rules.probe;
                    heap.push(Probe { node: near, ..p });
                    heap.push(Probe { cost: rule.cost(p.cost, margin), bound: rule.cost(p.bound, bound), node: far, ..p });
                }
                Expansion::Leaf { ids: page, pages: n } => {
                    if let Some(sc) = score.as_deref_mut() {
                        sc.add(query, &page, src)?;
                    }
                    ids.extend_from_slice(&page);
                    pages += n;
                    leaves += 1;
                }
            }
        }
        prof.count_visits(splits, leaves, pages as u64);
        if self.params.rules.select == SelectRule::Votes {
            return Ok(select_by_votes(&ids, self.params.max_candidates));
        }
        ids.truncate(self.params.max_candidates);
        ids.sort_unstable();
        ids.dedup();
        Ok(ids)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn votes_keep_the_most_repeated_ids_then_the_earliest_seen() {
        assert_eq!(select_by_votes(&[5, 9, 2, 9, 7, 2, 9], 2), vec![2, 9]);
        assert_eq!(select_by_votes(&[5, 9, 2, 9, 7, 2, 9], 3), vec![2, 5, 9]);
        assert_eq!(select_by_votes(&[4, 3], 9), vec![3, 4]);
        assert!(select_by_votes(&[], 3).is_empty());
    }
}
