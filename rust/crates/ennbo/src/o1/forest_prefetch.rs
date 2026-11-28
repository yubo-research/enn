
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;

use super::forest_lrows::NearRows;
use super::forest_node::{decode, Node, NodeFile, NIL};
use super::forest_tree::{row, RowSource, SplitRef, Tree};
use super::io_streams::as_prefetch;
use super::lsh::io_error;
use super::row_order::Rows;
use crate::error::ENNError;

pub(crate) fn prefetch_width() -> usize {
    static W: OnceLock<usize> = OnceLock::new();
    *W.get_or_init(|| {
        std::env::var("ENN_O1_PREFETCH").map_or(0, |v| {
            v.parse()
                .unwrap_or_else(|_| panic!("ENN_O1_PREFETCH must be a thread count, got {v:?}"))
        })
    })
}

static TOTALS: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];

pub fn totals() -> [u64; 6] {
    std::array::from_fn(|k| TOTALS[k].load(Ordering::Relaxed))
}

type Slice<U> = Result<Vec<(usize, U)>, ENNError>;

fn run_slice<T, U>(
    items: &[T],
    first: usize,
    width: usize,
    job: &(impl Fn(&T) -> Result<U, ENNError> + Sync),
) -> Slice<U> {
    let mut out = Vec::new();
    for k in (first..items.len()).step_by(width) {
        out.push((k, job(&items[k])?));
    }
    Ok(out)
}

fn spread<T: Sync, U: Send>(
    items: &[T],
    width: usize,
    job: impl Fn(&T) -> Result<U, ENNError> + Sync,
) -> Result<Vec<U>, ENNError> {
    assert!(width >= 1, "spread needs at least one thread");
    let job = &job;
    let parts: Vec<Slice<U>> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for t in 0..width.min(items.len()) {
            handles.push(scope.spawn(move || as_prefetch(|| run_slice(items, t, width, job))));
        }
        handles.into_iter().map(join).collect()
    });
    let mut out: Vec<(usize, U)> = Vec::with_capacity(items.len());
    for part in parts {
        out.extend(part?);
    }
    out.sort_unstable_by_key(|(k, _)| *k);
    Ok(out.into_iter().map(|(_, u)| u).collect())
}

fn join<R>(h: std::thread::ScopedJoinHandle<'_, R>) -> R {
    h.join().unwrap_or_else(|e| std::panic::resume_unwind(e))
}

type Group = (u32, Vec<u32>);
type Arrivals = Vec<(u32, usize)>;

pub(crate) fn gated() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var("ENN_O1_PREFETCH_GATE").map_or(true, |v| v != "0"))
}

const MAX_SKIP: u32 = 64;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct Gate {
    skip: u32,
    backoff: u32,
}

impl Gate {
    pub(crate) fn due(&mut self) -> bool {
        if self.skip == 0 {
            return true;
        }
        self.skip -= 1;
        TOTALS[5].fetch_add(1, Ordering::Relaxed);
        false
    }

    pub(crate) fn observe(&mut self, slow_reads: u64, rows: usize) {
        if !gated() || slow_reads * 100 >= rows as u64 {
            *self = Self::default();
            return;
        }
        self.backoff = (self.backoff * 2).clamp(1, MAX_SKIP);
        self.skip = self.backoff;
    }
}

fn peek(nodes: &NodeFile, id: u32) -> Result<Node, ENNError> {
    if id == NIL || id >= nodes.num_nodes() {
        return Err(io_error(format!("forest node {id} is out of range")));
    }
    match nodes.pack.as_ref().and_then(|p| super::forest_npack::copied(p, id)) {
        Some(raw) => decode(&raw, nodes.cap(), nodes.dim_payload().0),
        None => nodes.read_disk(id, false),
    }
}

impl Tree {
    #[cfg(test)]
    pub(crate) fn peek_leaf(&self, id: u32, rows: Rows<'_>, inv_scale: &[f64]) -> Result<u32, ENNError> {
        let x = row(rows, id)?;
        let src = RowSource {
            store: rows,
            cache: None,
            pread: false,
        };
        let mut at = self.nodes.root();
        loop {
            match peek(&self.nodes, at)? {
                Node::Split {
                    cut,
                    thr,
                    left,
                    right,
                } => {
                    let split = SplitRef {
                        cut,
                        thr,
                        left,
                        right,
                    };
                    at = self.branch(&x, inv_scale, &split, src, None)?.2;
                }
                Node::Leaf { .. } => return Ok(at),
                Node::Free { .. } => return Err(io_error("forest tree reaches a free node")),
            }
        }
    }

    fn peek_chain(&self, leaf: u32) -> Result<(Vec<u32>, Vec<u32>, usize), ENNError> {
        let (mut pages, mut ids, mut tail) = (Vec::new(), Vec::new(), 0);
        let mut at = leaf;
        while at != NIL {
            let Node::Leaf { ids: page_ids, next } = peek(&self.nodes, at)? else {
                return Err(io_error("forest leaf chain is corrupt"));
            };
            pages.push(at);
            tail = page_ids.len();
            ids.extend_from_slice(&page_ids);
            at = next;
        }
        Ok((pages, ids, tail))
    }

    fn warm_split(&self, pages: &[u32], ids: &[u32], rows: Rows<'_>) -> Result<(), ENNError> {
        let mut near = NearRows::new();
        if let Some(l) = &self.lrows {
            for &page in pages {
                l.get(page, Some(ids), &mut near)?;
            }
        }
        if rows.order.is_some() || super::forest_gather::split_gap().is_none() {
            return Ok(());
        }
        let mut miss: Vec<u32> = ids.iter().copied().filter(|i| !near.contains_key(i)).collect();
        miss.sort_unstable();
        let mut xs = Vec::with_capacity(miss.len() * rows.x.ncols);
        rows.x.pread_runs(&miss, 0, &mut xs)
    }

    pub(crate) fn prefetch_counted(
        &self,
        ids: &[u32],
        rows: Rows<'_>,
        inv_scale: &[f64],
        width: usize,
    ) -> Result<u64, ENNError> {
        let slow0 = super::io_streams::slow_reads(super::io_streams::Stream::Prefetch);
        let t0 = std::time::Instant::now();
        let (leaves, warmed) = self.prefetch(ids, rows, inv_scale, width)?;
        let add = [1, ids.len() as u64, leaves as u64, warmed as u64, t0.elapsed().as_nanos() as u64];
        for (t, v) in TOTALS.iter().zip(add) {
            t.fetch_add(v, Ordering::Relaxed);
        }
        Ok(super::io_streams::slow_reads(super::io_streams::Stream::Prefetch).saturating_sub(slow0))
    }

    fn split_group(
        &self,
        node: u32,
        group: &[u32],
        rows: Rows<'_>,
        inv_scale: &[f64],
    ) -> Result<Option<[Group; 2]>, ENNError> {
        let Node::Split {
            cut,
            thr,
            left,
            right,
        } = peek(&self.nodes, node)?
        else {
            return Ok(None);
        };
        let split = SplitRef {
            cut,
            thr,
            left,
            right,
        };
        let src = RowSource {
            store: rows,
            cache: None,
            pread: false,
        };
        let (mut lo, mut hi) = (Vec::new(), Vec::new());
        for &id in group {
            let near = self.branch(&row(rows, id)?, inv_scale, &split, src, None)?.2;
            if near == left {
                lo.push(id);
            } else {
                hi.push(id);
            }
        }
        Ok(Some([(left, lo), (right, hi)]))
    }

    fn route_down(&self, start: &Group, rows: Rows<'_>, inv_scale: &[f64]) -> Result<Vec<(u32, usize)>, ENNError> {
        let (mut stack, mut leaves) = (vec![start.clone()], Vec::new());
        while let Some((node, group)) = stack.pop() {
            match self.split_group(node, &group, rows, inv_scale)? {
                Some(kids) => stack.extend(kids.into_iter().filter(|(_, g)| !g.is_empty())),
                None => leaves.push((node, group.len())),
            }
        }
        Ok(leaves)
    }

    fn frontier(
        &self,
        ids: &[u32],
        want: usize,
        rows: Rows<'_>,
        inv_scale: &[f64],
    ) -> Result<(Arrivals, Vec<Group>), ENNError> {
        let (mut leaves, mut open) = (Vec::new(), vec![(self.nodes.root(), ids.to_vec())]);
        while !open.is_empty() && open.len() < want {
            let mut next = Vec::new();
            for (node, group) in open {
                match self.split_group(node, &group, rows, inv_scale)? {
                    Some(kids) => next.extend(kids.into_iter().filter(|(_, g)| !g.is_empty())),
                    None => leaves.push((node, group.len())),
                }
            }
            open = next;
        }
        Ok((leaves, open))
    }

    pub(crate) fn batch_leaves(
        &self,
        ids: &[u32],
        rows: Rows<'_>,
        inv_scale: &[f64],
        width: usize,
    ) -> Result<Vec<(u32, usize)>, ENNError> {
        let mut arrivals: HashMap<u32, usize> = HashMap::new();
        let (top, open) = self.frontier(ids, 4 * width, rows, inv_scale)?;
        let deep = spread(&open, width, |g| self.route_down(g, rows, inv_scale))?;
        for (leaf, n) in top.into_iter().chain(deep.into_iter().flatten()) {
            *arrivals.entry(leaf).or_default() += n;
        }
        let mut targets: Vec<(u32, usize)> = arrivals.into_iter().collect();
        targets.sort_unstable();
        Ok(targets)
    }

    pub(crate) fn prefetch(
        &self,
        ids: &[u32],
        rows: Rows<'_>,
        inv_scale: &[f64],
        width: usize,
    ) -> Result<(usize, usize), ENNError> {
        let targets = self.batch_leaves(ids, rows, inv_scale, width)?;
        let cap = self.nodes.cap();
        let warmed = spread(&targets, width, |&(leaf, arriving)| {
            let (pages, ids, tail) = self.peek_chain(leaf)?;
            let splits = tail + arriving > cap;
            if splits {
                self.warm_split(&pages, &ids, rows)?;
            }
            Ok(splits)
        })?;
        Ok((targets.len(), warmed.into_iter().filter(|&s| s).count()))
    }
}
