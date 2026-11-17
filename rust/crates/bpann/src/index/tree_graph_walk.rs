use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::index::tree::Tree;
use crate::index::tree_codes::coded_sq_l2;
use crate::index::tree_search::FrontierItem;
use crate::index::tree_store::TreeStore;
use crate::index::tree_topk::TopK;
use crate::small_n_search::OrderedF32;

static WALK_GRAPH: AtomicUsize = AtomicUsize::new(usize::MAX);

pub const GRAPH_OFF: usize = 0;
pub const GRAPH_FIRST: usize = 1;
pub const GRAPH_MERGED: usize = 2;

thread_local! {
    static LOCAL_MODE: std::cell::Cell<usize> = const { std::cell::Cell::new(usize::MAX) };
}

pub fn walk_graph() -> usize {
    let local = LOCAL_MODE.with(std::cell::Cell::get);
    if local != usize::MAX {
        return local;
    }
    let v = WALK_GRAPH.load(Ordering::Relaxed);
    if v != usize::MAX {
        return v;
    }
    let e = std::env::var("ENN_WALK_GRAPH")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(GRAPH_OFF);
    WALK_GRAPH.store(e, Ordering::Relaxed);
    e
}

pub fn set_local_walk_graph(mode: Option<usize>) {
    LOCAL_MODE.with(|c| c.set(mode.unwrap_or(usize::MAX)));
}

#[derive(Default)]
struct IdHasher(u64);

impl Hasher for IdHasher {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(0x100_0000_01b3);
        }
    }
    fn write_u32(&mut self, x: u32) {
        self.0 = u64::from(x).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

type IdSet = HashSet<u32, BuildHasherDefault<IdHasher>>;
type GraphItem = (OrderedF32, OrderedF32, u32);
type KeyFn = fn(f32, f32) -> f32;

pub(super) struct GraphWalk {
    mode: usize,
    seen: IdSet,
    pushed: IdSet,
    heap: BinaryHeap<Reverse<GraphItem>>,
}

impl GraphWalk {
    pub(super) fn new() -> Self {
        Self {
            mode: walk_graph(),
            seen: IdSet::default(),
            pushed: IdSet::default(),
            heap: BinaryHeap::new(),
        }
    }

    pub(super) fn on(&self) -> bool {
        self.mode != GRAPH_OFF
    }

    pub(super) fn first_visit(&mut self, leaf: u32) -> bool {
        !self.on() || self.seen.insert(leaf)
    }

    pub(super) fn pop(&mut self) -> Option<(f32, u32)> {
        if self.mode != GRAPH_FIRST {
            return None;
        }
        self.heap.pop().map(|Reverse((_, b, l))| (b.0, l))
    }

    pub(super) fn bound(
        &self,
        s: &TreeStore,
        (n, raw, b): (u32, u16, OrderedF32),
        root: u32,
        lower: KeyFn,
    ) -> Option<OrderedF32> {
        if !self.on() {
            return Some(b);
        }
        if self.seen.contains(&n) {
            return None;
        }
        offer_bound(s, (n, raw, b.0), root, lower).map(OrderedF32)
    }

    pub(super) fn expand(
        &mut self,
        tree: &Tree,
        leaf: u32,
        query: (&[f32], f32),
        frontier: &mut BinaryHeap<Reverse<FrontierItem>>,
        (top, key, lower): (&TopK, KeyFn, KeyFn),
    ) {
        if !self.on() {
            return;
        }
        let s = &tree.store;
        self.pushed.insert(leaf);
        for (n, r, code) in s.lgraph_entries(leaf) {
            if self.seen.contains(&n) || !self.pushed.insert(n) {
                continue;
            }
            let (d2, err) = coded_sq_l2(query.0, query.1, code);
            let low = if code_err_on() {
                (d2.sqrt() - err).max(0.0).powi(2)
            } else {
                d2
            };
            if prune_on() && top.cannot_improve(lower(low, r)) {
                continue;
            }
            if self.mode == GRAPH_FIRST {
                self.heap
                    .push(Reverse((OrderedF32(d2), OrderedF32(low), n)));
            } else {
                frontier.push(Reverse((0, OrderedF32(key(d2, r)), OrderedF32(low), n, 0)));
            }
        }
    }
}

fn offer_bound(
    s: &TreeStore,
    (n, raw, low): (u32, u16, f32),
    root: u32,
    lower: KeyFn,
) -> Option<f32> {
    if raw != 0 || n == root {
        return Some(low);
    }
    if (n as usize) >= s.num_pages() || s.is_free(n) || !s.is_leaf(n) || s.len(n) == 0 {
        return None;
    }
    Some(lower(low, s.radius(n)))
}

static PRUNE: AtomicUsize = AtomicUsize::new(usize::MAX);

fn prune_on() -> bool {
    let v = PRUNE.load(Ordering::Relaxed);
    if v != usize::MAX {
        return v == 1;
    }
    let e = std::env::var("ENN_GRAPH_PRUNE")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(1);
    PRUNE.store(e, Ordering::Relaxed);
    e == 1
}

static CODE_ERR: AtomicUsize = AtomicUsize::new(usize::MAX);

fn code_err_on() -> bool {
    let v = CODE_ERR.load(Ordering::Relaxed);
    if v != usize::MAX {
        return v == 1;
    }
    let e = std::env::var("ENN_GRAPH_ERR")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(0);
    CODE_ERR.store(e, Ordering::Relaxed);
    e == 1
}
