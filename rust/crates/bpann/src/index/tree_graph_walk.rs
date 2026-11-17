use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::distance::l2_sq_f32;
use crate::index::tree::Tree;
use crate::index::tree_lloyd::with_graph_neighbors;
use crate::index::tree_search::FrontierItem;
use crate::index::tree_store::TreeStore;
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

    pub(super) fn expand(
        &mut self,
        tree: &Tree,
        leaf: u32,
        query: &[f32],
        frontier: &mut BinaryHeap<Reverse<FrontierItem>>,
        keys: (KeyFn, KeyFn),
    ) {
        if !self.on() {
            return;
        }
        let s = &tree.store;
        self.pushed.insert(leaf);
        with_graph_neighbors(leaf, |ns| {
            for &n in ns {
                self.offer(s, n, query, frontier, keys);
            }
        });
    }

    fn offer(
        &mut self,
        s: &TreeStore,
        n: u32,
        query: &[f32],
        frontier: &mut BinaryHeap<Reverse<FrontierItem>>,
        keys: (KeyFn, KeyFn),
    ) {
        if self.seen.contains(&n) || !self.pushed.insert(n) || !live_leaf(s, n) {
            return;
        }
        let Some(d2) = centroid_d2(s, n, query) else {
            return;
        };
        let r = s.radius(n);
        let bound = OrderedF32((keys.1)(d2, r));
        if self.mode == GRAPH_FIRST {
            self.heap.push(Reverse((OrderedF32(d2), bound, n)));
        } else {
            frontier.push(Reverse((0, OrderedF32((keys.0)(d2, r)), bound, n, 0)));
        }
    }
}

fn live_leaf(s: &TreeStore, n: u32) -> bool {
    (n as usize) < s.num_pages() && !s.is_free(n) && s.is_leaf(n) && s.len(n) > 0
}

fn centroid_d2(s: &TreeStore, leaf: u32, query: &[f32]) -> Option<f32> {
    let p = s.parent(leaf)?;
    let slot = s.ids(p).iter().position(|&c| c == leaf)?;
    let d = s.num_dim();
    Some(l2_sq_f32(&s.block(p)[slot * d..(slot + 1) * d], query))
}
