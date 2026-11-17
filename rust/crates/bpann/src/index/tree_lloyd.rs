use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::distance::l2_sq_f32;
use crate::index::tree::Tree;
use crate::index::tree_reassign::{assign, Hood};
use crate::small_n_search::OrderedF32;

static LLOYD_EVERY: AtomicUsize = AtomicUsize::new(usize::MAX);
static LLOYD_HOOD: AtomicUsize = AtomicUsize::new(usize::MAX);
static LLOYD_PAGES: AtomicUsize = AtomicUsize::new(usize::MAX);
static LLOYD_GRAPH: AtomicUsize = AtomicUsize::new(usize::MAX);

fn knob(cell: &AtomicUsize, var: &str, fallback: usize) -> usize {
    let v = cell.load(Ordering::Relaxed);
    if v != usize::MAX {
        return v;
    }
    let e = std::env::var(var)
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(fallback);
    cell.store(e, Ordering::Relaxed);
    e
}

pub fn lloyd_every() -> usize {
    knob(&LLOYD_EVERY, "ENN_LLOYD_EVERY", 0)
}

fn lloyd_hood() -> usize {
    knob(&LLOYD_HOOD, "ENN_LLOYD_HOOD", 16)
}

fn lloyd_pages() -> usize {
    knob(&LLOYD_PAGES, "ENN_LLOYD_PAGES", 8)
}

pub fn set_lloyd(every: usize, hood: usize, pages: usize, graph: bool) {
    reset_lloyd_graph();
    LLOYD_GRAPH.store(usize::from(graph), Ordering::Relaxed);
    LLOYD_EVERY.store(every, Ordering::Relaxed);
    LLOYD_HOOD.store(hood, Ordering::Relaxed);
    LLOYD_PAGES.store(pages, Ordering::Relaxed);
}

thread_local! {
    static CURSOR: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

fn lloyd_graph() -> bool {
    knob(&LLOYD_GRAPH, "ENN_LLOYD_GRAPH", 0) == 1
}

pub fn reset_lloyd_graph() {
    CURSOR.with(|c| c.set(0));
}

type Candidate = (f32, u32, u32, usize);

impl Tree {
    fn next_sweep_leaf(&self) -> Option<u32> {
        let n = self.store.num_pages() as u32;
        if n == 0 {
            return None;
        }
        let start = CURSOR.with(|c| c.get()) % n;
        for step in 0..n {
            let id = (start + step) % n;
            if !self.store.is_free(id)
                && self.store.is_leaf(id)
                && self.store.len(id) > 0
                && self.store.parent(id).is_some()
            {
                CURSOR.with(|c| c.set(id + 1));
                return Some(id);
            }
        }
        None
    }

    fn leaf_candidates(&self, c: &[f32], m: usize, budget: usize) -> Vec<Candidate> {
        let s = &self.store;
        let d = s.num_dim();
        let mut heap = BinaryHeap::new();
        heap.push(Reverse((OrderedF32(0.0), self.header.root_page_id)));
        let mut found: Vec<Candidate> = Vec::new();
        let mut popped = 0usize;
        while let Some(Reverse((OrderedF32(bound), page))) = heap.pop() {
            if popped >= budget || (found.len() >= m && bound > kth(&mut found, m)) {
                break;
            }
            popped += 1;
            let block = s.block(page);
            for (i, &child) in s.ids(page).iter().enumerate() {
                let d2 = l2_sq_f32(&block[i * d..(i + 1) * d], c);
                if s.is_leaf(child) {
                    found.push((d2, child, page, i));
                } else {
                    let gap = (d2.sqrt() - s.radius(child)).max(0.0);
                    heap.push(Reverse((OrderedF32(gap * gap), child)));
                }
            }
        }
        found
    }

    fn global_hood(&self, leaf: u32, m: usize) -> Option<Hood> {
        let s = &self.store;
        let d = s.num_dim();
        let p = s.parent(leaf)?;
        let slot = s.ids(p).iter().position(|&x| x == leaf)?;
        let c = s.block(p)[slot * d..(slot + 1) * d].to_vec();
        let mut found = self.leaf_candidates(&c, m, lloyd_pages());
        found.retain(|x| x.1 != leaf);
        found.sort_by(|x, y| x.0.total_cmp(&y.0));
        found.truncate(m.saturating_sub(1));
        if found.is_empty() {
            return None;
        }
        let mut leaves = vec![leaf];
        let mut cents = vec![c];
        for (_, l, pg, i) in found {
            leaves.push(l);
            cents.push(s.block(pg)[i * d..(i + 1) * d].to_vec());
        }
        Some(Hood { leaves, cents })
    }

    fn live_leaf(&self, l: u32) -> bool {
        (l as usize) < self.store.num_pages()
            && !self.store.is_free(l)
            && self.store.is_leaf(l)
            && self.store.len(l) > 0
    }

    fn graph_candidates(&self, leaf: u32, m: usize) -> HashSet<u32> {
        let s = &self.store;
        let mut cand: HashSet<u32> = HashSet::new();
        let first: Vec<u32> = s.lgraph_ids(leaf);
        if first.is_empty() {
            if let Some(h) = self.hood(leaf, leaf, m) {
                cand.extend(h.leaves);
            }
        }
        for &n in &first {
            cand.insert(n);
            if self.live_leaf(n) {
                cand.extend(s.lgraph_entries(n).map(|e| e.0));
            }
        }
        if let Some(p) = s.parent(leaf) {
            cand.extend(s.ids(p).iter().copied().filter(|&c| s.is_leaf(c)));
        }
        cand.remove(&leaf);
        cand.retain(|&l| self.live_leaf(l));
        cand
    }

    fn graph_hood(&mut self, leaf: u32, m: usize) -> Option<Hood> {
        let c = self.centroid_of(leaf)?;
        let mut scored: Vec<(f32, u32, Vec<f32>)> = self
            .graph_candidates(leaf, m)
            .into_iter()
            .filter_map(|l| self.centroid_of(l).map(|cl| (l2_sq_f32(&cl, &c), l, cl)))
            .collect();
        scored.sort_by(|x, y| x.0.total_cmp(&y.0));
        scored.truncate(m.saturating_sub(1));
        if scored.is_empty() {
            return None;
        }
        let s = &mut self.store;
        let entries: Vec<(u32, f32, Vec<f32>)> = scored
            .iter()
            .map(|(_, l, cl)| (*l, s.radius(*l), cl.clone()))
            .collect();
        s.lgraph_set(leaf, &entries);
        let me = (leaf, s.radius(leaf), &c[..]);
        for &(_, n, _) in &scored {
            s.lgraph_link(n, me, 2 * m);
        }
        let mut leaves = vec![leaf];
        let mut cents = vec![c];
        for (_, l, cl) in scored {
            leaves.push(l);
            cents.push(cl);
        }
        Some(Hood { leaves, cents })
    }

    pub(super) fn lloyd_sweep_step(&mut self, row: u32) {
        let k = lloyd_every();
        if k == 0
            || !row.is_multiple_of(k as u32)
            || self.store.pbuf_on()
            || self.store.resid()
            || self.store.nib()
        {
            return;
        }
        let Some(leaf) = self.next_sweep_leaf() else {
            return;
        };
        let h = if lloyd_graph() {
            self.graph_hood(leaf, lloyd_hood())
        } else {
            self.global_hood(leaf, lloyd_hood())
        };
        let Some(h) = h else {
            return;
        };
        let g = self.gather(&h);
        let sizes: Vec<usize> = h.leaves.iter().map(|&l| self.store.len(l)).collect();
        let (dest, moved) = assign(&h, &g, sizes, self.store.num_dim(), true);
        if moved > 0 {
            self.rewrite(&h, &g, &dest);
        }
    }
}

fn kth(found: &mut [Candidate], m: usize) -> f32 {
    found.select_nth_unstable_by(m - 1, |x, y| x.0.total_cmp(&y.0));
    found[m - 1].0
}
