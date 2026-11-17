use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::OnceLock;

use crate::distance::l2_sq_f32;
use crate::index::tree::{Tree, TREE_FANOUT};
use crate::index::tree_counts::dist;

static REPARENT: AtomicU8 = AtomicU8::new(u8::MAX);
static RETAG: AtomicU8 = AtomicU8::new(u8::MAX);
static RETAGGED: AtomicU64 = AtomicU64::new(0);
static TRIED: AtomicU64 = AtomicU64::new(0);
static MOVES: AtomicU64 = AtomicU64::new(0);
static CARRIED: AtomicU64 = AtomicU64::new(0);
static EVERY: AtomicU64 = AtomicU64::new(u64::MAX);
const SCAN: u32 = 64;

thread_local! {
    static CURSOR: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

type Buffered = Vec<(u32, Vec<f32>, f32)>;

fn reparent_every() -> u32 {
    let v = EVERY.load(Ordering::Relaxed);
    if v != u64::MAX {
        return v as u32;
    }
    let e = std::env::var("ENN_REPARENT_EVERY")
        .ok()
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(10u32);
    EVERY.store(u64::from(e), Ordering::Relaxed);
    e
}

pub fn set_reparent_every(every: u32) {
    EVERY.store(u64::from(every), Ordering::Relaxed);
}

pub fn reparent() -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    match REPARENT.load(Ordering::Relaxed) {
        u8::MAX => *ENV.get_or_init(|| std::env::var("ENN_REPARENT").as_deref() == Ok("1")),
        v => v == 1,
    }
}

pub fn lloyd_retag() -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    match RETAG.load(Ordering::Relaxed) {
        u8::MAX => *ENV.get_or_init(|| std::env::var("ENN_LLOYD_RETAG").as_deref() == Ok("1")),
        v => v == 1,
    }
}

pub fn set_lloyd_retag(on: bool) {
    RETAG.store(u8::from(on), Ordering::Relaxed);
}

pub fn set_reparent(on: bool) {
    REPARENT.store(u8::from(on), Ordering::Relaxed);
}

pub fn take_reparent_stats() -> (u64, u64, u64, u64) {
    (
        TRIED.swap(0, Ordering::Relaxed),
        MOVES.swap(0, Ordering::Relaxed),
        CARRIED.swap(0, Ordering::Relaxed),
        RETAGGED.swap(0, Ordering::Relaxed),
    )
}

impl Tree {
    pub(super) fn retag_parents(&mut self, leaves: &[u32]) {
        if !lloyd_retag() || !self.store.pbuf_tagged() {
            return;
        }
        let mut ps: Vec<u32> = leaves.iter().filter_map(|&l| self.store.parent(l)).collect();
        ps.sort_unstable();
        ps.dedup();
        for p in ps {
            if !self.store.is_leaf(p) && self.store.pbuf_len(p) > 0 {
                RETAGGED.fetch_add(1, Ordering::Relaxed);
                self.store.refresh_routes(p);
            }
        }
    }

    pub(super) fn reparent_step(&mut self, row: u32) {
        let k = reparent_every();
        if !reparent() || k == 0 || !row.is_multiple_of(k) {
            return;
        }
        let n = self.store.num_pages() as u32;
        if n == 0 {
            return;
        }
        let start = CURSOR.with(|c| c.get()) % n;
        for step in 0..SCAN.min(n) {
            let id = (start + step) % n;
            if self.live_leaf(id) && self.store.parent(id).is_some() {
                CURSOR.with(|c| c.set(id + 1));
                self.reparent_leaf(id);
                return;
            }
        }
        CURSOR.with(|c| c.set(start + SCAN));
    }

    fn leaf_parent(&self, p: u32) -> bool {
        let s = &self.store;
        !s.is_free(p) && !s.is_leaf(p) && s.ids(p).first().is_some_and(|&k| s.is_leaf(k))
    }

    fn reparent_target(&self, leaf: u32, c: &[f32], src: u32) -> Option<u32> {
        let s = &self.store;
        let mut best = (l2_sq_f32(c, &self.centroid_of(src)?), src);
        for n in s.lgraph_ids(leaf) {
            if !self.live_leaf(n) {
                continue;
            }
            let Some(p) = s.parent(n) else {
                continue;
            };
            if p == best.1 || p == src || !self.leaf_parent(p) || s.len(p) >= TREE_FANOUT {
                continue;
            }
            if !s.ids(p).contains(&n) {
                continue;
            }
            let Some(cp) = self.centroid_of(p) else {
                continue;
            };
            let d2 = l2_sq_f32(c, &cp);
            if d2 < best.0 {
                best = (d2, p);
            }
        }
        (best.1 != src).then_some(best.1)
    }

    fn reparent_leaf(&mut self, leaf: u32) {
        let Some((src, slot)) = self.slot_of(leaf) else {
            return;
        };
        let s = &self.store;
        if s.len(src) < 3 || s.parent(src).is_none() {
            return;
        }
        let Some(c) = self.centroid_of(leaf) else {
            return;
        };
        TRIED.fetch_add(1, Ordering::Relaxed);
        let Some(dst) = self.reparent_target(leaf, &c, src) else {
            return;
        };
        let s = &self.store;
        let carry = s
            .pbuf_tagged_records(src)
            .filter(|r| r.2 as usize == slot)
            .count();
        if s.pbuf_len(dst) > 0 && s.pbuf_len(dst) + carry >= s.pbuf_cap(dst) {
            return;
        }
        MOVES.fetch_add(1, Ordering::Relaxed);
        CARRIED.fetch_add(carry as u64, Ordering::Relaxed);
        self.move_leaf(leaf, (src, slot), dst, &c);
    }

    fn move_leaf(&mut self, leaf: u32, (src, slot): (u32, usize), dst: u32, c: &[f32]) {
        let d = self.store.num_dim();
        let (stay, carry) = self.split_buffer(src, slot);
        let n = self.store.count(leaf) + carry.len();
        let s = &mut self.store;
        let mut ids = s.ids(src).to_vec();
        let mut block = s.block(src).to_vec();
        ids.remove(slot);
        block.drain(slot * d..(slot + 1) * d);
        s.set_entries(src, &ids, &block);
        s.push_entry(dst, leaf, c);
        s.set_parent(leaf, Some(dst));
        if s.pbuf_tagged() && s.pbuf_len(dst) > 0 {
            s.refresh_routes(dst);
        }
        for (row, v, slack) in stay {
            s.pbuf_push(src, row, &v, slack);
        }
        for (row, v, slack) in carry {
            s.pbuf_push(dst, row, &v, slack);
        }
        s.set_count(src, s.count(src).saturating_sub(n));
        self.cover_up_many(src, &[], 0.0, -(n as isize));
        self.cover_up_many(leaf, &[], 0.0, n as isize);
        self.refit(src);
        self.refit(dst);
        self.cover_ancestors(dst);
    }

    fn split_buffer(&mut self, src: u32, slot: usize) -> (Buffered, Buffered) {
        let s = &mut self.store;
        if s.pbuf_len(src) == 0 {
            return (Vec::new(), Vec::new());
        }
        let tags: Vec<u8> = s.pbuf_tagged_records(src).map(|r| r.2).collect();
        let (mut stay, mut carry) = (Vec::new(), Vec::new());
        for (r, t) in s.pbuf_take(src).into_iter().zip(tags) {
            if t as usize == slot {
                carry.push(r);
            } else {
                stay.push(r);
            }
        }
        (stay, carry)
    }

    fn refit(&mut self, p: u32) {
        let Some((g, gs)) = self.slot_of(p) else {
            return;
        };
        let s = &self.store;
        let d = s.num_dim();
        let kids = s.ids(p).to_vec();
        let block = s.block(p).to_vec();
        let mut mean = vec![0.0f64; d];
        let mut total = 0.0f64;
        for (k, v) in kids.iter().zip(block.chunks_exact(d)) {
            let w = s.count(*k).max(1) as f64;
            total += w;
            mean.iter_mut().zip(v).for_each(|(m, &x)| *m += w * f64::from(x));
        }
        if total == 0.0 {
            return;
        }
        let mean: Vec<f32> = mean.iter().map(|m| (m / total) as f32).collect();
        let r = kids
            .iter()
            .zip(block.chunks_exact(d))
            .map(|(&k, v)| dist(v, &mean) + s.radius(k))
            .fold(0.0, f32::max)
            .max(s.pbuf_max_dist(p, &mean));
        let shift = dist(&s.block(g)[gs * d..(gs + 1) * d], &mean);
        let old = s.route_radius(g, gs);
        self.store.entry_mut(g, gs).copy_from_slice(&mean);
        self.store.set_radius(p, r);
        if self.store.pbuf_tagged() && self.store.pbuf_len(g) > 0 {
            self.store.widen_route(g, gs, old + shift);
        }
    }

    fn cover_ancestors(&mut self, from: u32) {
        let d = self.store.num_dim();
        let mut node = from;
        while let Some((p, slot)) = self.slot_of(node) {
            let Some((g, gs)) = self.slot_of(p) else {
                return;
            };
            let s = &self.store;
            let cn = &s.block(p)[slot * d..(slot + 1) * d];
            let cp = &s.block(g)[gs * d..(gs + 1) * d];
            let need = dist(cn, cp) + s.radius(node);
            if need > s.radius(p) {
                self.store.set_radius(p, need);
            }
            node = p;
        }
    }
}

#[cfg(test)]
#[path = "tree_reparent_tests.rs"]
mod tests;
