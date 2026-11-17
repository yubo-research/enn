use std::sync::atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering};
use std::sync::OnceLock;

use crate::error::BpannError;
use crate::index::tree::{mean_update, nearest, updates_centroid, Tree, TREE_CENTROID_STRIDE};
use crate::index::tree_counts::dist;
use crate::index::tree_pbuf::thread_majflt;
use crate::index::tree_pbuf_cold::{add_ns, timed};
use crate::index::tree_store::TreeStore;

static ALL: AtomicU8 = AtomicU8::new(u8::MAX);
static STAGGER: AtomicU8 = AtomicU8::new(u8::MAX);
static PUNCH: AtomicU8 = AtomicU8::new(u8::MAX);
static CHARGE: AtomicU8 = AtomicU8::new(u8::MAX);
static FAULTS: [AtomicU64; 5] = [
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
    AtomicU64::new(0),
];

fn fault_split() -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    *ENV.get_or_init(|| std::env::var("ENN_PBUF_FAULTS").as_deref() == Ok("1"))
}

fn counted<T>(i: usize, f: impl FnOnce() -> T) -> T {
    if !fault_split() {
        return f();
    }
    let f0 = thread_majflt();
    let out = f();
    FAULTS[i].fetch_add(thread_majflt() - f0, Ordering::Relaxed);
    out
}

pub fn take_pbuf_fault_split() -> (u64, u64, u64, u64, u64) {
    let t = |i: usize| FAULTS[i].swap(0, Ordering::Relaxed);
    (t(0), t(1), t(2), t(3), t(4))
}

fn flag(over: &AtomicU8, env: &OnceLock<bool>, name: &str) -> bool {
    match over.load(Ordering::Relaxed) {
        u8::MAX => *env.get_or_init(|| std::env::var(name).as_deref() == Ok("1")),
        v => v == 1,
    }
}

pub fn pbuf_leaf_rows() -> usize {
    static ENV: OnceLock<usize> = OnceLock::new();
    match LEAF_ROWS.load(Ordering::Relaxed) {
        usize::MAX => *ENV.get_or_init(|| {
            std::env::var("ENN_PBUF_LEAF_ROWS")
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(0)
        }),
        v => v,
    }
}

pub fn set_pbuf_leaf_rows(rows: usize) {
    LEAF_ROWS.store(rows, Ordering::Relaxed);
}

static LEAF_ROWS: AtomicUsize = AtomicUsize::new(usize::MAX);

pub fn pbuf_all() -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    flag(&ALL, &ENV, "ENN_PBUF_ALL")
}

pub fn pbuf_stagger() -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    flag(&STAGGER, &ENV, "ENN_PBUF_STAGGER")
}

pub fn pbuf_prefetch() -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    *ENV.get_or_init(|| std::env::var("ENN_PBUF_PREFETCH").as_deref() != Ok("0"))
}

pub const PBUF_CHARGE_ROWS: usize = crate::index::tree::TREE_LEAF_CAPACITY / 2;

pub fn pbuf_charge() -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    flag(&CHARGE, &ENV, "ENN_PBUF_CHARGE")
}

pub fn set_pbuf_charge(on: bool) {
    CHARGE.store(u8::from(on), Ordering::Relaxed);
}

pub(super) fn charged_leaves(buffered_rows: usize) -> usize {
    if pbuf_charge() {
        buffered_rows.div_ceil(PBUF_CHARGE_ROWS)
    } else {
        0
    }
}

pub fn pbuf_punch() -> bool {
    static ENV: OnceLock<Option<bool>> = OnceLock::new();
    match PUNCH.load(Ordering::Relaxed) {
        u8::MAX => ENV
            .get_or_init(|| std::env::var("ENN_PBUF_PUNCH").ok().map(|v| v != "0"))
            .unwrap_or_else(pbuf_all),
        v => v == 1,
    }
}

pub fn set_pbuf_punch(on: bool) {
    PUNCH.store(u8::from(on), Ordering::Relaxed);
}

pub fn set_pbuf_all(on: bool) {
    ALL.store(u8::from(on), Ordering::Relaxed);
}

pub fn set_pbuf_stagger(on: bool) {
    STAGGER.store(u8::from(on), Ordering::Relaxed);
}

impl TreeStore {
    pub(super) fn pbuf_cap(&self, id: u32) -> usize {
        let leaf_rows = pbuf_leaf_rows();
        if leaf_rows > 0 && self.len(id) > 0 && self.route_child_is_leaf(id, 0) {
            return leaf_rows.min(self.pbuf.rows);
        }
        let r = self.pbuf.rows;
        if !pbuf_stagger() {
            return r;
        }
        let u = (id.wrapping_mul(2_654_435_761) >> 16) as usize & 0xffff;
        (r / 2 + (r - r / 2) * u / 0x1_0000).max(1)
    }
}

impl Tree {
    fn pbuf_step(
        &mut self,
        x: u32,
        row: u32,
        v: &[f32],
        slack: f32,
    ) -> Result<Option<u32>, BpannError> {
        let s = &mut self.store;
        let slot = nearest(s.block(x), v);
        let child = s.ids(x)[slot];
        if s.is_leaf(child) {
            timed(3, || {
                counted(2, || self.insert_inner(row, v, None, Some((slack, x))))
            })?;
            return Ok(None);
        }
        let dd = v.len();
        let d = dist(v, &s.block(x)[slot * dd..(slot + 1) * dd]);
        let r = if updates_centroid(row) {
            let n_child = s.count(child) + TREE_CENTROID_STRIDE as usize;
            s.set_count(child, n_child);
            let step = (TREE_CENTROID_STRIDE as f32 / n_child as f32).min(1.0);
            mean_update(s.entry_mut(x, slot), v, step);
            (s.radius(child) + d * step).max(d * (1.0 - step) + slack)
        } else {
            s.radius(child).max(d + slack)
        };
        s.set_radius(child, r);
        if fault_split() {
            s.probe_push(child);
        }
        if let Some(target) = self.pbuf_rehome_target(child, v) {
            return Ok(self.pbuf_push_rehomed(target, row, v, slack));
        }
        let s = &mut self.store;
        let n = timed(4, || counted(1, || s.pbuf_push(child, row, v, slack)));
        Ok((n >= s.pbuf_cap(child)).then_some(child))
    }

    pub(super) fn pbuf_add_root(&mut self, row: u32, v: &[f32]) -> Result<(), BpannError> {
        let root = self.header.root_page_id;
        let s = &mut self.store;
        if updates_centroid(row) {
            s.set_count(root, s.count(root) + TREE_CENTROID_STRIDE as usize);
        }
        let n = s.pbuf_push(root, row, v, 0.0);
        self.header.indexed_rows += 1;
        if n >= s.pbuf_cap(root) {
            self.pbuf_flush(root)?;
        }
        Ok(())
    }

    pub(super) fn pbuf_flush_levels(&mut self, id: u32) -> Result<u64, BpannError> {
        if fault_split() {
            let s = &self.store;
            s.probe_flush(id);
            for &c in s.ids(id) {
                if let Some(cold) = s.pbuf_tail_cold(c) {
                    FAULTS[3].fetch_add(1, Ordering::Relaxed);
                    FAULTS[4].fetch_add(u64::from(cold), Ordering::Relaxed);
                }
            }
        }
        let t2 = std::time::Instant::now();
        if pbuf_prefetch() {
            let s = &self.store;
            for &c in s.ids(id) {
                let n = s.pbuf_len(c);
                if n > 0 && !s.is_leaf(c) {
                    s.advise_pbuf(c, n, 1);
                }
            }
            s.advise_pbuf(id, 0, s.pbuf_len(id));
        }
        let rows = counted(0, || self.store.pbuf_take(id));
        add_ns(2, t2);
        let n = rows.len() as u64;
        for (row, v, slack) in rows {
            if let Some(c) = self.pbuf_step(id, row, &v, slack)? {
                self.pbuf_flush(c)?;
            }
        }
        Ok(n)
    }
}

#[cfg(test)]
#[path = "tree_pbuf_all_tests.rs"]
mod tests;
