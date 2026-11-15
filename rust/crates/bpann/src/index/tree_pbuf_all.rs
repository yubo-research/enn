use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::OnceLock;

use crate::error::BpannError;
use crate::index::tree::{mean_update, nearest, updates_centroid, Tree, TREE_CENTROID_STRIDE};
use crate::index::tree_counts::dist;
use crate::index::tree_store::TreeStore;

static ALL: AtomicU8 = AtomicU8::new(u8::MAX);
static STAGGER: AtomicU8 = AtomicU8::new(u8::MAX);

fn flag(over: &AtomicU8, env: &OnceLock<bool>, name: &str) -> bool {
    match over.load(Ordering::Relaxed) {
        u8::MAX => *env.get_or_init(|| std::env::var(name).as_deref() == Ok("1")),
        v => v == 1,
    }
}

pub fn pbuf_all() -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    flag(&ALL, &ENV, "ENN_PBUF_ALL")
}

pub fn pbuf_stagger() -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    flag(&STAGGER, &ENV, "ENN_PBUF_STAGGER")
}

pub fn set_pbuf_all(on: bool) {
    ALL.store(u8::from(on), Ordering::Relaxed);
}

pub fn set_pbuf_stagger(on: bool) {
    STAGGER.store(u8::from(on), Ordering::Relaxed);
}

impl TreeStore {
    pub(super) fn pbuf_cap(&self, id: u32) -> usize {
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
            self.insert_inner(row, v, None, Some((slack, x)))?;
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
        let n = s.pbuf_push(child, row, v);
        Ok((n >= s.pbuf_cap(child)).then_some(child))
    }

    pub(super) fn pbuf_add_root(&mut self, row: u32, v: &[f32]) -> Result<(), BpannError> {
        let root = self.header.root_page_id;
        let s = &mut self.store;
        if updates_centroid(row) {
            s.set_count(root, s.count(root) + TREE_CENTROID_STRIDE as usize);
        }
        let n = s.pbuf_push(root, row, v);
        self.header.indexed_rows += 1;
        if n >= s.pbuf_cap(root) {
            self.pbuf_flush(root)?;
        }
        Ok(())
    }

    pub(super) fn pbuf_flush_levels(&mut self, id: u32) -> Result<u64, BpannError> {
        let rows = self.store.pbuf_take(id);
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
