use std::sync::atomic::{AtomicU64, AtomicU8, Ordering};
use std::sync::OnceLock;

use crate::distance::l2_sq_f32;
use crate::index::tree::{nearest, Tree};
use crate::index::tree_codes::coded_sq_l2;

static REHOME: AtomicU8 = AtomicU8::new(u8::MAX);
static MOVED: AtomicU64 = AtomicU64::new(0);
static CHECKED: AtomicU64 = AtomicU64::new(0);
const REHOME_HOPS: usize = 8;
static AUDIT_SEQ: AtomicU64 = AtomicU64::new(0);

fn rehome_audit() -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    *ENV.get_or_init(|| std::env::var("ENN_PBUF_REHOME_AUDIT").as_deref() == Ok("1"))
}

pub fn pbuf_rehome() -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    match REHOME.load(Ordering::Relaxed) {
        u8::MAX => *ENV.get_or_init(|| std::env::var("ENN_PBUF_REHOME").as_deref() == Ok("1")),
        v => v == 1,
    }
}

pub fn set_pbuf_rehome(on: bool) {
    REHOME.store(u8::from(on), Ordering::Relaxed);
}

pub fn take_pbuf_rehome_stats() -> (u64, u64) {
    (
        CHECKED.swap(0, Ordering::Relaxed),
        MOVED.swap(0, Ordering::Relaxed),
    )
}

impl Tree {
    fn graph_nearest_leaf(&self, parent: u32, v: &[f32]) -> Option<u32> {
        let s = &self.store;
        let kids = s.ids(parent);
        if kids.is_empty() || !s.is_leaf(kids[0]) {
            return None;
        }
        let d = v.len();
        let block = s.block(parent);
        let slot = nearest(block, v);
        let own = kids[slot];
        let qn2: f32 = v.iter().map(|x| x * x).sum();
        let mut best = (l2_sq_f32(v, &block[slot * d..(slot + 1) * d]), own);
        for _ in 0..REHOME_HOPS {
            let at = best.1;
            for (id, _, code) in s.lgraph_entries(at) {
                let (d2, _) = coded_sq_l2(v, qn2, code);
                if d2 < best.0 {
                    best = (d2, id);
                }
            }
            if best.1 == at || !self.live_leaf(best.1) {
                break;
            }
        }
        CHECKED.fetch_add(1, Ordering::Relaxed);
        self.audit(v, own, best.1);
        (best.1 != own).then_some(best.1)
    }

    fn audit(&self, v: &[f32], start: u32, chosen: u32) {
        let n = AUDIT_SEQ.fetch_add(1, Ordering::Relaxed);
        if !rehome_audit() || !n.is_multiple_of(1000) {
            return;
        }
        let cents =
            crate::index::tree_pbuf_diag::leaf_centroids(&self.store, self.header.root_page_id);
        let rc = crate::index::tree_pbuf_diag::audit_rank(chosen, v, &cents);
        let rs = crate::index::tree_pbuf_diag::audit_rank(start, v, &cents);
        crate::index::tree_pbuf_diag::note_audit(n as u32, rc, rs);
    }

    pub(super) fn pbuf_rehome_target(&self, parent: u32, v: &[f32]) -> Option<(u32, u32)> {
        if !pbuf_rehome() || !self.store.pbuf_tagged() {
            return None;
        }
        let leaf = self.graph_nearest_leaf(parent, v)?;
        let s = &self.store;
        let live = self.live_leaf(leaf);
        let home = s
            .parent(leaf)
            .filter(|&p| live && p != parent && !s.is_leaf(p))?;
        s.ids(home).contains(&leaf).then_some((home, leaf))
    }

    pub(super) fn pbuf_push_rehomed(
        &mut self,
        (home, leaf): (u32, u32),
        row: u32,
        v: &[f32],
        slack: f32,
    ) -> Option<u32> {
        MOVED.fetch_add(1, Ordering::Relaxed);
        let n = self.store.pbuf_push(home, row, v, slack);
        self.cover_up_many(leaf, &[v], slack, 0);
        (n >= self.store.pbuf_cap(home)).then_some(home)
    }
}

#[cfg(test)]
#[path = "tree_pbuf_rehome_tests.rs"]
mod tests;
