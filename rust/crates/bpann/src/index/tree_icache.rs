use std::collections::{BTreeSet, HashMap};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::distance::l2_sq_f32;
use crate::index::tree_codes::{code_row_bytes, coded_sq_l2, encode_row};
use crate::index::tree_store::TreeStore;

pub const TREE_INTERNAL_CACHE_BYTES: usize = 64 << 20;

static BUDGET: AtomicUsize = AtomicUsize::new(TREE_INTERNAL_CACHE_BYTES);
static USED: AtomicUsize = AtomicUsize::new(0);
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);

pub fn set_internal_cache_bytes(bytes: usize) {
    BUDGET.store(bytes, Ordering::Relaxed);
}

pub fn internal_cache_bytes() -> usize {
    BUDGET.load(Ordering::Relaxed)
}

pub fn internal_cache_used() -> usize {
    USED.load(Ordering::Relaxed)
}

pub fn take_internal_cache_stats() -> (u64, u64) {
    (
        HITS.swap(0, Ordering::Relaxed),
        MISSES.swap(0, Ordering::Relaxed),
    )
}

struct Entry {
    depth: u16,
    rows: usize,
    bytes: Box<[u8]>,
}

#[derive(Default)]
struct Inner {
    map: HashMap<u32, Entry>,
    order: BTreeSet<(u16, u32)>,
}

impl Inner {
    fn remove(&mut self, id: u32) {
        if let Some(e) = self.map.remove(&id) {
            self.order.remove(&(e.depth, id));
            USED.fetch_sub(e.bytes.len(), Ordering::Relaxed);
        }
    }

    fn make_room(&mut self, bytes: usize, depth: u16) -> bool {
        let budget = internal_cache_bytes();
        while USED.load(Ordering::Relaxed) + bytes > budget {
            match self.order.last().copied() {
                Some((d, id)) if d > depth => self.remove(id),
                _ => return false,
            }
        }
        true
    }
}

#[derive(Default)]
pub struct InternalCache {
    inner: Mutex<Inner>,
}

impl InternalCache {
    pub fn invalidate(&self, id: u32) {
        if let Ok(mut g) = self.inner.lock() {
            g.remove(id);
        }
    }

    fn with_rows(&self, id: u32, rows: usize, f: impl FnOnce(&[u8])) -> bool {
        let Ok(g) = self.inner.lock() else {
            return false;
        };
        match g.map.get(&id) {
            Some(e) if e.rows == rows => {
                f(&e.bytes);
                true
            }
            _ => false,
        }
    }

    fn insert_with(
        &self,
        id: u32,
        depth: u16,
        rows: usize,
        len: usize,
        fill: impl FnOnce(&mut [u8]),
    ) {
        let Ok(mut g) = self.inner.lock() else {
            return;
        };
        g.remove(id);
        if !g.make_room(len, depth) {
            return;
        }
        let mut bytes = vec![0u8; len].into_boxed_slice();
        fill(&mut bytes);
        USED.fetch_add(len, Ordering::Relaxed);
        g.order.insert((depth, id));
        g.map.insert(id, Entry { depth, rows, bytes });
    }
}

impl Drop for InternalCache {
    fn drop(&mut self) {
        if let Ok(g) = self.inner.get_mut() {
            let total: usize = g.map.values().map(|e| e.bytes.len()).sum();
            USED.fetch_sub(total, Ordering::Relaxed);
        }
    }
}

impl TreeStore {
    pub fn internal_scores(
        &self,
        (id, depth): (u32, u16),
        (query, qn2): (&[f32], f32),
        cached: bool,
        mut f: impl FnMut(usize, f32, f32),
    ) {
        let n = self.len(id);
        let rb = code_row_bytes(self.num_dim);
        let hit = cached
            && self.icache.with_rows(id, n, |b| {
                for (slot, row) in b.chunks_exact(rb).enumerate() {
                    let (d2, e) = coded_sq_l2(query, qn2, row);
                    f(slot, d2, e);
                }
            });
        if hit {
            HITS.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let d = query.len().max(1);
        let block = self.block(id);
        for (slot, c) in block.chunks_exact(d).enumerate() {
            f(slot, l2_sq_f32(query, c), 0.0);
        }
        if cached {
            MISSES.fetch_add(1, Ordering::Relaxed);
            self.icache.insert_with(id, depth, n, n * rb, |out| {
                for (o, c) in out.chunks_exact_mut(rb).zip(block.chunks_exact(d)) {
                    encode_row(c, o);
                }
            });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::tree_store::Kind;

    fn scores(s: &TreeStore, id: u32, q: &[f32]) -> Vec<(f32, f32)> {
        let qn2 = q.iter().map(|x| x * x).sum();
        let mut out = Vec::new();
        s.internal_scores((id, 0), (q, qn2), true, |_, d2, e| out.push((d2, e)));
        out
    }

    #[test]
    fn hits_are_within_the_code_bound_and_writes_invalidate() {
        let dir = tempfile::TempDir::new().unwrap();
        let mut s = TreeStore::create(dir.path(), 3).unwrap();
        let node = s.alloc(Kind::Internal).unwrap();
        s.set_entries(node, &[1, 2], &[0.5, -1.0, 2.0, 3.0, 0.25, -0.75]);
        let q = [1.0, 1.0, -1.0];
        let miss = scores(&s, node, &q);
        assert!(miss.iter().all(|&(_, e)| e == 0.0));
        let hit = scores(&s, node, &q);
        for (&(exact, _), &(d2, e)) in miss.iter().zip(&hit) {
            assert!(e > 0.0 && (d2.sqrt() - exact.sqrt()).abs() <= e * 1.001 + 1e-4);
        }
        s.entry_mut(node, 0).copy_from_slice(&[9.0, 9.0, 9.0]);
        let fresh = scores(&s, node, &q);
        assert_eq!(fresh[0], (228.0, 0.0));
        s.push_entry(node, 3, &[0.0, 0.0, 0.0]);
        assert_eq!(scores(&s, node, &q).len(), 3);
    }

    #[test]
    fn deep_entries_make_room_for_shallow_ones() {
        let c = InternalCache::default();
        let budget = internal_cache_bytes();
        c.insert_with(1, 3, 1, 8, |b| b.fill(1));
        c.insert_with(2, 1, 1, 8, |b| b.fill(2));
        assert!(c.with_rows(1, 1, |b| assert_eq!(b[0], 1)));
        assert!(!c.with_rows(1, 2, |_| {}));
        c.invalidate(1);
        assert!(!c.with_rows(1, 1, |_| {}));
        assert!(c.with_rows(2, 1, |b| assert_eq!(b[0], 2)));
        assert!(budget >= 16);
    }
}
