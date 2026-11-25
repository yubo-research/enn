use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::cache_shard::Shard;
use super::forest_node::{Cut, Dir, Node};
use super::lsh::mix64;

const SHARDS: usize = 16;
pub(crate) const ENTRY_OVERHEAD: usize = 96;
pub(crate) const NODE_CLASS: usize = 0;
pub(crate) const ROW_CLASS: usize = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum CacheKey {
    Node { file: u64, id: u32 },
    Row { column: u8, row: u32 },
}

impl CacheKey {
    pub(crate) fn class(&self) -> usize {
        match self {
            Self::Node { .. } => NODE_CLASS,
            Self::Row { .. } => ROW_CLASS,
        }
    }

    pub(crate) fn shard(&self) -> usize {
        let h = match *self {
            Self::Node { file, id } => mix64((file << 32) ^ u64::from(id)),
            Self::Row { column, row } => mix64(0x9e37_79b9_7f4a_7c15 ^ (u64::from(column) << 40) ^ u64::from(row)),
        };
        (h % SHARDS as u64) as usize
    }
}

#[derive(Debug, Clone)]
pub(crate) enum Cached {
    Node(Arc<Node>),
    Row(Arc<[f64]>),
}

impl Cached {
    pub(crate) fn bytes(&self) -> usize {
        ENTRY_OVERHEAD
            + match self {
                Self::Node(n) => match &**n {
                    Node::Leaf { ids, .. } => std::mem::size_of::<Node>() + 4 * ids.capacity(),
                    Node::Split { cut: Cut::Dir(d), .. } => std::mem::size_of::<Node>() + std::mem::size_of::<Dir>() + d.payload_bytes(),
                    _ => std::mem::size_of::<Node>(),
                },
                Self::Row(r) => 8 * r.len(),
            }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CacheStats {
    pub budget_bytes: usize,
    pub node_budget_bytes: usize,
    pub node_hits: u64,
    pub node_misses: u64,
    pub row_hits: u64,
    pub row_misses: u64,
    pub node_evictions: u64,
    pub row_evictions: u64,
    pub node_bytes: usize,
    pub row_bytes: usize,
    pub node_entries: usize,
    pub row_entries: usize,
}

pub(crate) struct ObjCache {
    shards: Vec<Mutex<Shard>>,
    pool_budgets: Vec<usize>,
    hits: [AtomicU64; 2],
    misses: [AtomicU64; 2],
    evictions: [AtomicU64; 2],
}

impl ObjCache {
    pub(crate) fn new(budget_bytes: usize) -> Self {
        Self::with_pools(vec![budget_bytes])
    }

    pub(crate) fn split(node_bytes: usize, row_bytes: usize) -> Self {
        Self::with_pools(vec![node_bytes, row_bytes])
    }

    fn with_pools(pool_budgets: Vec<usize>) -> Self {
        Self {
            shards: (0..SHARDS * pool_budgets.len()).map(|_| Mutex::new(Shard::default())).collect(),
            pool_budgets,
            hits: Default::default(),
            misses: Default::default(),
            evictions: Default::default(),
        }
    }

    pub(crate) fn caches_rows(&self) -> bool {
        self.pool_budgets.get(ROW_CLASS).or(self.pool_budgets.first()).is_some_and(|&b| b > 0)
    }

    fn pool(&self, key: &CacheKey) -> usize {
        if self.pool_budgets.len() == 1 { 0 } else { key.class() }
    }

    fn shard(&self, key: &CacheKey) -> std::sync::MutexGuard<'_, Shard> {
        self.shards[self.pool(key) * SHARDS + key.shard()].lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn shard_budget(&self, key: &CacheKey) -> usize {
        self.pool_budgets[self.pool(key)] / SHARDS
    }

    pub(crate) fn get_or_load<E>(&self, key: CacheKey, load: impl FnOnce() -> Result<Cached, E>) -> Result<Cached, E> {
        if let Some(v) = self.shard(&key).touch(&key) {
            self.hits[key.class()].fetch_add(1, Ordering::Relaxed);
            return Ok(v);
        }
        self.misses[key.class()].fetch_add(1, Ordering::Relaxed);
        let value = load()?;
        self.put(key, value.clone());
        Ok(value)
    }

    pub(crate) fn put(&self, key: CacheKey, value: Cached) {
        let budget = self.shard_budget(&key);
        self.shard(&key).put(key, value, budget, &self.evictions);
    }

    pub(crate) fn remove(&self, key: &CacheKey) {
        self.shard(key).remove(key);
    }

    pub(crate) fn stats(&self) -> CacheStats {
        let mut s = CacheStats {
            budget_bytes: self.pool_budgets.iter().sum(),
            node_budget_bytes: if self.pool_budgets.len() == 1 { 0 } else { self.pool_budgets[NODE_CLASS] },
            node_hits: self.hits[NODE_CLASS].load(Ordering::Relaxed),
            node_misses: self.misses[NODE_CLASS].load(Ordering::Relaxed),
            row_hits: self.hits[ROW_CLASS].load(Ordering::Relaxed),
            row_misses: self.misses[ROW_CLASS].load(Ordering::Relaxed),
            node_evictions: self.evictions[NODE_CLASS].load(Ordering::Relaxed),
            row_evictions: self.evictions[ROW_CLASS].load(Ordering::Relaxed),
            ..CacheStats::default()
        };
        for shard in &self.shards {
            let g = shard.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
            s.node_bytes += g.bytes[NODE_CLASS];
            s.row_bytes += g.bytes[ROW_CLASS];
            s.node_entries += g.entries[NODE_CLASS];
            s.row_entries += g.entries[ROW_CLASS];
        }
        s
    }
}

pub(crate) type SharedCache = Option<Arc<ObjCache>>;

#[cfg(test)]
mod tests {
    use super::*;

    fn row(v: f64, d: usize) -> Cached {
        Cached::Row(vec![v; d].into())
    }

    fn loaded(c: &ObjCache, r: u32) -> f64 {
        match c.get_or_load::<()>(CacheKey::Row { column: 0, row: r }, || Ok(row(f64::from(r), 4))).unwrap() {
            Cached::Row(v) => v[0],
            Cached::Node(_) => panic!("row expected"),
        }
    }

    #[test]
    fn hits_misses_and_byte_budget_are_tracked() {
        let per = row(0.0, 4).bytes();
        let c = ObjCache::new(SHARDS * per * 3);
        for r in 0..2000 {
            assert_eq!(loaded(&c, r), f64::from(r));
        }
        let s = c.stats();
        assert_eq!((s.row_hits, s.row_misses), (0, 2000));
        assert!(s.row_bytes <= SHARDS * per * 3);
        assert_eq!(s.row_bytes, s.row_entries * per);
        assert_eq!(s.row_evictions as usize + s.row_entries, 2000);
        assert_eq!(loaded(&c, 1999), 1999.0);
        assert_eq!(c.stats().row_hits, 1);
        assert!(c.caches_rows() && ObjCache::split(0, 1).caches_rows());
        assert!(!ObjCache::split(1 << 20, 0).caches_rows());
    }

    #[test]
    fn least_recently_used_entry_is_evicted_first() {
        let per = row(0.0, 4).bytes();
        let c = ObjCache::new(SHARDS * per * 2);
        let same_shard: Vec<u32> = (0..10_000u32)
            .filter(|&r| CacheKey::Row { column: 0, row: r }.shard() == 0)
            .take(3)
            .collect();
        let (a, b, d) = (same_shard[0], same_shard[1], same_shard[2]);
        loaded(&c, a);
        loaded(&c, b);
        loaded(&c, a);
        loaded(&c, d);
        let before = c.stats().row_hits;
        loaded(&c, a);
        loaded(&c, d);
        assert_eq!(c.stats().row_hits, before + 2);
        loaded(&c, b);
        assert_eq!(c.stats().row_hits, before + 2);
    }

    #[test]
    fn put_replaces_and_remove_forgets_nodes() {
        let c = ObjCache::new(1 << 20);
        let key = CacheKey::Node { file: 3, id: 7 };
        c.put(key, Cached::Node(Arc::new(Node::Free { next: 1 })));
        c.put(key, Cached::Node(Arc::new(Node::Leaf { ids: vec![4, 5], next: 0 })));
        let got = c.get_or_load::<()>(key, || panic!("must hit")).unwrap();
        assert!(matches!(got, Cached::Node(n) if *n == Node::Leaf { ids: vec![4, 5], next: 0 }));
        assert_eq!(c.stats().node_entries, 1);
        c.remove(&key);
        assert_eq!(c.stats().node_entries, 0);
        let split = ObjCache::split(1 << 20, SHARDS * row(0.0, 4).bytes());
        split.put(key, Cached::Node(Arc::new(Node::Free { next: 1 })));
        for r in 0..100 {
            loaded(&split, r);
        }
        let s = split.stats();
        assert_eq!((s.node_entries, s.node_evictions, s.node_budget_bytes), (1, 0, 1 << 20));
        assert!(s.row_entries <= SHARDS && s.row_evictions > 0);
        let tiny = ObjCache::new(8);
        tiny.put(key, Cached::Node(Arc::new(Node::Free { next: 0 })));
        assert_eq!(tiny.stats().node_entries, 0);
    }
}
