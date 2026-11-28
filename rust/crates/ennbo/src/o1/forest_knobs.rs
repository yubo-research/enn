use std::sync::atomic::Ordering as AtomicOrdering;

use super::cache::SharedCache;
use super::forest::ForestIndex;
use super::forest_tree::Tree;

impl ForestIndex {
    pub(crate) fn set_repair(&mut self, repair_rows: usize, stale_drift: f64, prefer_fresh: bool) {
        self.params.repair_rows = repair_rows;
        self.params.stale_drift = stale_drift.max(0.0);
        self.params.prefer_fresh = prefer_fresh;
    }

    pub(crate) fn set_cache(&mut self, cache: SharedCache) {
        self.trees
            .iter_mut()
            .chain(self.repair.iter_mut().map(|r| &mut r.shadow))
            .for_each(|t| t.nodes.set_cache(cache.clone()));
        self.cache = cache;
    }

    pub(crate) fn batch_hygiene(&self) {
        use super::hygiene::{enabled, Hygiene};
        let (drop, wb) = (enabled(Hygiene::LeafRows), enabled(Hygiene::LeafRowsWriteback));
        if !(drop || wb) {
            return;
        }
        for l in self
            .trees
            .iter()
            .chain(self.repair.iter().map(|r| &r.shadow))
            .filter_map(|t| t.lrows.as_ref())
        {
            if drop {
                l.drop_cached();
            } else {
                l.start_writeback();
            }
        }
    }

    pub(crate) fn hash_scale(&self) -> &[f64] {
        &self.metrics.target
    }

    pub(crate) fn is_trivial(&self) -> bool {
        self.num_rows <= self.params.leaf_cap
    }

    pub(crate) fn memory_bytes(&self) -> usize {
        let d = self.metrics.target.len();
        8 * d * (2 * self.trees.len() + 3) + (self.trees.len() + 1) * std::mem::size_of::<Tree>()
    }

    pub(crate) fn set_budget(&mut self, max_leaves: usize, max_candidates: usize) {
        self.params.max_leaves = max_leaves.max(1);
        self.params.max_candidates = max_candidates.max(1);
    }

    pub(crate) fn set_io_width(&mut self, width: usize, slow_ns: u64) {
        self.params.io_width = width.max(1);
        self.params.io_slow_ns = slow_ns;
    }

    pub(crate) fn set_pread_pivots(&mut self, on: bool) {
        self.params.pread_pivots = on;
    }

    pub(crate) fn node_io(&self) -> (u64, u64) {
        self.trees
            .iter()
            .chain(self.repair.iter().map(|r| &r.shadow))
            .fold(self.retired_io, |(r, w), t| {
                (
                    r + t.nodes.counters.reads.load(AtomicOrdering::Relaxed),
                    w + t.nodes.counters.writes.load(AtomicOrdering::Relaxed),
                )
            })
    }
}
