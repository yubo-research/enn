use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Stage {
    Hash,
    Route,
    Leaf,
    Row,
    Distance,
    Search,
}

const STAGES: usize = 6;

#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ProfileStats {
    pub queries: u64,
    pub hash_ns: u64,
    pub route_ns: u64,
    pub leaf_ns: u64,
    pub row_ns: u64,
    pub distance_ns: u64,
    pub search_ns: u64,
    pub split_visits: u64,
    pub leaf_visits: u64,
    pub leaf_pages: u64,
    pub candidates: u64,
    pub coded: u64,
    pub reranked: u64,
}

#[derive(Debug, Default)]
pub(crate) struct QueryProfile {
    enabled: AtomicBool,
    ns: [AtomicU64; STAGES],
    queries: AtomicU64,
    split_visits: AtomicU64,
    leaf_visits: AtomicU64,
    leaf_pages: AtomicU64,
    candidates: AtomicU64,
    coded: AtomicU64,
    reranked: AtomicU64,
}

impl QueryProfile {
    pub(crate) fn set_enabled(&self, on: bool) {
        self.enabled.store(on, Ordering::Relaxed);
    }

    pub(crate) fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    pub(crate) fn start(&self) -> Option<Instant> {
        self.enabled().then(Instant::now)
    }

    pub(crate) fn stop(&self, stage: Stage, t0: Option<Instant>) {
        if let Some(t0) = t0 {
            self.ns[stage as usize].fetch_add(t0.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }

    pub(crate) fn count_query(&self, candidates: usize) {
        if self.enabled() {
            self.queries.fetch_add(1, Ordering::Relaxed);
            self.candidates.fetch_add(candidates as u64, Ordering::Relaxed);
        }
    }

    pub(crate) fn count_visits(&self, splits: u64, leaves: u64, pages: u64) {
        if self.enabled() {
            self.split_visits.fetch_add(splits, Ordering::Relaxed);
            self.leaf_visits.fetch_add(leaves, Ordering::Relaxed);
            self.leaf_pages.fetch_add(pages, Ordering::Relaxed);
        }
    }

    pub(crate) fn count_codes(&self, coded: usize, reranked: usize) {
        if self.enabled() {
            self.coded.fetch_add(coded as u64, Ordering::Relaxed);
            self.reranked.fetch_add(reranked as u64, Ordering::Relaxed);
        }
    }

    pub(crate) fn take(&self) -> ProfileStats {
        let get = |a: &AtomicU64| a.swap(0, Ordering::Relaxed);
        let ns = |s: Stage| get(&self.ns[s as usize]);
        ProfileStats {
            queries: get(&self.queries),
            hash_ns: ns(Stage::Hash),
            route_ns: ns(Stage::Route),
            leaf_ns: ns(Stage::Leaf),
            row_ns: ns(Stage::Row),
            distance_ns: ns(Stage::Distance),
            search_ns: ns(Stage::Search),
            split_visits: get(&self.split_visits),
            leaf_visits: get(&self.leaf_visits),
            leaf_pages: get(&self.leaf_pages),
            candidates: get(&self.candidates),
            coded: get(&self.coded),
            reranked: get(&self.reranked),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_profile_records_nothing_and_take_resets() {
        let p = QueryProfile::default();
        assert!(p.start().is_none());
        p.count_query(5);
        assert_eq!(p.take(), ProfileStats::default());
        p.set_enabled(true);
        let t0 = p.start();
        std::thread::sleep(std::time::Duration::from_millis(2));
        p.stop(Stage::Row, t0);
        p.count_query(7);
        p.count_visits(3, 2, 4);
        p.count_codes(6, 2);
        let s = p.take();
        assert!(s.row_ns >= 2_000_000);
        assert_eq!((s.queries, s.candidates, s.split_visits, s.leaf_visits, s.leaf_pages, s.coded, s.reranked), (1, 7, 3, 2, 4, 6, 2));
        assert_eq!(p.take(), ProfileStats::default());
    }
}
