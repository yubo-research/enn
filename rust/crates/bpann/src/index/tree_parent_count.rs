use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::index::tree_store::TreeStore;

static FORCED: AtomicBool = AtomicBool::new(false);
static WALKS: AtomicU64 = AtomicU64::new(0);
static PARENTS: AtomicU64 = AtomicU64::new(0);
static LEAVES: AtomicU64 = AtomicU64::new(0);
static LEAF_PARENT_EXPANDS: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static SEEN: RefCell<Vec<u32>> = const { RefCell::new(Vec::new()) };
}

pub fn count_parents_enabled() -> bool {
    FORCED.load(Ordering::Relaxed)
}

pub fn set_count_parents(value: bool) {
    FORCED.store(value, Ordering::Relaxed);
}

pub(super) fn note_leaf(s: &TreeStore, page: u32) {
    if !count_parents_enabled() {
        return;
    }
    let p = s.parent(page).unwrap_or(u32::MAX);
    SEEN.with(|v| v.borrow_mut().push(p));
}

pub(super) fn note_expand(s: &TreeStore, page: u32) {
    if !count_parents_enabled() {
        return;
    }
    let ids = s.ids(page);
    if ids.first().is_some_and(|&c| s.is_leaf(c)) {
        LEAF_PARENT_EXPANDS.fetch_add(1, Ordering::Relaxed);
    }
}

pub(super) fn end_walk() {
    if !count_parents_enabled() {
        return;
    }
    SEEN.with(|v| {
        let mut v = v.borrow_mut();
        LEAVES.fetch_add(v.len() as u64, Ordering::Relaxed);
        v.sort_unstable();
        v.dedup();
        PARENTS.fetch_add(v.len() as u64, Ordering::Relaxed);
        v.clear();
    });
    WALKS.fetch_add(1, Ordering::Relaxed);
}

pub fn take_parent_counts() -> (u64, u64, u64, u64) {
    (
        WALKS.swap(0, Ordering::Relaxed),
        PARENTS.swap(0, Ordering::Relaxed),
        LEAVES.swap(0, Ordering::Relaxed),
        LEAF_PARENT_EXPANDS.swap(0, Ordering::Relaxed),
    )
}
