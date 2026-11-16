use std::cell::RefCell;
use std::sync::atomic::{AtomicBool, Ordering};

pub type WalkTrace = (Vec<(u32, u32, f32)>, Vec<(u32, u32, u32)>);

static ENABLED: AtomicBool = AtomicBool::new(false);

#[derive(Default)]
struct Trace {
    walk: u32,
    ordinal: u32,
    leaves: Vec<(u32, u32, f32)>,
    rows: Vec<(u32, u32, u32)>,
}

thread_local! {
    static TRACE: RefCell<Trace> = RefCell::new(Trace::default());
}

pub fn set_walk_trace(on: bool) {
    ENABLED.store(on, Ordering::Relaxed);
    TRACE.with(|t| *t.borrow_mut() = Trace::default());
}

pub(super) fn begin_walk() {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    TRACE.with(|t| {
        let mut t = t.borrow_mut();
        t.walk += 1;
        t.ordinal = 0;
    });
}

pub(super) fn note_leaf(bound: f32, ids: &[u32]) {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    TRACE.with(|t| {
        let mut t = t.borrow_mut();
        let (w, o) = (t.walk, t.ordinal);
        t.leaves.push((w, o, bound));
        t.rows.extend(ids.iter().map(|&r| (w, o, r)));
        t.ordinal += 1;
    });
}

pub fn take_walk_trace() -> WalkTrace {
    TRACE.with(|t| {
        let mut t = t.borrow_mut();
        let out = (std::mem::take(&mut t.leaves), std::mem::take(&mut t.rows));
        t.walk = 0;
        out
    })
}
