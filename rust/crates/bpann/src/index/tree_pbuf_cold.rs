use std::sync::atomic::{AtomicU64, Ordering};

use crate::index::tree_pbuf::PAGE;
use crate::index::tree_store::TreeStore;

static COLD: [AtomicU64; 6] = [const { AtomicU64::new(0) }; 6];
static TIMERS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

pub fn take_pbuf_cold() -> Vec<u64> {
    COLD.iter()
        .chain(TIMERS.iter())
        .map(|c| c.swap(0, Ordering::Relaxed))
        .collect()
}

pub fn add_timers() -> bool {
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| std::env::var("ENN_ADD_TIMERS").as_deref() == Ok("1"))
}

pub fn timed<T>(i: usize, f: impl FnOnce() -> T) -> T {
    if !add_timers() {
        return f();
    }
    let t = std::time::Instant::now();
    let out = f();
    TIMERS[i].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    out
}

fn bump(i: usize, n: u64) {
    COLD[i].fetch_add(n, Ordering::Relaxed);
}

pub fn add_ns(i: usize, t: std::time::Instant) {
    if add_timers() {
        TIMERS[i].fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }
}

pub(super) fn cold_pages(base: *const u8, start: usize, len: usize) -> (u64, u64) {
    if len == 0 {
        return (0, 0);
    }
    let addr = base as usize + start;
    let a = addr / PAGE * PAGE;
    let n = (addr + len).div_ceil(PAGE) - a / PAGE;
    let mut v = vec![0u8; n];
    let rc = unsafe { libc::mincore(a as *mut libc::c_void, n * PAGE, v.as_mut_ptr()) };
    if rc != 0 {
        return (0, 0);
    }
    (n as u64, v.iter().filter(|&&b| b & 1 == 0).count() as u64)
}

impl TreeStore {
    pub(super) fn probe_push(&self, id: u32) {
        let (n, rec) = (self.pbuf_len(id), self.pbuf.rec);
        let Some(m) = self.pbuf.map.as_ref() else {
            return;
        };
        if (n * rec).is_multiple_of(PAGE) {
            return;
        }
        let at = id as usize * self.pbuf.page_len() + n * rec;
        let (_, c) = cold_pages(m.as_ptr(), at, 1);
        bump(0, c);
        bump(1, 1);
    }

    pub(super) fn probe_flush(&self, id: u32) {
        let Some(m) = self.pbuf.map.as_ref() else {
            return;
        };
        let len = self.pbuf_len(id) * self.pbuf.rec;
        let (t, c) = cold_pages(m.as_ptr(), id as usize * self.pbuf.page_len(), len);
        bump(2, c);
        bump(3, t);
        let ml = crate::index::tree_store::META_WORDS * 4;
        for &k in self.ids(id) {
            let (t, c) = cold_pages(self.meta.as_ptr(), k as usize * ml, ml);
            bump(4, c);
            bump(5, t);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cold_pages_counts_resident_heap_pages() {
        let v = vec![1u8; 3 * PAGE];
        let (n, cold) = cold_pages(v.as_ptr(), 0, 2 * PAGE);
        assert!(n >= 2);
        assert_eq!(cold, 0);
        assert_eq!(cold_pages(v.as_ptr(), 0, 0), (0, 0));
    }

    #[test]
    fn timers_and_counters_drain() {
        bump(0, 3);
        add_ns(1, std::time::Instant::now());
        assert_eq!(timed(2, || 7), 7);
        let t = take_pbuf_cold();
        assert_eq!(t.len(), 11);
        assert!(t[0] >= 3);
        assert_eq!(take_pbuf_cold()[0], 0);
    }
}
