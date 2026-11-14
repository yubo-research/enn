use std::cell::Cell;

use crate::index::tree_search_knobs::{meta_prefetch, prefetch_pages};
use crate::index::tree_store::TreeStore;

thread_local! {
    static COLD: Cell<bool> = const { Cell::new(true) };
}

#[cfg(target_os = "linux")]
const WHO: libc::c_int = libc::RUSAGE_THREAD;
#[cfg(not(target_os = "linux"))]
const WHO: libc::c_int = libc::RUSAGE_SELF;

fn major_faults() -> i64 {
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        if libc::getrusage(WHO, &mut ru) == 0 {
            ru.ru_majflt
        } else {
            0
        }
    }
}

pub(super) struct ColdProbe {
    start: i64,
    ahead: usize,
}

impl ColdProbe {
    pub(super) fn start() -> Self {
        let ahead = if COLD.with(Cell::get) {
            meta_prefetch()
        } else {
            0
        };
        Self {
            start: major_faults(),
            ahead,
        }
    }

    pub(super) fn ahead(&self) -> usize {
        self.ahead
    }

    pub(super) fn advise(&self, s: &TreeStore, fresh: &mut [(f32, u32)], codes: bool) {
        let mm = self.ahead.min(fresh.len());
        if mm > 0 {
            fresh.select_nth_unstable_by(mm - 1, |a, b| a.0.total_cmp(&b.0));
            fresh[..mm]
                .iter()
                .for_each(|&(_, child)| s.advise_meta(child));
        }
        let m = prefetch_pages().min(fresh.len());
        if m > 0 {
            fresh.select_nth_unstable_by(m - 1, |a, b| a.0.total_cmp(&b.0));
            fresh[..m]
                .iter()
                .for_each(|&(_, child)| s.prefetch_child(child, codes));
        }
    }
}

impl Drop for ColdProbe {
    fn drop(&mut self) {
        COLD.with(|c| c.set(major_faults() > self.start));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_without_faults_marks_thread_warm() {
        let _ = std::thread::spawn(|| {
            assert!(ColdProbe::start().ahead() > 0);
            drop(ColdProbe::start());
            assert_eq!(ColdProbe::start().ahead(), 0);
        })
        .join();
    }
}
