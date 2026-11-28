use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) const PAGE: u64 = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stream {
    Node = 0,
    LeafRows = 1,
    Scratch = 2,
    StoreAppend = 3,
    StoreRead = 4,
    Prefetch = 5,
}

pub(crate) const STREAM_NAMES: [&str; 6] =
    ["node", "leaf_rows", "scratch", "store_append", "store_read", "prefetch"];

thread_local! {
    static PREFETCHING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

pub(crate) fn as_prefetch<T>(f: impl FnOnce() -> T) -> T {
    let was = PREFETCHING.with(|p| p.replace(true));
    let out = f();
    PREFETCHING.with(|p| p.set(was));
    out
}

const FIELDS: usize = 8;
pub(crate) const SLOW_READ_NS: u64 = 50_000;
static COUNTS: [[AtomicU64; FIELDS]; STREAM_NAMES.len()] =
    [const { [const { AtomicU64::new(0) }; FIELDS] }; STREAM_NAMES.len()];

pub(crate) fn pages_spanned(offset: u64, len: u64) -> u64 {
    if len == 0 {
        return 0;
    }
    (offset + len - 1) / PAGE - offset / PAGE + 1
}

fn add(stream: Stream, base: usize, offset: u64, len: usize) {
    let c = &COUNTS[stream as usize];
    c[base].fetch_add(1, Ordering::Relaxed);
    c[base + 1].fetch_add(len as u64, Ordering::Relaxed);
    c[base + 2].fetch_add(pages_spanned(offset, len as u64), Ordering::Relaxed);
}

pub(crate) fn wrote(stream: Stream, offset: u64, len: usize) {
    add(stream, 0, offset, len);
}

pub(crate) fn read(stream: Stream, offset: u64, len: usize, started: std::time::Instant) {
    let stream = if PREFETCHING.with(std::cell::Cell::get) {
        Stream::Prefetch
    } else {
        stream
    };
    add(stream, 3, offset, len);
    let ns = started.elapsed().as_nanos() as u64;
    let c = &COUNTS[stream as usize];
    c[6].fetch_add(ns, Ordering::Relaxed);
    if ns >= SLOW_READ_NS {
        c[7].fetch_add(1, Ordering::Relaxed);
    }
}

pub(crate) fn slow_reads(stream: Stream) -> u64 {
    COUNTS[stream as usize][7].load(Ordering::Relaxed)
}

pub fn snapshot() -> Vec<(&'static str, [u64; FIELDS])> {
    STREAM_NAMES
        .iter()
        .zip(COUNTS.iter())
        .map(|(name, c)| (*name, std::array::from_fn(|k| c[k].load(Ordering::Relaxed))))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pages_spanned_counts_boundaries() {
        assert_eq!(pages_spanned(0, 0), 0);
        assert_eq!(pages_spanned(0, 1), 1);
        assert_eq!(pages_spanned(0, 4096), 1);
        assert_eq!(pages_spanned(4095, 2), 2);
        assert_eq!(pages_spanned(4096, 8192), 2);
    }

    #[test]
    fn counters_accumulate_per_stream() {
        let before = snapshot()[Stream::Scratch as usize].1;
        wrote(Stream::Scratch, 4000, 200);
        read(Stream::Scratch, 0, 10, std::time::Instant::now());
        let after = snapshot()[Stream::Scratch as usize].1;
        assert!(after[0] > before[0] && after[1] >= before[1] + 200);
        assert!(after[2] >= before[2] + 2 && after[3] > before[3]);
    }

    #[test]
    fn prefetch_reads_are_counted_apart() {
        let pf = |s: Stream| snapshot()[s as usize].1[3];
        let (scratch0, pf0) = (pf(Stream::Scratch), pf(Stream::Prefetch));
        as_prefetch(|| read(Stream::Scratch, 0, 10, std::time::Instant::now()));
        assert!(pf(Stream::Prefetch) > pf0);
        read(Stream::Scratch, 0, 10, std::time::Instant::now());
        assert!(pf(Stream::Scratch) > scratch0);
    }
}
