use std::fs::File;
use std::sync::atomic::{AtomicU64, Ordering};

pub(crate) const TOKENS: [&str; 2] = ["lrows", "lrows_wb"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Hygiene {
    LeafRows = 0,
    LeafRowsWriteback = 1,
}

fn parse(spec: &str) -> [bool; TOKENS.len()] {
    let mut on = [false; TOKENS.len()];
    for t in spec.split(',').map(str::trim).filter(|t| !t.is_empty()) {
        if let Some(k) = TOKENS.iter().position(|&n| n == t) {
            on[k] = true;
        } else {
            panic!("ENN_HYGIENE: unknown token {t:?} (expected {TOKENS:?})");
        }
    }
    on
}

pub(crate) fn enabled(h: Hygiene) -> bool {
    static ON: std::sync::OnceLock<[bool; TOKENS.len()]> = std::sync::OnceLock::new();
    ON.get_or_init(|| parse(&std::env::var("ENN_HYGIENE").unwrap_or_default()))[h as usize]
}

static COUNTS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

pub fn counts() -> [u64; 4] {
    std::array::from_fn(|k| COUNTS[k].load(Ordering::Relaxed))
}

#[cfg(target_os = "linux")]
mod sys {
    use std::os::fd::AsRawFd;

    extern "C" {
        fn posix_fadvise(fd: i32, offset: i64, len: i64, advice: i32) -> i32;
        fn sync_file_range(fd: i32, offset: i64, nbytes: i64, flags: u32) -> i32;
    }
    const SYNC_FILE_RANGE_WRITE: u32 = 2;
    const POSIX_FADV_DONTNEED: i32 = 4;

    pub(super) fn dont_need(file: &std::fs::File) -> bool {
        unsafe { posix_fadvise(file.as_raw_fd(), 0, 0, POSIX_FADV_DONTNEED) == 0 }
    }

    pub(super) fn start_writeback(file: &std::fs::File) -> bool {
        unsafe { sync_file_range(file.as_raw_fd(), 0, 0, SYNC_FILE_RANGE_WRITE) == 0 }
    }
}

#[cfg(not(target_os = "linux"))]
mod sys {
    pub(super) fn dont_need(_: &std::fs::File) -> bool {
        false
    }

    pub(super) fn start_writeback(_: &std::fs::File) -> bool {
        false
    }
}

pub(crate) fn drop_cached(file: &File) {
    COUNTS[0].fetch_add(1, Ordering::Relaxed);
    if !sys::dont_need(file) {
        COUNTS[1].fetch_add(1, Ordering::Relaxed);
    }
}

pub(crate) fn start_writeback(file: &File) {
    COUNTS[2].fetch_add(1, Ordering::Relaxed);
    if !sys::start_writeback(file) {
        COUNTS[3].fetch_add(1, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::os::unix::fs::FileExt;

    #[test]
    fn parse_accepts_tokens() {
        assert_eq!(parse(""), [false, false]);
        assert_eq!(parse("lrows"), [true, false]);
        assert_eq!(parse(" lrows , lrows_wb"), [true, true]);
        assert_eq!(parse("lrows_wb"), [false, true]);
    }

    #[test]
    #[should_panic(expected = "unknown token")]
    fn parse_rejects_unknown_tokens() {
        parse("lrows,bogus");
    }

    #[test]
    fn drop_cached_keeps_contents() {
        let mut f = tempfile::tempfile().expect("tempfile");
        f.write_all(&[1, 2, 3, 4]).expect("write");
        drop_cached(&f);
        start_writeback(&f);
        let mut buf = [0u8; 4];
        f.read_exact_at(&mut buf, 0).expect("read");
        assert_eq!(buf, [1, 2, 3, 4]);
    }
}
