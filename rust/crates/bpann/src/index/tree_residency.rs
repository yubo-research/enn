//! Bounded residency for memory-mapped files ([`crate::index::tree_store`] and
//! [`crate::mmap_store`]).
//!
//! Tracking is on only while the mapped files are larger than the budget. A read
//! fault on a file mapping maps up to 64 KiB of neighboring cached pages at once
//! (fault-around), so residency is tracked in [`UNIT_BYTES`] windows: each touched
//! window sets one bit of a fixed-size bitmap (by hash, so the bitmap does not grow
//! with the files), and a newly set bit adds a window to an upper bound on the
//! resident bytes. A small file the caller does not track (the tree's page records)
//! is counted as wholly resident. When the bound exceeds the budget, every page of the
//! mappings is released with `MADV_DONTNEED`, which for a shared file mapping keeps
//! the data (dirty pages stay in the page cache and later accesses fault them back
//! in) and only uncharges the pages from the process.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use memmap2::{MmapMut, UncheckedAdvice};

/// Resident bytes of mapped pages an observation store keeps before releasing them.
pub const DEFAULT_RESIDENT_BUDGET_BYTES: usize = 256 << 20;
/// Bytes tracked as one unit: the largest run of pages one fault maps.
pub const UNIT_BYTES: usize = 64 << 10;
const TOUCH_BITS_LOG2: u32 = 23;
const COPY_CHUNK_BYTES: usize = 8 << 20;

pub struct Residency {
    bits: Vec<AtomicU64>,
    bytes: AtomicUsize,
    budget: usize,
    reserved: usize,
    tracking: bool,
}

fn release_map(map: &MmapMut) {
    if !map.is_empty() {
        let _ = unsafe { map.unchecked_advise(UncheckedAdvice::DontNeed) };
    }
}

impl Residency {
    pub fn new(budget: usize) -> Self {
        Self {
            bits: Vec::new(),
            bytes: AtomicUsize::new(0),
            budget,
            reserved: 0,
            tracking: false,
        }
    }

    pub fn budget(&self) -> usize {
        self.budget
    }

    pub fn set_budget(&mut self, bytes: usize) {
        self.budget = bytes;
    }

    pub fn tracking(&self) -> bool {
        self.tracking
    }

    /// Track while `mapped_bytes` exceed the budget, counting `reserved` of them as
    /// always resident; the bitmap is allocated the first time it is needed.
    pub fn update_tracking(&mut self, mapped_bytes: usize, reserved: usize) {
        self.reserved = reserved;
        let on = mapped_bytes > self.budget;
        if on && self.bits.is_empty() {
            self.bits = (0..(1usize << TOUCH_BITS_LOG2) / 64).map(|_| AtomicU64::new(0)).collect();
        }
        self.tracking = on;
    }

    /// Upper bound on the tracked bytes made resident since the last release.
    pub fn estimate(&self) -> usize {
        self.bytes.load(Ordering::Relaxed)
    }

    /// Record an access to bytes `offset..offset + len` of the tracked file;
    /// release `maps` once the tracked and reserved bytes exceed the budget.
    #[inline]
    pub fn touch(&self, offset: usize, len: usize, maps: &[&MmapMut]) {
        if self.tracking && len > 0 {
            self.touch_tracked(offset, len, maps);
        }
    }

    #[inline(never)]
    fn touch_tracked(&self, offset: usize, len: usize, maps: &[&MmapMut]) {
        for unit in offset / UNIT_BYTES..=(offset + len - 1) / UNIT_BYTES {
            let key = unit as u64;
            let bit = (key.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - TOUCH_BITS_LOG2)) as usize;
            let (word, mask) = (&self.bits[bit / 64], 1u64 << (bit % 64));
            if word.load(Ordering::Relaxed) & mask != 0 || word.fetch_or(mask, Ordering::Relaxed) & mask != 0 {
                continue;
            }
            if self.bytes.fetch_add(UNIT_BYTES, Ordering::Relaxed) + UNIT_BYTES + self.reserved > self.budget {
                self.release(maps);
            }
        }
    }

    /// Release every resident page of `maps` and forget what was touched.
    pub fn release(&self, maps: &[&MmapMut]) {
        maps.iter().for_each(|m| release_map(m));
        self.forget();
    }

    /// Forget what was touched (after the caller dropped its resident pages).
    pub fn forget(&self) {
        self.bits.iter().for_each(|w| w.store(0, Ordering::Relaxed));
        self.bytes.store(0, Ordering::Relaxed);
    }

    /// Copy the first `len` bytes of `src` into `dst`, releasing each chunk of both
    /// mappings once copied.
    pub fn copy_released(src: &MmapMut, dst: &mut MmapMut, len: usize) {
        let mut off = 0;
        while off < len {
            let end = (off + COPY_CHUNK_BYTES).min(len);
            dst[off..end].copy_from_slice(&src[off..end]);
            unsafe {
                let _ = src.unchecked_advise_range(UncheckedAdvice::DontNeed, off, end - off);
                let _ = dst.unchecked_advise_range(UncheckedAdvice::DontNeed, off, end - off);
            }
            off = end;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn maps(bytes: usize) -> (tempfile::NamedTempFile, MmapMut) {
        let f = tempfile::NamedTempFile::new().unwrap();
        f.as_file().set_len(bytes as u64).unwrap();
        let m = unsafe { MmapMut::map_mut(f.as_file()) }.unwrap();
        (f, m)
    }

    #[test]
    fn touches_count_each_window_once_and_release_past_the_budget() {
        let (_a, mut m) = maps(1 << 20);
        m[100] = 7;
        let mut r = Residency::new(1 << 30);
        r.update_tracking(1 << 20, 0);
        assert!(!r.tracking());
        r.touch(0, 10, &[&m]);
        assert_eq!(r.estimate(), 0);
        r.set_budget(7 * UNIT_BYTES / 2);
        r.update_tracking(1 << 20, UNIT_BYTES);
        assert!(r.tracking() && r.budget() == 7 * UNIT_BYTES / 2);
        r.touch(0, 10, &[&m]);
        r.touch(100, 10, &[&m]);
        assert_eq!(r.estimate(), UNIT_BYTES);
        r.touch(3 * UNIT_BYTES, 10, &[&m]);
        assert_eq!(r.estimate(), 2 * UNIT_BYTES);
        r.touch(UNIT_BYTES - 1, 2, &[&m]);
        assert_eq!(r.estimate(), 0, "a range across two windows adds both; with the reserved bytes past the budget it releases");
        assert_eq!(m[100], 7, "a release keeps the data");
        r.touch(0, 1, &[&m]);
        assert_eq!(r.estimate(), UNIT_BYTES);
        r.forget();
        assert_eq!(r.estimate(), 0);
    }

    #[test]
    fn copy_released_copies_every_byte() {
        let ((_a, mut src), (_b, mut dst)) = (maps(COPY_CHUNK_BYTES + 4096), maps(COPY_CHUNK_BYTES + 8192));
        src.iter_mut().enumerate().for_each(|(i, b)| *b = (i % 251) as u8);
        Residency::copy_released(&src, &mut dst, src.len());
        assert!(src.iter().zip(dst.iter()).all(|(a, b)| a == b));
        release_map(&dst);
        assert_eq!(dst[COPY_CHUNK_BYTES + 5], ((COPY_CHUNK_BYTES + 5) % 251) as u8);
    }
}
