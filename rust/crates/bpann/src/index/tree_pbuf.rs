use std::fs::File;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::OnceLock;

use memmap2::MmapMut;

use crate::error::BpannError;
use crate::index::tree::Tree;
use crate::index::tree_codes::{code_row_bytes, coded_sq_l2, decode_header, encode_row};
use crate::index::tree_counts::dist;
use crate::index::tree_store::{TreeStore, META_HDR_WORDS, META_WORDS};
use crate::index::tree_store_files::{io_err, map, named_file, scratch_file};

pub const PBUF_FILE: &str = "tree_pbuf.bin";
pub(super) const BUF_WORD: usize = META_HDR_WORDS + crate::index::tree_routes::ROUTE_SLOTS - 1;
const PAGE: usize = 4096;
static OVERRIDE: AtomicUsize = AtomicUsize::new(usize::MAX);
static FLUSHES: AtomicU64 = AtomicU64::new(0);
static FLUSH_NS: AtomicU64 = AtomicU64::new(0);
static FLUSH_ROWS: AtomicU64 = AtomicU64::new(0);
static FLUSH_FAULTS: AtomicU64 = AtomicU64::new(0);
static TAKE_FAULTS: AtomicU64 = AtomicU64::new(0);
static FLUSH_LEAVES: AtomicU64 = AtomicU64::new(0);
static FLUSH_SPLITS: AtomicU64 = AtomicU64::new(0);

thread_local! {
    static REPLAY_LEAVES: std::cell::RefCell<Vec<u32>> = const { std::cell::RefCell::new(Vec::new()) };
}

pub(super) fn note_replay_leaf(leaf: u32, split: bool) {
    REPLAY_LEAVES.with(|v| v.borrow_mut().push(leaf));
    if split {
        FLUSH_SPLITS.fetch_add(1, Ordering::Relaxed);
    }
}

pub fn take_pbuf_detail() -> (u64, u64, u64) {
    (
        TAKE_FAULTS.swap(0, Ordering::Relaxed),
        FLUSH_LEAVES.swap(0, Ordering::Relaxed),
        FLUSH_SPLITS.swap(0, Ordering::Relaxed),
    )
}

fn thread_majflt() -> u64 {
    let mut ru: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_THREAD, &mut ru) } == 0 {
        ru.ru_majflt as u64
    } else {
        0
    }
}

pub fn take_pbuf_stats() -> (u64, u64, u64, u64) {
    (
        FLUSHES.swap(0, Ordering::Relaxed),
        FLUSH_ROWS.swap(0, Ordering::Relaxed),
        FLUSH_NS.swap(0, Ordering::Relaxed),
        FLUSH_FAULTS.swap(0, Ordering::Relaxed),
    )
}

pub fn pbuf_rows() -> usize {
    static ENV: OnceLock<usize> = OnceLock::new();
    match OVERRIDE.load(Ordering::Relaxed) {
        usize::MAX => *ENV.get_or_init(|| {
            std::env::var("ENN_PBUF")
                .ok()
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(0)
        }),
        v => v,
    }
}

pub fn pbuf_advise() -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    *ENV.get_or_init(|| std::env::var("ENN_PBUF_ADVISE").as_deref() != Ok("0"))
}

pub fn pbuf_from_root() -> bool {
    static ENV: OnceLock<bool> = OnceLock::new();
    *ENV.get_or_init(|| std::env::var("ENN_PBUF_ROOT").as_deref() == Ok("1"))
}

pub fn set_pbuf_rows(rows: usize) {
    OVERRIDE.store(rows, Ordering::Relaxed);
}

pub struct PBuf {
    pub(super) rows: usize,
    rec: usize,
    file: Option<File>,
    map: Option<MmapMut>,
}

impl PBuf {
    fn page_len(&self) -> usize {
        (self.rows * self.rec).div_ceil(PAGE) * PAGE
    }

    fn sized(rows: usize, num_dim: usize, coded8: bool) -> Self {
        Self {
            rows: if coded8 { rows } else { 0 },
            rec: 4 + code_row_bytes(num_dim),
            file: None,
            map: None,
        }
    }

    pub(super) fn create(
        dir: &std::path::Path,
        named: bool,
        (num_dim, coded8): (usize, bool),
        capacity: usize,
    ) -> Result<Self, BpannError> {
        let mut b = Self::sized(pbuf_rows(), num_dim, coded8);
        if b.rows > 0 {
            b.file = Some(if named {
                named_file(dir, PBUF_FILE)?
            } else {
                scratch_file(dir, "tree_pbuf")?
            });
            b.grow(capacity)?;
        }
        Ok(b)
    }

    pub(super) fn open(
        dir: &std::path::Path,
        (num_dim, coded8): (usize, bool),
        capacity: usize,
    ) -> Option<Self> {
        let mut b = Self::sized(pbuf_rows(), num_dim, coded8);
        if b.rows > 0 {
            b.file = Some(
                std::fs::OpenOptions::new()
                    .read(true)
                    .write(true)
                    .create(true)
                    .truncate(false)
                    .open(dir.join(PBUF_FILE))
                    .ok()?,
            );
            b.grow(capacity).ok()?;
        }
        Some(b)
    }

    pub(super) fn grow(&mut self, capacity: usize) -> Result<(), BpannError> {
        if let Some(f) = &self.file {
            self.map = Some(map(f, capacity * self.page_len())?);
        }
        Ok(())
    }

    pub(super) fn clone_into(
        &self,
        dir: &std::path::Path,
        capacity: usize,
        num_pages: usize,
    ) -> Result<Self, BpannError> {
        let mut b = Self {
            rows: self.rows,
            rec: self.rec,
            file: None,
            map: None,
        };
        if b.rows > 0 {
            b.file = Some(scratch_file(dir, "tree_pbuf")?);
            b.grow(capacity)?;
            b.copy_from(self, num_pages);
        }
        Ok(b)
    }

    fn copy_from(&mut self, other: &Self, num_pages: usize) {
        if let (Some(dst), Some(src)) = (self.map.as_mut(), other.map.as_ref()) {
            let n = (num_pages * other.page_len()).min(dst.len()).min(src.len());
            dst[..n].copy_from_slice(&src[..n]);
        }
    }

    pub(super) fn sync(&self) -> Result<(), BpannError> {
        self.file
            .as_ref()
            .map_or(Ok(()), |f| f.sync_all().map_err(io_err))
    }

    fn region(&self, id: u32, n: usize) -> &[u8] {
        let start = id as usize * self.page_len();
        &self.map.as_ref().expect("pbuf mapped")[start..start + n * self.rec]
    }
}

impl TreeStore {
    #[inline]
    pub fn pbuf_on(&self) -> bool {
        self.pbuf.rows > 0
    }

    pub fn pbuf_len(&self, id: u32) -> usize {
        if !self.pbuf_on() || self.is_leaf(id) {
            return 0;
        }
        self.meta_words(id)[BUF_WORD] as usize
    }

    pub(super) fn clear_pbuf_word(&mut self, id: u32) {
        self.meta_words_mut(id)[BUF_WORD] = 0;
    }

    pub(super) fn pbuf_push(&mut self, id: u32, row: u32, v: &[f32]) -> usize {
        let n = self.pbuf_len(id);
        let (rec, plen) = (self.pbuf.rec, self.pbuf.page_len());
        let at = id as usize * plen + n * rec;
        let m = self.pbuf.map.as_mut().expect("pbuf mapped");
        m[at..at + 4].copy_from_slice(&row.to_le_bytes());
        encode_row(v, &mut m[at + 4..at + rec]);
        self.meta_words_mut(id)[BUF_WORD] = (n + 1) as u32;
        n + 1
    }

    pub fn pbuf_records(&self, id: u32) -> impl Iterator<Item = (u32, &[u8])> + '_ {
        let n = self.pbuf_len(id);
        let rec = self.pbuf.rec;
        let bytes: &[u8] = if n == 0 { &[] } else { self.pbuf.region(id, n) };
        bytes
            .chunks_exact(rec)
            .map(|r| (u32::from_le_bytes([r[0], r[1], r[2], r[3]]), &r[4..]))
    }

    pub fn pbuf_ids(&self, id: u32) -> Vec<u32> {
        self.pbuf_records(id).map(|r| r.0).collect()
    }

    pub(super) fn advise_pbuf(&self, id: u32) {
        let n = self.pbuf_len(id);
        if let (true, Some(m)) = (n > 0, self.pbuf.map.as_ref()) {
            let start = id as usize * self.pbuf.page_len();
            let _ = m.advise_range(memmap2::Advice::WillNeed, start, n * self.pbuf.rec);
        }
    }

    pub(super) fn advise_children(&self, id: u32) {
        let mut kids = self.ids(id).to_vec();
        kids.sort_unstable();
        let (cl, ml) = (self.code_page_len(), META_WORDS * 4);
        let mut i = 0;
        while i < kids.len() {
            let mut j = i;
            while j + 1 < kids.len() && kids[j + 1] <= kids[j] + 2 {
                j += 1;
            }
            let (a, b) = (kids[i] as usize, kids[j] as usize + 1);
            let _ = self
                .codes
                .advise_range(memmap2::Advice::WillNeed, a * cl, (b - a) * cl);
            let _ = self
                .meta
                .advise_range(memmap2::Advice::WillNeed, a * ml, (b - a) * ml);
            i = j + 1;
        }
    }

    fn decode_code(&self, code: &[u8], out: &mut Vec<f32>) {
        let (scale, _, c) = decode_header(code);
        out.clear();
        out.extend(c.iter().map(|&x| scale * f32::from(x)));
    }

    pub(super) fn pbuf_max_dist(&self, id: u32, c: &[f32]) -> f32 {
        let half = 0.5 * (self.num_dim as f32).sqrt();
        let mut v = Vec::with_capacity(self.num_dim);
        self.pbuf_records(id).fold(0.0, |m, (_, code)| {
            self.decode_code(code, &mut v);
            m.max(dist(&v, c) + half * decode_header(code).0)
        })
    }

    fn pbuf_take(&mut self, id: u32) -> Vec<(u32, Vec<f32>, f32)> {
        let half = 0.5 * (self.num_dim as f32).sqrt();
        let mut out = Vec::with_capacity(self.pbuf_len(id));
        for (row, code) in self.pbuf_records(id) {
            let mut v = Vec::with_capacity(self.num_dim);
            self.decode_code(code, &mut v);
            out.push((row, v, half * decode_header(code).0));
        }
        if !out.is_empty() {
            self.clear_pbuf_word(id);
        }
        out
    }

    pub(super) fn pbuf_rescale(&mut self, id: u32, ratio: &[f64]) {
        let rows = self.pbuf_take(id);
        for (row, mut v, _) in rows {
            for (x, &r) in v.iter_mut().zip(ratio) {
                *x = (f64::from(*x) * r) as f32;
            }
            self.pbuf_push(id, row, &v);
        }
    }

    pub(super) fn pbuf_scan(&self, id: u32, query: &[f32], qn2: f32) -> Vec<(u32, f32, f32)> {
        self.pbuf_records(id)
            .map(|(row, code)| {
                let (d2, e) = coded_sq_l2(query, qn2, code);
                (row, d2, e)
            })
            .collect()
    }
}

impl Tree {
    pub(super) fn pbuf_flush(&mut self, id: u32) -> Result<(), BpannError> {
        let (t, f0) = (std::time::Instant::now(), thread_majflt());
        if pbuf_advise() {
            self.store.advise_children(id);
        }
        let rows = self.store.pbuf_take(id);
        TAKE_FAULTS.fetch_add(thread_majflt() - f0, Ordering::Relaxed);
        FLUSH_ROWS.fetch_add(rows.len() as u64, Ordering::Relaxed);
        let start = if pbuf_from_root() { u32::MAX } else { id };
        for (row, v, slack) in rows {
            self.insert_inner(row, &v, None, Some((slack, start)))?;
        }
        FLUSHES.fetch_add(1, Ordering::Relaxed);
        REPLAY_LEAVES.with(|v| {
            let mut v = v.borrow_mut();
            v.sort_unstable();
            v.dedup();
            FLUSH_LEAVES.fetch_add(v.len() as u64, Ordering::Relaxed);
            v.clear();
        });
        FLUSH_FAULTS.fetch_add(thread_majflt() - f0, Ordering::Relaxed);
        FLUSH_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
        Ok(())
    }

    pub fn pbuf_flush_subtree(&mut self, root: u32) -> Result<(), BpannError> {
        if !self.store.pbuf_on() {
            return Ok(());
        }
        let mut stack = vec![root];
        let mut internal = Vec::new();
        while let Some(p) = stack.pop() {
            if !self.store.is_leaf(p) && !self.store.is_free(p) {
                internal.push(p);
                stack.extend(self.store.ids(p).iter().copied());
            }
        }
        for p in internal {
            if !self.store.is_free(p) && !self.store.is_leaf(p) {
                self.pbuf_flush(p)?;
            }
        }
        Ok(())
    }

    pub fn pbuf_flush_all(&mut self) -> Result<(), BpannError> {
        if self.num_pages() == 0 {
            return Ok(());
        }
        self.pbuf_flush_subtree(self.header.root_page_id)
    }

    pub fn pbuf_row_ids(&self) -> Vec<u32> {
        (0..self.num_pages() as u32)
            .filter(|&p| !self.store.is_free(p))
            .flat_map(|p| self.store.pbuf_ids(p))
            .collect()
    }
}

pub(super) fn score_pbuf(
    s: &TreeStore,
    page_id: u32,
    (query, qn2): (&[f32], f32),
    coded: Option<&mut crate::index::tree_codes_search::CodedScan>,
    top: &mut crate::index::tree_topk::TopK,
    score: &mut impl FnMut(u32, Option<&[f32]>, f64) -> Result<Option<f64>, BpannError>,
) -> Result<(), BpannError> {
    use crate::index::tree_search::XP_LEAF_ROWS;
    if s.pbuf_len(page_id) == 0 {
        return Ok(());
    }
    s.advise_pbuf(page_id);
    if let Some(scan) = coded {
        let n = scan.scan_pbuf(s, page_id, query, qn2, &mut |i, d2| top.offer(i, d2));
        XP_LEAF_ROWS.fetch_add(n as u64, Ordering::Relaxed);
        return Ok(());
    }
    let ids = s.pbuf_ids(page_id);
    XP_LEAF_ROWS.fetch_add(ids.len() as u64, Ordering::Relaxed);
    for row in ids {
        if let Some(distance) = score(row, None, top.limit())? {
            top.offer(row, distance);
        }
    }
    Ok(())
}

#[cfg(test)]
#[path = "tree_pbuf_tests.rs"]
mod tests;
