use std::fs::File;
use std::path::PathBuf;

use memmap2::MmapMut;

use crate::error::BpannError;
use crate::index::tree::TREE_LEAF_CAPACITY;
use crate::index::tree_codes::code_page_bytes;
use crate::index::tree_icache::InternalCache;
use crate::index::tree_residency::Residency;
use crate::index::tree_store_files::map;

pub const TREE_RESIDENT_BUDGET_BYTES: usize = 1 << 30;

pub const SLOT_ENTRIES: usize = TREE_LEAF_CAPACITY + 1;
pub(super) const META_HDR_WORDS: usize = 4;
pub(super) const META_WORDS: usize = META_HDR_WORDS + SLOT_ENTRIES;
const PAGE_BYTES: usize = 4096;
const PAD_QUANTUM: usize = PAGE_BYTES / 4;
pub fn pad_dim(dim: usize) -> usize {
    let padded = dim.div_ceil(PAD_QUANTUM) * PAD_QUANTUM;
    if dim >= PAD_QUANTUM / 2 && padded * 4 <= dim * 5 {
        padded
    } else {
        dim
    }
}
pub(super) const GROW_MIN_SLOTS: usize = 64;
pub(super) const NO_PARENT: u32 = u32::MAX;
pub(super) const LEN_MASK: u32 = 0x00FF_FFFF;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Leaf,
    Internal,
}

pub(super) fn kind_tag(kind: Kind) -> u32 {
    match kind {
        Kind::Leaf => 1 << 24,
        Kind::Internal => 2 << 24,
    }
}
fn mmap_append_forced() -> bool {
    static FORCED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *FORCED.get_or_init(|| std::env::var_os("ENN_MMAP_APPEND").is_some())
}

#[cfg(unix)]
fn append_whole_pages(file: &File, off: usize, v: &[f32]) -> bool {
    use std::os::unix::fs::FileExt;
    if !(v.len() * 4).is_multiple_of(PAGE_BYTES) || mmap_append_forced() {
        return false;
    }
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    file.write_all_at(&bytes, off as u64).is_ok()
}

#[cfg(not(unix))]
fn append_whole_pages(_file: &File, _off: usize, _v: &[f32]) -> bool {
    false
}

pub struct TreeStore {
    pub(super) named: bool,
    pub(super) stamped: bool,
    pub(super) num_dim: usize,
    pub(super) slot_words: usize,
    pub(super) num_pages: usize,
    pub(super) capacity: usize,
    pub(super) dir: PathBuf,
    pub(super) files: (File, File),
    pub(super) meta: MmapMut,
    pub(super) slots: MmapMut,
    pub(super) codes_file: File,
    pub(super) codes: MmapMut,
    pub(super) residency: Residency,
    pub(super) icache: InternalCache,
}

impl TreeStore {
    pub fn num_dim(&self) -> usize {
        self.num_dim
    }

    pub fn num_pages(&self) -> usize {
        self.num_pages
    }

    pub fn file_bytes(&self) -> usize {
        self.meta.len() + self.slots.len()
    }

    pub fn set_resident_budget(&mut self, bytes: usize) {
        self.residency.set_budget(bytes);
        self.update_tracking();
    }

    pub(super) fn update_tracking(&mut self) {
        self.residency
            .update_tracking(self.file_bytes(), self.meta.len());
    }

    pub fn release_resident(&self) {
        self.residency.release(&[&self.meta, &self.slots]);
    }

    pub fn alloc(&mut self, kind: Kind) -> Result<u32, BpannError> {
        self.alloc_tagged(kind_tag(kind))
    }

    pub fn set_kind(&mut self, id: u32, kind: Kind) {
        let m = self.meta_words_mut(id);
        m[0] = kind_tag(kind) | (m[0] & LEN_MASK);
    }

    pub(super) fn grow(&mut self, capacity: usize) -> Result<(), BpannError> {
        self.unstamp();
        self.meta = map(&self.files.0, capacity * META_WORDS * 4)?;
        self.slots = map(&self.files.1, capacity * self.slot_words * 4)?;
        self.codes = map(&self.codes_file, capacity * code_page_bytes(self.num_dim))?;
        self.capacity = capacity;
        self.update_tracking();
        Ok(())
    }

    #[inline]
    pub(super) fn meta_words(&self, id: u32) -> &[u32] {
        let i = id as usize;
        assert!(i < self.num_pages, "tree page {id} out of range");
        unsafe {
            std::slice::from_raw_parts(
                (self.meta.as_ptr() as *const u32).add(i * META_WORDS),
                META_WORDS,
            )
        }
    }

    #[inline]
    pub(super) fn meta_words_mut(&mut self, id: u32) -> &mut [u32] {
        self.unstamp();
        let i = id as usize;
        assert!(i < self.num_pages, "tree page {id} out of range");
        unsafe {
            std::slice::from_raw_parts_mut(
                (self.meta.as_mut_ptr() as *mut u32).add(i * META_WORDS),
                META_WORDS,
            )
        }
    }

    #[inline]
    fn slot_words(&self, id: u32) -> &[u32] {
        let i = id as usize;
        assert!(i < self.num_pages, "tree page {id} out of range");
        self.residency.touch(
            i * self.slot_words * 4,
            self.slot_words * 4,
            &[&self.meta, &self.slots],
        );
        let n = self.slot_words;
        unsafe { std::slice::from_raw_parts((self.slots.as_ptr() as *const u32).add(i * n), n) }
    }

    #[inline]
    fn slot_words_mut(&mut self, id: u32) -> &mut [u32] {
        self.unstamp();
        self.icache.invalidate(id);
        let i = id as usize;
        assert!(i < self.num_pages, "tree page {id} out of range");
        self.residency.touch(
            i * self.slot_words * 4,
            self.slot_words * 4,
            &[&self.meta, &self.slots],
        );
        let n = self.slot_words;
        unsafe {
            std::slice::from_raw_parts_mut((self.slots.as_mut_ptr() as *mut u32).add(i * n), n)
        }
    }

    pub fn prefetch_page(&self, id: u32) {
        let i = id as usize;
        if i < self.num_pages && self.residency.first_advice(id) {
            let bytes = self.len(id) * self.num_dim * 4;
            let _ =
                self.slots
                    .advise_range(memmap2::Advice::WillNeed, i * self.slot_words * 4, bytes);
        }
    }
    pub fn is_leaf(&self, id: u32) -> bool {
        self.meta_words(id)[0] & !LEN_MASK == kind_tag(Kind::Leaf)
    }

    pub fn len(&self, id: u32) -> usize {
        (self.meta_words(id)[0] & LEN_MASK) as usize
    }

    pub fn ids(&self, id: u32) -> &[u32] {
        let n = self.len(id);
        &self.meta_words(id)[META_HDR_WORDS..META_HDR_WORDS + n]
    }

    pub fn block(&self, id: u32) -> &[f32] {
        self.prefetch_page(id);
        let n = self.len(id) * self.num_dim;
        let words = &self.slot_words(id)[..n];
        unsafe { std::slice::from_raw_parts(words.as_ptr() as *const f32, n) }
    }

    pub fn block_mut(&mut self, id: u32) -> &mut [f32] {
        self.set_codes_valid(id, false);
        let n = self.len(id) * self.num_dim;
        let words = &mut self.slot_words_mut(id)[..n];
        unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr() as *mut f32, n) }
    }

    pub fn entry_mut(&mut self, id: u32, slot: usize) -> &mut [f32] {
        let d = self.num_dim;
        &mut self.block_mut(id)[slot * d..(slot + 1) * d]
    }

    pub fn set_entries(&mut self, id: u32, ids: &[u32], block: &[f32]) {
        let n = ids.len();
        assert!(
            n <= SLOT_ENTRIES && block.len() == n * self.num_dim,
            "bad entries for page {id}"
        );
        let words = self.slot_words_mut(id);
        for (w, &v) in words.iter_mut().zip(block) {
            *w = v.to_bits();
        }
        for (slot, v) in block.chunks_exact(self.num_dim).enumerate() {
            self.encode_code_row(id, slot, v);
        }
        self.set_codes_valid(id, true);
        self.set_ids_only(id, ids);
    }

    pub fn set_ids_only(&mut self, id: u32, ids: &[u32]) {
        let n = ids.len();
        assert!(n <= SLOT_ENTRIES, "bad entries for page {id}");
        let m = self.meta_words_mut(id);
        m[META_HDR_WORDS..META_HDR_WORDS + n].copy_from_slice(ids);
        m[0] = (m[0] & !LEN_MASK) | n as u32;
        self.refresh_routes(id);
        if !self.codes_valid(id) {
            self.encode_page(id);
        }
    }

    pub fn push_entry(&mut self, id: u32, entry_id: u32, v: &[f32]) {
        let (n, d) = (self.len(id), self.num_dim);
        assert!(n < SLOT_ENTRIES && v.len() == d, "page {id} is full");
        let off = id as usize * self.slot_words * 4 + n * d * 4;
        self.residency.touch(off, d * 4, &[&self.meta, &self.slots]);
        self.icache.invalidate(id);
        if !append_whole_pages(&self.files.1, off, v) {
            let words = self.slot_words_mut(id);
            for (w, &x) in words[n * d..].iter_mut().zip(v) {
                *w = x.to_bits();
            }
        }
        let m = self.meta_words_mut(id);
        m[META_HDR_WORDS + n] = entry_id;
        m[0] += 1;
        if !self.is_leaf(id) {
            let r = self.radius(entry_id);
            self.set_route(id, n, r);
        }
        if self.codes_valid(id) {
            self.encode_code_row(id, n, v);
        }
    }
    pub fn count(&self, id: u32) -> usize {
        self.meta_words(id)[1] as usize
    }

    pub fn set_count(&mut self, id: u32, n: usize) {
        self.meta_words_mut(id)[1] = n as u32;
    }

    pub fn radius(&self, id: u32) -> f32 {
        f32::from_bits(self.meta_words(id)[2])
    }

    pub fn set_radius(&mut self, id: u32, r: f32) {
        if self.meta_words(id)[2] != r.to_bits() {
            self.icache.child_changed(id);
            self.meta_words_mut(id)[2] = r.to_bits();
            self.route_child_changed(id, r);
        }
    }

    pub fn parent(&self, id: u32) -> Option<u32> {
        Some(self.meta_words(id)[3]).filter(|&p| p != NO_PARENT)
    }

    pub fn set_parent(&mut self, id: u32, parent: Option<u32>) {
        self.meta_words_mut(id)[3] = parent.unwrap_or(NO_PARENT);
        let r = self.radius(id);
        self.route_child_changed(id, r);
    }

    pub fn try_clone(&self) -> Result<Self, BpannError> {
        let mut out = Self::create(&self.dir, self.num_dim)?;
        out.residency.set_budget(self.residency.budget());
        out.grow(self.capacity)?;
        out.num_pages = self.num_pages;
        let (meta_len, slot_len) = (
            self.num_pages * META_WORDS * 4,
            self.num_pages * self.slot_words * 4,
        );
        Residency::copy_released(&self.meta, &mut out.meta, meta_len);
        Residency::copy_released(&self.slots, &mut out.slots, slot_len);
        let code_len = self.num_pages * code_page_bytes(self.num_dim);
        out.codes[..code_len].copy_from_slice(&self.codes[..code_len]);
        Ok(out)
    }
}

#[cfg(test)]
#[path = "tree_store_tests.rs"]
mod tests;

pub use crate::index::tree_store_files::{CODES_FILE, META_FILE, SLOTS_FILE, STAMP_FILE};
