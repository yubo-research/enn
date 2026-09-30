//! Disk-backed storage of the incremental tree, so its size is limited by disk, not RAM.
//!
//! Each page has a 16-byte record in a metadata file (kind, entry count, subtree row
//! count, radius, parent) and a fixed-size slot in a slot file with room for
//! [`SLOT_ENTRIES`] ids (a leaf's row ids or an internal page's child page ids) and as
//! many `num_dim`-float vectors (a leaf's scaled row coordinates or an internal page's
//! running-mean child centroids). Page ids are slot numbers.
//!
//! While both regions fit in the residency budget ([`TREE_RESIDENT_BUDGET_BYTES`])
//! they live in anonymous memory, which is cheaper to fault in. Past the budget they
//! move to scratch files, memory-mapped and unlinked at once, so they last as long as
//! the store; the kernel writes their pages back to disk and evicts them as it needs
//! memory. Pages the process touches stay charged to it until released, so the store
//! then tracks which parts of the slot file were touched, counts the small metadata
//! file as wholly resident, and drops every resident page when the two exceed the
//! budget ([`crate::index::tree_residency`]). Resident memory is bounded by the budget
//! whatever the number of rows.

use std::fs::{self, File, OpenOptions};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use memmap2::MmapMut;

use crate::error::BpannError;
use crate::index::tree::TREE_LEAF_CAPACITY;
use crate::index::tree_residency::Residency;

/// Resident bytes of tree pages the store keeps before releasing them. A search
/// batch touches most of the tree, so a budget below the tree's size makes every
/// batch fault much of it back in.
pub const TREE_RESIDENT_BUDGET_BYTES: usize = 1 << 30;

/// Entries a slot holds: a full leaf plus the row that makes it split.
pub const SLOT_ENTRIES: usize = TREE_LEAF_CAPACITY + 1;

const META_WORDS: usize = 4;
const GROW_MIN_SLOTS: usize = 64;
const NO_PARENT: u32 = u32::MAX;
const LEN_MASK: u32 = 0x00FF_FFFF;

/// What a page holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Leaf,
    Internal,
}

static SCRATCH_SEQ: AtomicUsize = AtomicUsize::new(0);

fn io_err(e: std::io::Error) -> BpannError {
    BpannError::InvalidParameter(e.to_string())
}

/// A scratch file in `dir`, unlinked once open (where the platform allows it).
fn scratch_file(dir: &Path, tag: &str) -> Result<File, BpannError> {
    fs::create_dir_all(dir).map_err(io_err)?;
    let seq = SCRATCH_SEQ.fetch_add(1, Ordering::Relaxed);
    let path = dir.join(format!("{tag}.{}.{seq}.scratch", std::process::id()));
    let file = OpenOptions::new().read(true).write(true).create_new(true).open(&path).map_err(io_err)?;
    let _ = fs::remove_file(&path);
    Ok(file)
}

fn kind_tag(kind: Kind) -> u32 {
    match kind {
        Kind::Leaf => 1 << 24,
        Kind::Internal => 2 << 24,
    }
}

/// `bytes` of `file`, or of anonymous memory without a file.
fn map(file: Option<&File>, bytes: usize) -> Result<MmapMut, BpannError> {
    match file {
        Some(f) => {
            f.set_len(bytes as u64).map_err(io_err)?;
            unsafe { MmapMut::map_mut(f) }.map_err(io_err)
        }
        None => MmapMut::map_anon(bytes).map_err(io_err),
    }
}

/// Enlarge anonymous memory `region` to `bytes`, keeping its contents.
fn grow_anon(region: &mut MmapMut, bytes: usize) -> Result<(), BpannError> {
    #[cfg(target_os = "linux")]
    {
        unsafe { region.remap(bytes, memmap2::RemapOptions::new().may_move(true)) }.map_err(io_err)
    }
    #[cfg(not(target_os = "linux"))]
    {
        let mut out = map(None, bytes)?;
        out[..region.len()].copy_from_slice(region);
        *region = out;
        Ok(())
    }
}

pub struct TreeStore {
    num_dim: usize,
    slot_words: usize,
    num_pages: usize,
    capacity: usize,
    dir: PathBuf,
    files: Option<(File, File)>,
    meta: MmapMut,
    slots: MmapMut,
    residency: Residency,
}

impl TreeStore {
    /// An empty store whose scratch files (once it outgrows the budget) live in `dir`.
    pub fn create(dir: &Path, num_dim: usize) -> Result<Self, BpannError> {
        let slot_words = SLOT_ENTRIES * (1 + num_dim.max(1));
        Ok(Self {
            num_dim: num_dim.max(1),
            slot_words,
            num_pages: 0,
            capacity: GROW_MIN_SLOTS,
            dir: dir.to_path_buf(),
            files: None,
            meta: map(None, GROW_MIN_SLOTS * META_WORDS * 4)?,
            slots: map(None, GROW_MIN_SLOTS * slot_words * 4)?,
            residency: Residency::new(TREE_RESIDENT_BUDGET_BYTES),
        })
    }

    pub fn num_dim(&self) -> usize {
        self.num_dim
    }

    pub fn num_pages(&self) -> usize {
        self.num_pages
    }

    /// Bytes of both regions (the scratch files' disk footprint once written).
    pub fn file_bytes(&self) -> usize {
        self.meta.len() + self.slots.len()
    }

    pub fn set_resident_budget(&mut self, bytes: usize) -> Result<(), BpannError> {
        self.residency.set_budget(bytes);
        self.update_tracking()
    }

    fn update_tracking(&mut self) -> Result<(), BpannError> {
        if self.files.is_none() && self.file_bytes() > self.residency.budget() {
            self.spill()?;
        }
        self.residency.update_tracking(self.file_bytes(), self.meta.len());
        Ok(())
    }

    /// Move both regions from anonymous memory to scratch files.
    fn spill(&mut self) -> Result<(), BpannError> {
        let files = (scratch_file(&self.dir, "tree_meta")?, scratch_file(&self.dir, "tree_slots")?);
        let mut meta = map(Some(&files.0), self.meta.len())?;
        let mut slots = map(Some(&files.1), self.slots.len())?;
        Residency::copy_released(&self.meta, &mut meta, self.meta.len());
        Residency::copy_released(&self.slots, &mut slots, self.slots.len());
        (self.meta, self.slots, self.files) = (meta, slots, Some(files));
        Ok(())
    }

    /// Drop every resident page of both files from the process; contents are kept.
    /// Anonymous memory (a store within its budget) is kept.
    pub fn release_resident(&self) {
        if self.files.is_some() {
            self.residency.release(&[&self.meta, &self.slots]);
        }
    }

    /// A new empty page of `kind` (no entries, count 0, no parent, infinite radius).
    pub fn alloc(&mut self, kind: Kind) -> Result<u32, BpannError> {
        if self.num_pages == self.capacity {
            self.grow(self.capacity * 2)?;
        }
        let id = self.num_pages as u32;
        self.num_pages += 1;
        self.meta_words_mut(id).copy_from_slice(&[kind_tag(kind), 0, f32::INFINITY.to_bits(), NO_PARENT]);
        Ok(id)
    }

    /// Change the kind of page `id`, keeping its entries.
    pub fn set_kind(&mut self, id: u32, kind: Kind) {
        let m = self.meta_words_mut(id);
        m[0] = kind_tag(kind) | (m[0] & LEN_MASK);
    }

    fn grow(&mut self, capacity: usize) -> Result<(), BpannError> {
        let (meta_bytes, slot_bytes) = (capacity * META_WORDS * 4, capacity * self.slot_words * 4);
        if self.files.is_none() && meta_bytes + slot_bytes > self.residency.budget() {
            self.spill()?;
        }
        match &self.files {
            Some((meta_file, slot_file)) => {
                self.meta = map(Some(meta_file), meta_bytes)?;
                self.slots = map(Some(slot_file), slot_bytes)?;
            }
            None => {
                grow_anon(&mut self.meta, meta_bytes)?;
                grow_anon(&mut self.slots, slot_bytes)?;
            }
        }
        self.capacity = capacity;
        self.update_tracking()
    }

    #[inline]
    fn meta_words(&self, id: u32) -> &[u32] {
        let i = id as usize;
        assert!(i < self.num_pages, "tree page {id} out of range");
        unsafe { std::slice::from_raw_parts((self.meta.as_ptr() as *const u32).add(i * META_WORDS), META_WORDS) }
    }

    #[inline]
    fn meta_words_mut(&mut self, id: u32) -> &mut [u32] {
        let i = id as usize;
        assert!(i < self.num_pages, "tree page {id} out of range");
        unsafe { std::slice::from_raw_parts_mut((self.meta.as_mut_ptr() as *mut u32).add(i * META_WORDS), META_WORDS) }
    }

    #[inline]
    fn slot_words(&self, id: u32) -> &[u32] {
        let i = id as usize;
        assert!(i < self.num_pages, "tree page {id} out of range");
        self.residency.touch(i * self.slot_words * 4, self.slot_words * 4, &[&self.meta, &self.slots]);
        let n = self.slot_words;
        unsafe { std::slice::from_raw_parts((self.slots.as_ptr() as *const u32).add(i * n), n) }
    }

    #[inline]
    fn slot_words_mut(&mut self, id: u32) -> &mut [u32] {
        let i = id as usize;
        assert!(i < self.num_pages, "tree page {id} out of range");
        self.residency.touch(i * self.slot_words * 4, self.slot_words * 4, &[&self.meta, &self.slots]);
        let n = self.slot_words;
        unsafe { std::slice::from_raw_parts_mut((self.slots.as_mut_ptr() as *mut u32).add(i * n), n) }
    }

    pub fn is_leaf(&self, id: u32) -> bool {
        self.meta_words(id)[0] & !LEN_MASK == kind_tag(Kind::Leaf)
    }

    /// Entries of page `id`: rows of a leaf, children of an internal page.
    pub fn len(&self, id: u32) -> usize {
        (self.meta_words(id)[0] & LEN_MASK) as usize
    }

    /// Row ids of a leaf or child page ids of an internal page.
    pub fn ids(&self, id: u32) -> &[u32] {
        let n = self.len(id);
        &self.slot_words(id)[..n]
    }

    /// The page's vectors, `len * num_dim` values in `ids` order.
    pub fn block(&self, id: u32) -> &[f32] {
        let n = self.len(id) * self.num_dim;
        let words = &self.slot_words(id)[SLOT_ENTRIES..SLOT_ENTRIES + n];
        unsafe { std::slice::from_raw_parts(words.as_ptr() as *const f32, n) }
    }

    /// Mutable vectors of page `id` (its current entries only).
    pub fn block_mut(&mut self, id: u32) -> &mut [f32] {
        let n = self.len(id) * self.num_dim;
        let words = &mut self.slot_words_mut(id)[SLOT_ENTRIES..SLOT_ENTRIES + n];
        unsafe { std::slice::from_raw_parts_mut(words.as_mut_ptr() as *mut f32, n) }
    }

    /// Vector `slot` of page `id`.
    pub fn entry_mut(&mut self, id: u32, slot: usize) -> &mut [f32] {
        let d = self.num_dim;
        &mut self.block_mut(id)[slot * d..(slot + 1) * d]
    }

    /// Replace the entries of page `id` (at most [`SLOT_ENTRIES`]).
    pub fn set_entries(&mut self, id: u32, ids: &[u32], block: &[f32]) {
        let n = ids.len();
        assert!(n <= SLOT_ENTRIES && block.len() == n * self.num_dim, "bad entries for page {id}");
        let words = self.slot_words_mut(id);
        words[..n].copy_from_slice(ids);
        for (w, &v) in words[SLOT_ENTRIES..].iter_mut().zip(block) {
            *w = v.to_bits();
        }
        let m = self.meta_words_mut(id);
        m[0] = (m[0] & !LEN_MASK) | n as u32;
    }

    /// Append one entry (`entry_id`, vector `v`) to page `id`.
    pub fn push_entry(&mut self, id: u32, entry_id: u32, v: &[f32]) {
        let (n, d) = (self.len(id), self.num_dim);
        assert!(n < SLOT_ENTRIES && v.len() == d, "page {id} is full");
        let words = self.slot_words_mut(id);
        words[n] = entry_id;
        for (w, &x) in words[SLOT_ENTRIES + n * d..].iter_mut().zip(v) {
            *w = x.to_bits();
        }
        self.meta_words_mut(id)[0] += 1;
    }

    pub fn count(&self, id: u32) -> usize {
        self.meta_words(id)[1] as usize
    }

    pub fn set_count(&mut self, id: u32, n: usize) {
        self.meta_words_mut(id)[1] = n as u32;
    }

    /// Upper bound on the distance from the page's centroid to any row below it.
    pub fn radius(&self, id: u32) -> f32 {
        f32::from_bits(self.meta_words(id)[2])
    }

    pub fn set_radius(&mut self, id: u32, r: f32) {
        self.meta_words_mut(id)[2] = r.to_bits();
    }

    pub fn parent(&self, id: u32) -> Option<u32> {
        Some(self.meta_words(id)[3]).filter(|&p| p != NO_PARENT)
    }

    pub fn set_parent(&mut self, id: u32, parent: Option<u32>) {
        self.meta_words_mut(id)[3] = parent.unwrap_or(NO_PARENT);
    }

    /// An independent copy; a file-backed store is copied into new scratch files a
    /// chunk at a time so the copy never holds more than one chunk resident.
    pub fn try_clone(&self) -> Result<Self, BpannError> {
        let mut out = Self::create(&self.dir, self.num_dim)?;
        out.residency.set_budget(self.residency.budget());
        out.grow(self.capacity)?;
        out.num_pages = self.num_pages;
        let (meta_len, slot_len) = (self.num_pages * META_WORDS * 4, self.num_pages * self.slot_words * 4);
        if self.files.is_none() {
            out.meta[..meta_len].copy_from_slice(&self.meta[..meta_len]);
            out.slots[..slot_len].copy_from_slice(&self.slots[..slot_len]);
            return Ok(out);
        }
        if out.files.is_none() {
            out.spill()?;
        }
        Residency::copy_released(&self.meta, &mut out.meta, meta_len);
        Residency::copy_released(&self.slots, &mut out.slots, slot_len);
        Ok(out)
    }
}

#[cfg(test)]
#[path = "tree_store_tests.rs"]
mod tests;
