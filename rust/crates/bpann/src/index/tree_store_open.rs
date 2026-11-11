use std::fs::{self, OpenOptions};
use std::path::Path;

use memmap2::MmapMut;

use crate::error::BpannError;
use crate::index::tree_residency::{file_budget, Residency, TREE_FILE_BUDGET_PER_MILLE};
use crate::index::tree_store::{
    TreeStore, GROW_MIN_SLOTS, META_WORDS, SLOT_ENTRIES, TREE_RESIDENT_BUDGET_BYTES,
};
use crate::index::tree_store::{META_FILE, SLOTS_FILE, STAMP_FILE};
use crate::index::tree_store_files::{io_err, map, named_file, same_file, scratch_file, sync_dir};

impl TreeStore {
    pub fn create(dir: &Path, num_dim: usize) -> Result<Self, BpannError> {
        Self::create_with(dir, num_dim, false)
    }

    pub fn create_named(dir: &Path, num_dim: usize) -> Result<Self, BpannError> {
        let _ = fs::remove_file(dir.join(STAMP_FILE));
        Self::create_with(dir, num_dim, true)
    }

    fn create_with(dir: &Path, num_dim: usize, named: bool) -> Result<Self, BpannError> {
        let slot_words = SLOT_ENTRIES * (1 + num_dim.max(1));
        let files = if named {
            (named_file(dir, META_FILE)?, named_file(dir, SLOTS_FILE)?)
        } else {
            (
                scratch_file(dir, "tree_meta")?,
                scratch_file(dir, "tree_slots")?,
            )
        };
        let meta = map(&files.0, GROW_MIN_SLOTS * META_WORDS * 4)?;
        let slots = map(&files.1, GROW_MIN_SLOTS * slot_words * 4)?;
        Ok(Self {
            named,
            stamped: false,
            num_dim: num_dim.max(1),
            slot_words,
            num_pages: 0,
            capacity: GROW_MIN_SLOTS,
            dir: dir.to_path_buf(),
            files,
            meta,
            slots,
            residency: Residency::new(file_budget(
                TREE_RESIDENT_BUDGET_BYTES,
                TREE_FILE_BUDGET_PER_MILLE,
            )),
        })
    }

    pub fn open_named(dir: &Path, num_dim: usize, num_pages: usize) -> Option<Self> {
        let slot_words = SLOT_ENTRIES * (1 + num_dim.max(1));
        let open = |name: &str| {
            OpenOptions::new()
                .read(true)
                .write(true)
                .open(dir.join(name))
                .ok()
        };
        let files = (open(META_FILE)?, open(SLOTS_FILE)?);
        let capacity = files.0.metadata().ok()?.len() as usize / (META_WORDS * 4);
        let slot_bytes = files.1.metadata().ok()?.len() as usize;
        if capacity < num_pages.max(1) || slot_bytes != capacity * slot_words * 4 {
            return None;
        }
        let meta = unsafe { MmapMut::map_mut(&files.0) }.ok()?;
        let slots = unsafe { MmapMut::map_mut(&files.1) }.ok()?;
        let mut store = Self {
            named: true,
            stamped: true,
            num_dim: num_dim.max(1),
            slot_words,
            num_pages,
            capacity,
            dir: dir.to_path_buf(),
            files,
            meta,
            slots,
            residency: Residency::new(file_budget(
                TREE_RESIDENT_BUDGET_BYTES,
                TREE_FILE_BUDGET_PER_MILLE,
            )),
        };
        store.update_tracking();
        Some(store)
    }

    pub fn is_named_in(&self, dir: &Path) -> bool {
        self.named
            && self.dir == dir
            && same_file(&self.files.0, &dir.join(META_FILE))
            && same_file(&self.files.1, &dir.join(SLOTS_FILE))
    }

    pub fn file_lens(&self) -> (usize, usize) {
        (self.meta.len(), self.slots.len())
    }

    pub fn flush(&self) -> Result<(), BpannError> {
        self.files.0.sync_all().map_err(io_err)?;
        self.files.1.sync_all().map_err(io_err)
    }

    pub fn mark_stamped(&mut self) {
        self.stamped = true;
    }

    pub(super) fn unstamp(&mut self) {
        if self.stamped {
            self.stamped = false;
            let _ = fs::remove_file(self.dir.join(STAMP_FILE));
            sync_dir(&self.dir);
        }
    }
}
