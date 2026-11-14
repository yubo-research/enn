use crate::error::BpannError;
use crate::index::tree_store::{Kind, TreeStore, LEN_MASK, META_HDR_WORDS, NO_PARENT};

pub const SIB_BLOCK: usize = 16;

fn sib_blocks_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ENN_SIB_BLOCKS").map_or(true, |v| v != "0"))
}

impl TreeStore {
    pub(super) fn alloc_tagged(&mut self, tag: u32) -> Result<u32, BpannError> {
        if self.num_pages == self.capacity {
            self.grow(self.capacity * 2)?;
        }
        let id = self.num_pages as u32;
        self.num_pages += 1;
        self.meta_words_mut(id)[..META_HDR_WORDS].copy_from_slice(&[
            tag,
            0,
            f32::INFINITY.to_bits(),
            NO_PARENT,
        ]);
        if tag != 0 {
            self.set_codes_valid(id, true);
        }
        Ok(id)
    }

    pub fn is_free(&self, id: u32) -> bool {
        self.meta_words(id)[0] & !LEN_MASK == 0
    }

    pub fn mark_free(&mut self, id: u32) {
        self.meta_words_mut(id)[0] = 0;
    }

    pub fn alloc_near(&mut self, kind: Kind, near: &[u32]) -> Result<u32, BpannError> {
        if !sib_blocks_enabled() {
            return self.alloc(kind);
        }
        let mut tried: Vec<usize> = Vec::new();
        for &sib in near {
            let b = sib as usize / SIB_BLOCK;
            if tried.contains(&b) {
                continue;
            }
            tried.push(b);
            let end = ((b + 1) * SIB_BLOCK).min(self.num_pages);
            if let Some(id) = (b * SIB_BLOCK..end).find(|&i| self.is_free(i as u32)) {
                self.set_kind(id as u32, kind);
                return Ok(id as u32);
            }
            if end < (b + 1) * SIB_BLOCK {
                return self.alloc(kind);
            }
        }
        while !self.num_pages.is_multiple_of(SIB_BLOCK) {
            self.alloc_tagged(0)?;
        }
        let base = self.alloc(kind)?;
        for _ in 1..SIB_BLOCK {
            self.alloc_tagged(0)?;
        }
        Ok(base)
    }
}
