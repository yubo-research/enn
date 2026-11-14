use crate::error::BpannError;
use crate::index::tree_store::{Kind, TreeStore, LEN_MASK, META_HDR_WORDS, NO_PARENT};

pub const SIB_BLOCK: usize = 16;

fn sib_blocks_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ENN_SIB_BLOCKS").map_or(true, |v| v != "0"))
}

fn sib_split_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ENN_SIB_SPLIT").is_ok_and(|v| v == "1"))
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
        self.set_codes_valid(id, true);
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

    fn alloc_block(&mut self) -> Result<u32, BpannError> {
        while !self.num_pages.is_multiple_of(SIB_BLOCK) {
            self.alloc_tagged(0)?;
        }
        let base = self.num_pages as u32;
        for _ in 0..SIB_BLOCK {
            self.alloc_tagged(0)?;
        }
        Ok(base)
    }

    fn move_node(&mut self, old: u32, new: u32) {
        let leaf = self.is_leaf(old);
        let ids = self.ids(old).to_vec();
        let block = self.block(old).to_vec();
        let (count, radius, parent) = (self.count(old), self.radius(old), self.parent(old));
        self.set_kind(new, if leaf { Kind::Leaf } else { Kind::Internal });
        self.set_entries(new, &ids, &block);
        self.set_count(new, count);
        self.set_radius(new, radius);
        self.set_parent(new, parent);
        if !leaf {
            ids.iter().for_each(|&c| self.set_parent(c, Some(new)));
        }
        self.icache.invalidate(old);
        self.icache.invalidate(new);
        self.mark_free(old);
    }

    pub(super) fn relocate_block(&mut self, kids: &[u32], parent: u32) -> Result<Vec<u32>, BpannError> {
        let base = self.alloc_block()?;
        let mut out = Vec::with_capacity(kids.len());
        for (i, &kid) in kids.iter().enumerate() {
            let target = base + i as u32;
            self.move_node(kid, target);
            self.set_parent(target, Some(parent));
            out.push(target);
        }
        Ok(out)
    }

    pub(super) fn sib_split_fits(n: usize) -> bool {
        sib_split_enabled() && n <= SIB_BLOCK
    }
}
