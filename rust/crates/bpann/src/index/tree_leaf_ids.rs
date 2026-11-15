use std::sync::atomic::{AtomicU8, Ordering};

use crate::index::tree_codes::{code_page_bytes, code_row_bytes, CODE_PAGE_HEADER};
use crate::index::tree_store::{TreeStore, SLOT_ENTRIES};

const MIRROR: u32 = 0x1D5_0001;
const MAGIC_AT: usize = 4;
const TAIL_BYTES: usize = SLOT_ENTRIES * 4;

static LEAF_IDS: AtomicU8 = AtomicU8::new(2);

pub fn leaf_ids_enabled() -> bool {
    match LEAF_IDS.load(Ordering::Relaxed) {
        2 => {
            let on = std::env::var("ENN_LEAF_IDS").map_or(true, |v| v != "0");
            LEAF_IDS.store(u8::from(on), Ordering::Relaxed);
            on
        }
        v => v != 0,
    }
}

pub fn set_leaf_ids(value: bool) {
    LEAF_IDS.store(u8::from(value), Ordering::Relaxed);
}

fn tail_offset(num_dim: usize) -> usize {
    CODE_PAGE_HEADER + SLOT_ENTRIES * code_row_bytes(num_dim)
}

const ONE_PAGE: usize = 4096;

pub fn tail_fits(num_dim: usize) -> bool {
    code_page_bytes(num_dim) == ONE_PAGE && tail_offset(num_dim) + TAIL_BYTES <= ONE_PAGE
}

const HINT_KNOWN: u16 = 1 << 15;
const HINT_LEAF: u16 = 1 << 14;
const DEPTH_MASK: u16 = HINT_LEAF - 1;

#[derive(Clone, Copy)]
pub(super) struct LeafTest {
    on: bool,
}

impl LeafTest {
    pub(super) fn depth(raw: u16) -> u16 {
        raw & DEPTH_MASK
    }

    pub(super) fn new(s: &TreeStore, coded: bool) -> Self {
        Self {
            on: coded && leaf_ids_enabled() && tail_fits(s.num_dim),
        }
    }

    pub(super) fn children(self, s: &TreeStore, id: u32) -> Option<u32> {
        if self.on {
            s.leaf_children(id)
        } else {
            None
        }
    }

    pub(super) fn pack(depth: u16, mask: Option<u32>, slot: usize) -> u16 {
        match mask {
            Some(m) if m >> slot & 1 == 1 => depth | HINT_KNOWN | HINT_LEAF,
            Some(_) => depth | HINT_KNOWN,
            None => depth,
        }
    }

    pub(super) fn is_leaf(self, s: &TreeStore, id: u32, raw: u16) -> bool {
        if raw & HINT_KNOWN == 0 {
            return s.is_leaf(id);
        }
        let leaf = raw & HINT_LEAF != 0;
        debug_assert_eq!(leaf, s.is_leaf(id), "stale leaf bit for page {id}");
        leaf
    }
}

impl TreeStore {
    fn mirror_word(&self, id: u32, off: usize) -> u32 {
        let at = self.code_page_range(id).start + off;
        let b = &self.codes[at..at + 4];
        u32::from_le_bytes([b[0], b[1], b[2], b[3]])
    }

    fn set_mirror_word(&mut self, id: u32, off: usize, w: u32) {
        let at = self.code_page_range(id).start + off;
        self.codes[at..at + 4].copy_from_slice(&w.to_le_bytes());
    }

    pub(super) fn mirror_ids(&mut self, id: u32, ids: &[u32]) {
        if !tail_fits(self.num_dim) {
            return;
        }
        let t = self.code_page_range(id).start + tail_offset(self.num_dim);
        let tail = &mut self.codes[t..t + TAIL_BYTES];
        tail.fill(0);
        for (o, &r) in tail.chunks_exact_mut(4).zip(ids) {
            o.copy_from_slice(&(r + 1).to_le_bytes());
        }
        self.set_mirror_word(id, MAGIC_AT, MIRROR);
    }

    pub(super) fn mirror_clear(&mut self, id: u32) {
        if tail_fits(self.num_dim) {
            self.set_mirror_word(id, MAGIC_AT, 0);
        }
    }

    pub(super) fn mirror_push(&mut self, id: u32, slot: usize, entry: u32) {
        if tail_fits(self.num_dim) && self.mirror_word(id, MAGIC_AT) == MIRROR {
            let off = tail_offset(self.num_dim) + 4 * slot;
            self.set_mirror_word(id, off, entry + 1);
        }
    }

    pub fn leaf_id_tail(&self, id: u32) -> Option<&[u8]> {
        if !leaf_ids_enabled() || !tail_fits(self.num_dim) {
            return None;
        }
        if self.mirror_word(id, MAGIC_AT) != MIRROR {
            return None;
        }
        let t = self.code_page_range(id).start + tail_offset(self.num_dim);
        Some(&self.codes[t..t + TAIL_BYTES])
    }

    pub fn leaf_ids_coded(&self, id: u32, out: &mut Vec<u32>) -> bool {
        let Some(tail) = self.leaf_id_tail(id) else {
            return false;
        };
        out.clear();
        out.extend(
            tail.chunks_exact(4)
                .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .take_while(|&w| w != 0)
                .map(|w| w - 1),
        );
        true
    }

    pub(super) fn coded_extent(&self, id: u32, used: usize) -> usize {
        if leaf_ids_enabled() && tail_fits(self.num_dim) && self.is_leaf(id) {
            tail_offset(self.num_dim) + TAIL_BYTES
        } else {
            used
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::tree::Tree;
    use crate::index::tree_store::Kind;

    fn assert_mirrors(t: &Tree) -> (u16, u16) {
        let s = &t.store;
        let mut out = Vec::new();
        let (mut lo, mut hi) = (u16::MAX, 0u16);
        let mut stack = vec![(t.header.root_page_id, 0u16)];
        while let Some((p, d)) = stack.pop() {
            if s.is_leaf(p) {
                assert!(s.leaf_ids_coded(p, &mut out));
                assert_eq!(out, s.ids(p));
                (lo, hi) = (lo.min(d), hi.max(d));
                continue;
            }
            let mask = s.leaf_children(p).expect("kind word");
            for (i, &c) in s.ids(p).iter().enumerate() {
                assert_eq!(mask >> i & 1 == 1, s.is_leaf(c), "page {p} slot {i}");
                stack.push((c, d + 1));
            }
        }
        (lo, hi)
    }

    #[test]
    fn bulk_and_incremental_trees_keep_leaf_bits_and_ids_exact() {
        let (n0, d) = (3000usize, 4usize);
        let dir = tempfile::tempdir().unwrap();
        let v = |r: usize| -> Vec<f32> {
            (0..d)
                .map(|j| ((r * 7919 + j * 104_729) % 1000) as f32 / 1000.0 + (r % 7) as f32)
                .collect()
        };
        let rows: Vec<u32> = (0..n0 as u32).collect();
        let mut vs: Vec<f32> = (0..n0).flat_map(v).collect();
        let mut t =
            crate::index::tree_bulk::bulk_build(&rows, &mut vs, d, dir.path().join("i")).unwrap();
        assert_mirrors(&t);
        for r in n0..3 * n0 {
            t.insert_row(r as u32, &v(r)).unwrap();
        }
        let (lo, hi) = assert_mirrors(&t);
        assert!(lo <= hi);
    }

    #[test]
    fn leaf_ids_follow_meta_through_push_set_and_reuse() {
        for d in [3usize, 10, 100] {
            let dir = tempfile::tempdir().unwrap();
            let mut s = TreeStore::create(dir.path(), d).unwrap();
            let leaf = s.alloc(Kind::Leaf).unwrap();
            let v = vec![0.5f32; d];
            for r in [7u32, 0, 42] {
                s.push_entry(leaf, r, &v);
            }
            let mut out = Vec::new();
            assert!(s.leaf_ids_coded(leaf, &mut out) == tail_fits(d));
            if tail_fits(d) {
                assert_eq!(out, s.ids(leaf));
                s.set_ids_only(leaf, &[9]);
                assert!(s.leaf_ids_coded(leaf, &mut out));
                assert_eq!(out, vec![9]);
                s.mark_free(leaf);
                assert!(!s.leaf_ids_coded(leaf, &mut out));
                s.set_kind(leaf, Kind::Leaf);
                assert!(s.leaf_ids_coded(leaf, &mut out));
                assert!(out.is_empty());
            }
        }
    }
}
