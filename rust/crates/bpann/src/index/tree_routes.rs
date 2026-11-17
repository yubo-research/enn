use crate::index::tree_store::{TreeStore, META_HDR_WORDS, META_WORDS};

pub(super) const ROUTE_SLOTS: usize = 32;
const ROUTE_WORD: usize = META_WORDS - ROUTE_SLOTS;
const KIND_WORD: usize = ROUTE_WORD - 1;
const KIND_MAGIC: u32 = 0xFEED_1EAF;
const LEAF_BIT: u32 = 1 << 31;
const _: () = assert!(META_HDR_WORDS + ROUTE_SLOTS <= KIND_WORD);

impl TreeStore {
    fn route_words(&self, id: u32) -> &[u32] {
        let n = self.len(id);
        assert!(n <= ROUTE_SLOTS, "internal page {id} has {n} children");
        &self.meta_words(id)[ROUTE_WORD..ROUTE_WORD + n]
    }

    pub fn child_radii(&self, id: u32) -> Vec<f32> {
        self.radii_iter(id).collect()
    }

    pub fn radii_iter(&self, id: u32) -> impl Iterator<Item = f32> + '_ {
        self.route_words(id)
            .iter()
            .map(|&w| f32::from_bits(w & !LEAF_BIT))
    }

    pub fn leaf_children(&self, id: u32) -> Option<u32> {
        if self.is_leaf(id) || self.meta_words(id)[KIND_WORD] != KIND_MAGIC {
            return None;
        }
        Some(
            self.route_words(id)
                .iter()
                .enumerate()
                .fold(0u32, |m, (i, &w)| m | (w >> 31) << i),
        )
    }

    fn route_bits(&self, child: u32, r: f32) -> u32 {
        let leaf = if self.is_leaf(child) { LEAF_BIT } else { 0 };
        r.to_bits() & !LEAF_BIT | leaf
    }

    pub(super) fn refresh_routes(&mut self, id: u32) {
        if self.is_leaf(id) {
            return;
        }
        let radii: Vec<u32> = self
            .ids(id)
            .iter()
            .map(|&c| self.route_bits(c, self.radius(c)))
            .collect();
        assert!(radii.len() <= ROUTE_SLOTS, "internal page {id} too wide");
        let m = self.meta_words_mut(id);
        m[ROUTE_WORD..ROUTE_WORD + radii.len()].copy_from_slice(&radii);
        m[KIND_WORD] = KIND_MAGIC;
        if self.pbuf_tagged() && self.pbuf_len(id) > 0 {
            self.pbuf_retag(id);
        }
    }

    pub(super) fn route_radius(&self, parent: u32, slot: usize) -> f32 {
        f32::from_bits(self.meta_words(parent)[ROUTE_WORD + slot] & !LEAF_BIT)
    }

    pub(super) fn widen_route(&mut self, parent: u32, slot: usize, r: f32) {
        if slot >= ROUTE_SLOTS || r <= self.route_radius(parent, slot) {
            return;
        }
        self.icache.invalidate(parent);
        let w = &mut self.meta_words_mut(parent)[ROUTE_WORD + slot];
        *w = (*w & LEAF_BIT) | (r.to_bits() & !LEAF_BIT);
    }

    pub(super) fn route_child_is_leaf(&self, parent: u32, slot: usize) -> bool {
        self.meta_words(parent)[ROUTE_WORD + slot] & LEAF_BIT != 0
    }

    pub(super) fn set_route(&mut self, parent: u32, slot: usize, child: u32, r: f32) {
        assert!(slot < ROUTE_SLOTS, "internal page {parent} too wide");
        let w = self.route_bits(child, r);
        self.meta_words_mut(parent)[ROUTE_WORD + slot] = w;
    }

    pub(super) fn route_child_changed(&mut self, child: u32, r: f32) {
        let Some(p) = self.parent(child) else {
            return;
        };
        if self.is_leaf(p) {
            return;
        }
        if let Some(slot) = self.ids(p).iter().position(|&c| c == child) {
            let r = if self.pbuf_tagged() && self.pbuf_len(p) > 0 {
                r.max(self.route_radius(p, slot))
            } else {
                r
            };
            self.set_route(p, slot, child, r);
        }
    }

    pub fn rebuild_routes(&mut self) {
        for id in 0..self.num_pages as u32 {
            self.refresh_routes(id);
        }
    }
}
