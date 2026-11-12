use crate::index::tree_store::{TreeStore, META_HDR_WORDS, META_WORDS};

pub(super) const ROUTE_SLOTS: usize = 32;
const ROUTE_WORD: usize = META_WORDS - ROUTE_SLOTS;
const _: () = assert!(META_HDR_WORDS + ROUTE_SLOTS <= ROUTE_WORD);

impl TreeStore {
    pub fn child_radii(&self, id: u32) -> &[f32] {
        let n = self.len(id);
        assert!(n <= ROUTE_SLOTS, "internal page {id} has {n} children");
        let w = &self.meta_words(id)[ROUTE_WORD..ROUTE_WORD + n];
        unsafe { std::slice::from_raw_parts(w.as_ptr() as *const f32, n) }
    }

    pub(super) fn refresh_routes(&mut self, id: u32) {
        if self.is_leaf(id) {
            return;
        }
        let radii: Vec<u32> = self
            .ids(id)
            .iter()
            .map(|&c| self.radius(c).to_bits())
            .collect();
        assert!(radii.len() <= ROUTE_SLOTS, "internal page {id} too wide");
        self.meta_words_mut(id)[ROUTE_WORD..ROUTE_WORD + radii.len()].copy_from_slice(&radii);
    }

    pub(super) fn set_route(&mut self, parent: u32, slot: usize, r: f32) {
        assert!(slot < ROUTE_SLOTS, "internal page {parent} too wide");
        self.meta_words_mut(parent)[ROUTE_WORD + slot] = r.to_bits();
    }

    pub(super) fn route_child_changed(&mut self, child: u32, r: f32) {
        let Some(p) = self.parent(child) else {
            return;
        };
        if self.is_leaf(p) {
            return;
        }
        if let Some(slot) = self.ids(p).iter().position(|&c| c == child) {
            self.set_route(p, slot, r);
        }
    }

    pub fn rebuild_routes(&mut self) {
        for id in 0..self.num_pages as u32 {
            self.refresh_routes(id);
        }
    }
}
