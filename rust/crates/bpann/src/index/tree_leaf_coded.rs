use crate::index::tree_codes::CODE_PAGE_HEADER;
use crate::index::tree_store::{TreeStore, SLOT_ENTRIES};

impl TreeStore {
    pub fn leaf_code_bytes(&self, id: u32) -> Vec<u8> {
        self.code_rows(id, self.len(id)).to_vec()
    }

    pub fn set_leaf_coded(&mut self, id: u32, ids: &[u32], rows: &[u8]) {
        let rb = self.code_rb();
        assert!(
            self.is_leaf(id) && ids.len() <= SLOT_ENTRIES && rows.len() == ids.len() * rb,
            "bad coded entries for leaf {id}"
        );
        let start = self.code_page_range(id).start + CODE_PAGE_HEADER;
        self.codes[start..start + rows.len()].copy_from_slice(rows);
        self.set_codes_valid(id, true);
        self.set_ids_only(id, ids);
    }
}
