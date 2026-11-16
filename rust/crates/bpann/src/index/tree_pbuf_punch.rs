use crate::index::tree_pbuf::{PBuf, PAGE};
use crate::index::tree_store::TreeStore;

impl PBuf {
    pub(super) fn punch(&self, id: u32, n: usize) {
        if let (true, Some(m)) = (n > 0, self.map.as_ref()) {
            let start = id as usize * self.page_len();
            let len = (n * self.rec).div_ceil(PAGE) * PAGE;
            let _ =
                unsafe { m.unchecked_advise_range(memmap2::UncheckedAdvice::Remove, start, len) };
        }
    }
}

impl TreeStore {
    pub(super) fn pbuf_tail_cold(&self, id: u32) -> Option<bool> {
        let n = self.pbuf_len(id);
        let (rec, m) = (self.pbuf.rec, self.pbuf.map.as_ref()?);
        if n == 0 || self.is_leaf(id) || (n * rec).is_multiple_of(PAGE) {
            return None;
        }
        let at = id as usize * self.pbuf.page_len() + (n * rec) / PAGE * PAGE;
        let mut v = [0u8; 1];
        let rc = unsafe {
            libc::mincore(
                m.as_ptr().add(at) as *mut libc::c_void,
                PAGE,
                v.as_mut_ptr(),
            )
        };
        (rc == 0).then_some(v[0] & 1 == 0)
    }

    #[cfg(test)]
    pub(super) fn pbuf_region_bytes(&self, id: u32, n: usize) -> &[u8] {
        self.pbuf.region(id, n)
    }
}
