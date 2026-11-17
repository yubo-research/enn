use std::fs::File;

use memmap2::MmapMut;

use crate::error::BpannError;
use crate::index::tree_codes::{code_row_bytes, encode_row};
use crate::index::tree_store::TreeStore;
use crate::index::tree_store_files::{map, named_file, scratch_file};

pub const LGRAPH_FILE: &str = "tree_lgraph.bin";
pub const LGRAPH_SLOTS: usize = 32;
const PAGE: usize = 4096;
const HDR: usize = 8;

fn entry_bytes(num_dim: usize) -> usize {
    8 + code_row_bytes(num_dim)
}

fn record_bytes(num_dim: usize) -> usize {
    let need = HDR + LGRAPH_SLOTS * entry_bytes(num_dim);
    if need <= PAGE {
        need.next_power_of_two()
    } else {
        need.div_ceil(PAGE) * PAGE
    }
}

pub struct LeafGraph {
    rec: usize,
    eb: usize,
    file: Option<File>,
    map: Option<MmapMut>,
}

impl LeafGraph {
    pub(super) fn new(num_dim: usize) -> Self {
        Self {
            rec: record_bytes(num_dim),
            eb: entry_bytes(num_dim),
            file: None,
            map: None,
        }
    }

    pub(super) fn open(dir: &std::path::Path, num_dim: usize, capacity: usize) -> Self {
        let mut g = Self::new(num_dim);
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(dir.join(LGRAPH_FILE));
        if let Ok(f) = file {
            let ok = f
                .metadata()
                .is_ok_and(|m| m.len() as usize == capacity * g.rec);
            if ok {
                g.map = unsafe { MmapMut::map_mut(&f) }.ok();
                g.file = Some(f);
            }
        }
        g
    }

    pub(super) fn grow(&mut self, capacity: usize) -> Result<(), BpannError> {
        if let Some(f) = &self.file {
            self.map = Some(map(f, capacity * self.rec)?);
        }
        Ok(())
    }

    pub(super) fn clone_into(
        &self,
        dir: &std::path::Path,
        capacity: usize,
        num_pages: usize,
    ) -> Result<Self, BpannError> {
        let mut g = Self {
            rec: self.rec,
            eb: self.eb,
            file: None,
            map: None,
        };
        if let Some(src) = &self.map {
            g.file = Some(scratch_file(dir, "tree_lgraph")?);
            g.grow(capacity)?;
            let n = (num_pages * self.rec).min(src.len());
            if let Some(dst) = g.map.as_mut() {
                dst[..n].copy_from_slice(&src[..n]);
            }
        }
        Ok(g)
    }

    fn record(&self, id: u32) -> Option<&[u8]> {
        let m = self.map.as_ref()?;
        let at = id as usize * self.rec;
        m.get(at..at + self.rec)
    }

    fn record_mut(&mut self, id: u32) -> Option<&mut [u8]> {
        let rec = self.rec;
        let m = self.map.as_mut()?;
        let at = id as usize * rec;
        m.get_mut(at..at + rec)
    }
}

fn word(b: &[u8], at: usize) -> u32 {
    u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
}

fn count(r: &[u8]) -> usize {
    (word(r, 0) as usize).min(LGRAPH_SLOTS)
}

fn put_entry(r: &mut [u8], eb: usize, i: usize, (id, radius, cent): (u32, f32, &[f32])) {
    let at = HDR + i * eb;
    r[at..at + 4].copy_from_slice(&id.to_le_bytes());
    r[at + 4..at + 8].copy_from_slice(&radius.to_le_bytes());
    encode_row(cent, &mut r[at + 8..at + eb]);
}

pub(super) struct Entries<'a> {
    rec: &'a [u8],
    eb: usize,
    i: usize,
    n: usize,
}

impl<'a> Iterator for Entries<'a> {
    type Item = (u32, f32, &'a [u8]);
    fn next(&mut self) -> Option<Self::Item> {
        if self.i >= self.n {
            return None;
        }
        let at = HDR + self.i * self.eb;
        self.i += 1;
        let r = self.rec;
        Some((
            word(r, at),
            f32::from_bits(word(r, at + 4)),
            &r[at + 8..at + self.eb],
        ))
    }
}

impl TreeStore {
    fn lgraph_ensure(&mut self) -> Result<(), BpannError> {
        if self.lgraph.file.is_none() {
            self.lgraph.file = Some(if self.named {
                named_file(&self.dir, LGRAPH_FILE)?
            } else {
                scratch_file(&self.dir, "tree_lgraph")?
            });
            self.lgraph.grow(self.capacity)?;
        }
        Ok(())
    }

    pub(super) fn lgraph_entries(&self, id: u32) -> Entries<'_> {
        let eb = self.lgraph.eb;
        match self.lgraph.record(id) {
            Some(rec) => Entries {
                rec,
                eb,
                i: 0,
                n: count(rec),
            },
            None => Entries {
                rec: &[],
                eb,
                i: 0,
                n: 0,
            },
        }
    }

    pub(super) fn lgraph_ids(&self, id: u32) -> Vec<u32> {
        self.lgraph_entries(id).map(|e| e.0).collect()
    }

    pub(super) fn lgraph_set(&mut self, id: u32, entries: &[(u32, f32, Vec<f32>)]) {
        if self.lgraph_ensure().is_err() {
            return;
        }
        let eb = self.lgraph.eb;
        let Some(r) = self.lgraph.record_mut(id) else {
            return;
        };
        let n = entries.len().min(LGRAPH_SLOTS);
        for (i, (l, rad, c)) in entries[..n].iter().enumerate() {
            put_entry(r, eb, i, (*l, *rad, c));
        }
        r[..4].copy_from_slice(&(n as u32).to_le_bytes());
    }

    pub(super) fn lgraph_link(&mut self, id: u32, leaf: (u32, f32, &[f32]), cap: usize) {
        if self.lgraph_ensure().is_err() {
            return;
        }
        let eb = self.lgraph.eb;
        let Some(r) = self.lgraph.record_mut(id) else {
            return;
        };
        let n = count(r);
        if let Some(i) = (0..n).find(|&i| word(r, HDR + i * eb) == leaf.0) {
            put_entry(r, eb, i, leaf);
            return;
        }
        let cap = cap.clamp(1, LGRAPH_SLOTS);
        let n = if n >= cap {
            r.copy_within(HDR + eb..HDR + n * eb, HDR);
            n - 1
        } else {
            n
        };
        put_entry(r, eb, n, leaf);
        r[..4].copy_from_slice(&((n + 1) as u32).to_le_bytes());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_fits_pages() {
        assert_eq!(record_bytes(10), 1024);
        assert_eq!(record_bytes(100), 4096);
        assert!(record_bytes(1000).is_multiple_of(PAGE));
    }

    #[test]
    fn set_link_and_read_back() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = TreeStore::create(dir.path(), 4).unwrap();
        assert_eq!(s.lgraph_entries(3).count(), 0);
        s.lgraph_set(3, &[(7, 0.5, vec![1.0, 0.0, 0.0, 0.0])]);
        s.lgraph_link(3, (9, 0.25, &[0.0, 2.0, 0.0, 0.0]), 2);
        assert_eq!(s.lgraph_ids(3), vec![7, 9]);
        s.lgraph_link(3, (11, 0.1, &[0.0, 0.0, 3.0, 0.0]), 2);
        assert_eq!(s.lgraph_ids(3), vec![9, 11]);
        s.lgraph_link(3, (9, 0.3, &[0.0, 1.0, 0.0, 0.0]), 2);
        let e: Vec<_> = s.lgraph_entries(3).collect();
        assert_eq!(e.len(), 2);
        assert_eq!((e[0].0, e[0].1), (9, 0.3));
        let q = [0.0f32, 1.0, 0.0, 0.0];
        let (d2, _) = crate::index::tree_codes::coded_sq_l2(&q, 1.0, e[0].2);
        assert!(d2 < 1e-3, "{d2}");
    }
}
