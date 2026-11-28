use std::collections::{BTreeMap, HashMap};
use std::io::Write;

use super::graph::GraphIndex;
use super::graph_io::Items;
use super::lsh::io_error;
use crate::backend::MmapColumnStore;
use crate::error::ENNError;

const BUF_LOG: &str = "o1g_buf.bin";

impl GraphIndex {
    pub(crate) fn buffer(&mut self, groups: BTreeMap<u32, Items>) -> Result<BTreeMap<u32, Items>, ENNError> {
        if self.buf_cap == 0 {
            return Ok(groups);
        }
        let mut new_ids = Vec::new();
        for (leaf, rows) in groups {
            new_ids.extend_from_slice(&rows.ids);
            self.buf_rows += rows.len();
            let b = self.buf.entry(leaf).or_default();
            b.ids.extend_from_slice(&rows.ids);
            b.x.extend_from_slice(&rows.x);
        }
        if self.buf_rows <= self.buf_cap {
            self.log_append(&new_ids)?;
            return Ok(BTreeMap::new());
        }
        let due = self.take_largest(self.buf_cap / 2);
        self.log_rewrite()?;
        Ok(due)
    }

    fn take_largest(&mut self, keep: usize) -> BTreeMap<u32, Items> {
        let mut by: Vec<(usize, u32)> = self.buf.iter().map(|(&l, it)| (it.len(), l)).collect();
        by.sort_unstable_by(|a, b| b.cmp(a));
        let mut due = BTreeMap::new();
        for (n, leaf) in by {
            if self.buf_rows <= keep {
                break;
            }
            if let Some(it) = self.buf.remove(&leaf) {
                self.buf_rows -= n;
                self.meta.stats.buf_flushed_rows += n as u64;
                due.insert(leaf, it);
            }
        }
        self.meta.stats.buf_flushes += 1;
        due
    }

    fn log_append(&self, ids: &[u32]) -> Result<(), ENNError> {
        let mut f = std::fs::OpenOptions::new().create(true).append(true).open(self.dir.join(BUF_LOG)).map_err(io_error)?;
        let bytes: Vec<u8> = ids.iter().flat_map(|i| i.to_le_bytes()).collect();
        f.write_all(&bytes).map_err(io_error)
    }

    fn log_rewrite(&self) -> Result<(), ENNError> {
        let bytes: Vec<u8> = self.buf.values().flat_map(|it| it.ids.iter().flat_map(|i| i.to_le_bytes())).collect();
        std::fs::write(self.dir.join(BUF_LOG), bytes).map_err(io_error)
    }

    pub(crate) fn reload_buffer(&mut self, rows: &MmapColumnStore) -> Result<(), ENNError> {
        let Ok(bytes) = std::fs::read(self.dir.join(BUF_LOG)) else {
            return Ok(());
        };
        let w = self.weights();
        let mut memo = HashMap::new();
        let mut groups: BTreeMap<u32, Items> = BTreeMap::new();
        for c in bytes.chunks_exact(4) {
            let id = u32::from_le_bytes(c.try_into().unwrap_or_default());
            let r = rows.mmap_row_slice(id as usize)?;
            groups.entry(self.route(r, &w, &mut memo)?).or_default().push(id, r);
        }
        let due = match self.buf_cap {
            0 => groups,
            _ => {
                self.buf_rows = groups.values().map(|g| g.len()).sum();
                self.buf = groups;
                BTreeMap::new()
            }
        };
        self.apply(due, &w)?;
        self.log_rewrite()
    }
}
