use std::collections::HashMap;

use super::buckets::BucketFile;
use crate::error::ENNError;

const KEY_BYTES: usize = 64;
const ROW_BYTES: usize = 8;

pub(crate) enum Entry {
    Rows(Vec<u32>),
    Slot(Vec<u8>),
}

#[derive(Default)]
pub(crate) struct Pending {
    entries: HashMap<u64, Entry>,
    rows: usize,
    bytes: usize,
}

impl Pending {
    pub(crate) fn push(&mut self, key: u64, row: u32, file: &BucketFile) -> Result<(), ENNError> {
        self.rows += 1;
        let entry = self.entries.entry(key).or_insert_with(|| {
            self.bytes += KEY_BYTES;
            Entry::Rows(Vec::new())
        });
        match entry {
            Entry::Slot(slot) => file.place_row(slot, row),
            Entry::Rows(rows) if rows.len() + 1 < file.cap() => {
                rows.push(row);
                self.bytes += ROW_BYTES;
            }
            Entry::Rows(rows) => {
                let mut slot = file.read_slot(key)?;
                for &r in rows.iter().chain(std::iter::once(&row)) {
                    file.place_row(&mut slot, r);
                }
                self.bytes += slot.len();
                self.bytes -= ROW_BYTES * rows.len();
                *entry = Entry::Slot(slot);
            }
        }
        Ok(())
    }

    pub(crate) fn get(&self, key: u64) -> Option<&Entry> {
        self.entries.get(&key)
    }

    pub(crate) fn rows(&self) -> usize {
        self.rows
    }

    pub(crate) fn bytes(&self) -> usize {
        self.bytes
    }

    pub(crate) fn clear(&mut self) {
        *self = Self::default();
    }

    pub(crate) fn take_sorted(&mut self) -> Vec<(u64, Entry)> {
        let mut out: Vec<(u64, Entry)> = std::mem::take(&mut self.entries).into_iter().collect();
        out.sort_unstable_by_key(|e| e.0);
        self.clear();
        out
    }
}
