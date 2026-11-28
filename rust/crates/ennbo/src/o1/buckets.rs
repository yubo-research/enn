use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use super::keys::{file_len, install, open_rw, KeyFile};
use super::lsh::{io_error, mix64};
use super::pending::Entry;
use crate::error::ENNError;

const KEY_CHUNK_ROWS: usize = 1 << 16;
const MAX_OPEN_PARTS: usize = 64;
const PART_RECORD_BYTES: usize = 8;
const FLUSH_WINDOW_BYTES: u64 = 1 << 20;
const FLUSH_GAP_BYTES: u64 = 16 << 10;

pub(crate) fn slot_bytes(cap: usize) -> usize {
    8 + 4 * cap
}

pub(crate) fn rebuild_partitions(bits: usize, cap: usize, budget: usize) -> (usize, usize) {
    let slots = 1usize << bits;
    let fit = (budget / slot_bytes(cap)).max(1);
    let part_slots = (1usize << fit.ilog2()).min(slots);
    (part_slots, slots / part_slots)
}

fn place(slot: &mut [u8], row: u32, cap: usize, seed: u64) {
    let arrivals = u64::from_le_bytes(slot[..8].try_into().expect("slot header")) + 1;
    slot[..8].copy_from_slice(&arrivals.to_le_bytes());
    let pos = if arrivals <= cap as u64 {
        Some(arrivals as usize - 1)
    } else {
        let j = mix64(seed ^ u64::from(row)) % arrivals;
        (j < cap as u64).then_some(j as usize)
    };
    if let Some(j) = pos {
        slot[8 + 4 * j..12 + 4 * j].copy_from_slice(&row.to_le_bytes());
    }
}

fn for_each_key(
    keys: &KeyFile,
    n: usize,
    mut f: impl FnMut(u32, u64) -> Result<(), ENNError>,
) -> Result<(), ENNError> {
    let mut chunk = vec![0u64; KEY_CHUNK_ROWS];
    for start in (0..n).step_by(KEY_CHUNK_ROWS) {
        let len = KEY_CHUNK_ROWS.min(n - start);
        keys.read_at(start, &mut chunk[..len])?;
        for (i, &key) in chunk[..len].iter().enumerate() {
            f((start + i) as u32, key)?;
        }
    }
    Ok(())
}

pub(crate) struct BucketFile {
    path: PathBuf,
    file: File,
    cap: usize,
    seed: u64,
}

impl BucketFile {
    pub(crate) fn open(path: PathBuf, cap: usize, seed: u64) -> Result<Self, ENNError> {
        let file = open_rw(&path, false)?;
        Ok(Self {
            path,
            file,
            cap,
            seed,
        })
    }

    pub(crate) fn has_bits(&self, bits: usize) -> bool {
        file_len(&self.path) == ((1u64 << bits) * slot_bytes(self.cap) as u64)
    }

    pub(crate) fn reset(&self) -> Result<(), ENNError> {
        self.file.set_len(0).map_err(io_error)?;
        self.file.set_len(self.sb() as u64).map_err(io_error)
    }

    fn sb(&self) -> usize {
        slot_bytes(self.cap)
    }

    pub(crate) fn cap(&self) -> usize {
        self.cap
    }

    pub(crate) fn read_slot(&self, key: u64) -> Result<Vec<u8>, ENNError> {
        let mut slot = vec![0u8; self.sb()];
        self.file
            .read_exact_at(&mut slot, key * self.sb() as u64)
            .map_err(io_error)?;
        Ok(slot)
    }

    pub(crate) fn place_row(&self, slot: &mut [u8], row: u32) {
        place(slot, row, self.cap, self.seed);
    }

    pub(crate) fn read_ids(
        &self,
        key: u64,
        pending: Option<&Entry>,
        out: &mut Vec<u32>,
    ) -> Result<u64, ENNError> {
        let owned;
        let slot: &[u8] = match pending {
            Some(Entry::Slot(slot)) => slot,
            rows => {
                let mut slot = self.read_slot(key)?;
                if let Some(Entry::Rows(rows)) = rows {
                    rows.iter().for_each(|&r| self.place_row(&mut slot, r));
                }
                owned = slot;
                &owned
            }
        };
        let arrivals = u64::from_le_bytes(slot[..8].try_into().expect("slot header"));
        let count = (arrivals as usize).min(self.cap);
        out.extend(
            slot[8..8 + 4 * count]
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().expect("4-byte id"))),
        );
        Ok(arrivals)
    }

    pub(crate) fn write_pending(&self, entries: &[(u64, Entry)]) -> Result<(), ENNError> {
        let sb = self.sb() as u64;
        let mut buf = Vec::new();
        let mut lo = 0;
        while lo < entries.len() {
            let first = entries[lo].0;
            let mut hi = lo + 1;
            while hi < entries.len()
                && (entries[hi].0 - first + 1) * sb <= FLUSH_WINDOW_BYTES
                && (entries[hi].0 - entries[hi - 1].0) * sb <= FLUSH_GAP_BYTES
            {
                hi += 1;
            }
            let last = entries[hi - 1].0;
            buf.resize(((last - first + 1) * sb) as usize, 0);
            self.file
                .read_exact_at(&mut buf, first * sb)
                .map_err(io_error)?;
            for (key, entry) in &entries[lo..hi] {
                let at = ((key - first) * sb) as usize;
                let slot = &mut buf[at..at + sb as usize];
                match entry {
                    Entry::Slot(bytes) => slot.copy_from_slice(bytes),
                    Entry::Rows(rows) => rows.iter().for_each(|&r| self.place_row(slot, r)),
                }
            }
            self.file.write_all_at(&buf, first * sb).map_err(io_error)?;
            lo = hi;
        }
        Ok(())
    }

    pub(crate) fn rebuild(
        &mut self,
        keys: &KeyFile,
        n: usize,
        bits: usize,
        budget: usize,
    ) -> Result<(), ENNError> {
        let sb = self.sb();
        let (part_slots, parts) = rebuild_partitions(bits, self.cap, budget);
        let tmp = self.path.with_extension("tmp");
        let out = open_rw(&tmp, true)?;
        out.set_len((1u64 << bits) * sb as u64).map_err(io_error)?;
        if parts == 1 {
            let mut buf = vec![0u8; part_slots * sb];
            for_each_key(keys, n, |row, key| {
                let at = key as usize * sb;
                place(&mut buf[at..at + sb], row, self.cap, self.seed);
                Ok(())
            })?;
            out.write_all_at(&buf, 0).map_err(io_error)?;
        } else {
            let scratch = self.path.with_extension("parts");
            std::fs::create_dir_all(&scratch).map_err(io_error)?;
            for first in (0..parts).step_by(MAX_OPEN_PARTS) {
                let window = first..(first + MAX_OPEN_PARTS).min(parts);
                self.distribute(keys, n, part_slots, window.clone(), &scratch)?;
                for p in window {
                    self.fill_part(&out, &scratch, p, part_slots)?;
                }
            }
            std::fs::remove_dir_all(&scratch).map_err(io_error)?;
        }
        drop(out);
        install(&tmp, &self.path)?;
        self.file = open_rw(&self.path, false)?;
        Ok(())
    }

    fn distribute(
        &self,
        keys: &KeyFile,
        n: usize,
        part_slots: usize,
        window: std::ops::Range<usize>,
        scratch: &Path,
    ) -> Result<(), ENNError> {
        let mut writers = window
            .clone()
            .map(|p| {
                Ok(BufWriter::with_capacity(
                    1 << 16,
                    open_rw(&part_path(scratch, p), true)?,
                ))
            })
            .collect::<Result<Vec<_>, ENNError>>()?;
        for_each_key(keys, n, |row, key| {
            let p = key as usize / part_slots;
            if !window.contains(&p) {
                return Ok(());
            }
            let local = (key as usize % part_slots) as u32;
            let w = &mut writers[p - window.start];
            w.write_all(&local.to_le_bytes()).map_err(io_error)?;
            w.write_all(&row.to_le_bytes()).map_err(io_error)
        })?;
        for mut w in writers {
            w.flush().map_err(io_error)?;
        }
        Ok(())
    }

    fn fill_part(
        &self,
        out: &File,
        scratch: &Path,
        p: usize,
        part_slots: usize,
    ) -> Result<(), ENNError> {
        let sb = self.sb();
        let path = part_path(scratch, p);
        let mut reader = BufReader::with_capacity(1 << 16, File::open(&path).map_err(io_error)?);
        let mut buf = vec![0u8; part_slots * sb];
        let mut rec = [0u8; PART_RECORD_BYTES];
        loop {
            match reader.read_exact(&mut rec) {
                Ok(()) => {}
                Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(io_error(e)),
            }
            let local = u32::from_le_bytes(rec[..4].try_into().expect("local key")) as usize;
            let row = u32::from_le_bytes(rec[4..].try_into().expect("row id"));
            place(
                &mut buf[local * sb..(local + 1) * sb],
                row,
                self.cap,
                self.seed,
            );
        }
        out.write_all_at(&buf, (p * part_slots * sb) as u64)
            .map_err(io_error)?;
        std::fs::remove_file(&path).map_err(io_error)
    }
}

fn part_path(scratch: &Path, p: usize) -> PathBuf {
    scratch.join(format!("part_{p}.bin"))
}
