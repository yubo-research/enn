use ennbo_bpann::{bpann_read_json, bpann_write_json};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::ops::Range;
use std::path::{Path, PathBuf};

use super::buckets::BucketFile;
use super::keys::{KeyFile, KeyWriter};
use super::lsh::{io_error, table_seed, target_bits, O1Params, Planes};
use crate::backend::MmapColumnStore;
use crate::error::ENNError;

const CHUNK_ROWS: usize = 4096;
pub(crate) const O1_INDEX_FILE: &str = "o1_index.json";

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub(crate) struct IndexState {
    pub num_tables: usize,
    pub bucket_cap: usize,
    pub indexed_rows: usize,
    pub hash_scale: Vec<f64>,
    pub anchors: Vec<Vec<usize>>,
}

pub(crate) struct LshIndex {
    params: O1Params,
    planes: Planes,
    keys: Vec<KeyFile>,
    buckets: Vec<BucketFile>,
    num_rows: usize,
    state_path: PathBuf,
}

fn table_path(work_dir: &Path, kind: &str, t: usize) -> PathBuf {
    work_dir.join(format!("o1_{kind}_{t}.bin"))
}

impl LshIndex {
    pub(crate) fn open(
        work_dir: &Path,
        seed: u64,
        params: O1Params,
        fallback_scale: &[f64],
        rows: &MmapColumnStore,
    ) -> Result<Self, ENNError> {
        let state_path = work_dir.join(O1_INDEX_FILE);
        let saved: Option<IndexState> = bpann_read_json(&state_path).ok().flatten();
        let planes = Planes::new(rows.ncols, seed, params.num_tables);
        let mut keys = Vec::with_capacity(params.num_tables);
        let mut buckets = Vec::with_capacity(params.num_tables);
        for t in 0..params.num_tables {
            keys.push(KeyFile::open(table_path(work_dir, "keys", t), false)?);
            let seed_t = table_seed(seed, t);
            buckets.push(BucketFile::open(table_path(work_dir, "buckets", t), params.bucket_cap, seed_t)?);
        }
        let mut index = Self { params, planes, keys, buckets, num_rows: 0, state_path };
        if let Some(state) = saved.filter(|s| index.can_restore(s, rows.nrows)) {
            index.planes.set_scale(&state.hash_scale, rows)?;
            index.planes.restore(&state.anchors, rows)?;
            index.num_rows = state.indexed_rows;
            for k in &index.keys {
                k.truncate(index.num_rows)?;
            }
            return Ok(index);
        }
        index.planes.set_scale(fallback_scale, rows)?;
        index.invalidate()?;
        index.reset()?;
        index.append(rows, 0)?;
        index.commit()?;
        Ok(index)
    }

    fn invalidate(&self) -> Result<(), ENNError> {
        match std::fs::remove_file(&self.state_path) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => Err(io_error(e)),
            _ => Ok(()),
        }
    }

    fn commit(&self) -> Result<(), ENNError> {
        bpann_write_json(&self.state_path, &self.state()).map_err(io_error)
    }

    pub(crate) fn insert_rows(&mut self, rows: &MmapColumnStore, start: usize) -> Result<(), ENNError> {
        self.invalidate()?;
        self.append(rows, start)?;
        self.commit()
    }

    pub(crate) fn rehash(&mut self, rows: &MmapColumnStore, hash_scale: &[f64]) -> Result<(), ENNError> {
        self.invalidate()?;
        self.planes.set_scale(hash_scale, rows)?;
        self.rewrite_keys(rows, 0)?;
        self.rebuild_buckets()?;
        self.commit()
    }

    fn can_restore(&self, s: &IndexState, nrows: usize) -> bool {
        let bits = s.anchors.first().map_or(0, Vec::len);
        s.num_tables == self.params.num_tables
            && s.bucket_cap == self.params.bucket_cap
            && s.indexed_rows == nrows
            && s.anchors.len() == self.params.num_tables
            && s.anchors.iter().all(|a| a.len() == bits)
            && self.keys.iter().all(|k| k.len() >= nrows)
            && self.buckets.iter().all(|b| b.has_bits(bits))
    }

    fn reset(&mut self) -> Result<(), ENNError> {
        self.planes = self.planes.cleared();
        self.num_rows = 0;
        for (k, b) in self.keys.iter().zip(&self.buckets) {
            k.truncate(0)?;
            b.reset()?;
        }
        Ok(())
    }

    pub(crate) fn state(&self) -> IndexState {
        IndexState {
            num_tables: self.params.num_tables,
            bucket_cap: self.params.bucket_cap,
            indexed_rows: self.num_rows,
            hash_scale: self.planes.hash_scale().to_vec(),
            anchors: self.planes.anchors(),
        }
    }

    pub(crate) fn bits(&self) -> usize {
        self.planes.bits()
    }

    pub(crate) fn hash_scale(&self) -> &[f64] {
        self.planes.hash_scale()
    }

    pub(crate) fn memory_bytes(&self) -> usize {
        self.planes.memory_bytes()
    }

    fn chunk_keys(&self, rows: &MmapColumnStore, range: Range<usize>, from_bit: usize) -> Result<Vec<u64>, ENNError> {
        let l = self.params.num_tables;
        let mut out = vec![0u64; range.len() * l];
        out.par_chunks_mut(l).enumerate().try_for_each(|(i, slot)| {
            let z = self.planes.scaled(rows.mmap_row_slice(range.start + i)?);
            for (t, k) in slot.iter_mut().enumerate() {
                *k = self.planes.key_bits(t, &z, from_bit);
            }
            Ok::<(), ENNError>(())
        })?;
        Ok(out)
    }

    fn column(&self, flat: &[u64], t: usize) -> Vec<u64> {
        flat.iter().skip(t).step_by(self.params.num_tables).copied().collect()
    }

    fn append(&mut self, rows: &MmapColumnStore, start: usize) -> Result<(), ENNError> {
        if u32::try_from(rows.nrows).is_err() {
            return Err(ENNError::InvalidParameter("O1 store holds at most 2^32 rows".into()));
        }
        let target = target_bits(rows.nrows, self.params.bucket_target);
        let incremental = self.bits() >= target;
        for lo in (start..rows.nrows).step_by(CHUNK_ROWS) {
            let range = lo..(lo + CHUNK_ROWS).min(rows.nrows);
            let flat = self.chunk_keys(rows, range, 0)?;
            for t in 0..self.params.num_tables {
                let col = self.column(&flat, t);
                self.keys[t].write_at(lo, &col)?;
                if incremental {
                    let mut pairs: Vec<(u64, u32)> =
                        col.iter().enumerate().map(|(i, &k)| (k, (lo + i) as u32)).collect();
                    self.buckets[t].insert(&mut pairs)?;
                }
            }
        }
        self.num_rows = rows.nrows;
        if !incremental {
            let old = self.bits();
            while self.bits() < target {
                self.planes.push_bit(rows)?;
            }
            self.rewrite_keys(rows, old)?;
            self.rebuild_buckets()?;
        }
        Ok(())
    }

    fn rewrite_keys(&mut self, rows: &MmapColumnStore, from_bit: usize) -> Result<(), ENNError> {
        let shift = self.bits() - from_bit;
        let mut writers = (0..self.params.num_tables)
            .map(|t| KeyWriter::create(self.keys[t].tmp_path()))
            .collect::<Result<Vec<_>, ENNError>>()?;
        let mut old = vec![0u64; CHUNK_ROWS];
        for lo in (0..self.num_rows).step_by(CHUNK_ROWS) {
            let range = lo..(lo + CHUNK_ROWS).min(self.num_rows);
            let len = range.len();
            let flat = self.chunk_keys(rows, range, from_bit)?;
            for (t, w) in writers.iter_mut().enumerate() {
                let mut col = self.column(&flat, t);
                if from_bit > 0 {
                    self.keys[t].read_at(lo, &mut old[..len])?;
                    for (k, o) in col.iter_mut().zip(&old) {
                        *k |= o << shift;
                    }
                }
                w.push(&col)?;
            }
        }
        for (k, w) in self.keys.iter_mut().zip(writers) {
            let tmp = w.finish()?;
            k.replace_with(&tmp)?;
        }
        Ok(())
    }

    pub(crate) fn rebuild_buckets(&mut self) -> Result<(), ENNError> {
        let bits = self.bits();
        for (b, k) in self.buckets.iter_mut().zip(&self.keys) {
            b.rebuild(k, self.num_rows, bits, self.params.build_budget_bytes)?;
        }
        Ok(())
    }

    pub(crate) fn candidate_ids(&self, query: &[f64]) -> Result<Vec<u32>, ENNError> {
        let mut ids = Vec::new();
        for (t, (key, flips)) in self.planes.query_keys(query, self.params.num_probes).into_iter().enumerate() {
            self.buckets[t].read_ids(key, &mut ids)?;
            for bit in flips {
                self.buckets[t].read_ids(key ^ (1u64 << bit), &mut ids)?;
            }
        }
        ids.sort_unstable();
        ids.dedup();
        Ok(ids)
    }
}
