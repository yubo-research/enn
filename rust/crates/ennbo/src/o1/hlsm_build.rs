use std::time::Instant;

use super::hlsm::HlsmIndex;
use super::hlsm_hash::TableFrame;
use super::hlsm_run::{remove_run, ERun, ERunMeta};
use super::lsh::io_error;
use crate::backend::MmapColumnStore;
use crate::error::ENNError;

const FIT_SAMPLE: usize = 4096;

impl HlsmIndex {
    fn fit_frames(&mut self, tables: &[usize], (lo, hi): (usize, usize), rows: &MmapColumnStore) -> Result<Vec<TableFrame>, ENNError> {
        let (k, step) = (self.meta.k, (hi - lo).div_ceil(FIT_SAMPLE).max(1));
        let mut z: Vec<Vec<f64>> = tables.iter().map(|_| Vec::with_capacity(FIT_SAMPLE * k)).collect();
        let mut zr = vec![0.0; k];
        for i in (lo..hi).step_by(step) {
            let x: Vec<f64> = rows.mmap_row_range(i, i + 1)?.iter().copied().collect();
            self.meta.stats.row_bytes_read += (x.len() * 8) as u64;
            for (zt, &t) in z.iter_mut().zip(tables) {
                self.hashes[t].project(&x, &self.meta.target, &mut zr);
                zt.extend_from_slice(&zr);
            }
        }
        Ok(z.iter().map(|zt| TableFrame::from_projections(zt, k, &self.meta.target)).collect())
    }

    fn rekey_range(&mut self, tables: &[usize], (lo, hi): (usize, usize), level: u32, frames: &[TableFrame], rows: &MmapColumnStore) -> Result<Vec<ERunMeta>, ENNError> {
        let chunk = ((2usize << 20) / (8 * self.meta.d)).max(256);
        let mut temps: Vec<Vec<ERun>> = tables.iter().map(|_| Vec::new()).collect();
        for a in (lo..hi).step_by(chunk) {
            let b = (a + chunk).min(hi);
            let x: Vec<f64> = rows.mmap_row_range(a, b)?.iter().copied().collect();
            self.meta.stats.row_bytes_read += (x.len() * 8) as u64;
            let ids: Vec<u32> = (a as u32..b as u32).collect();
            let codes = self.encode(&ids, &x);
            for (i, &t) in tables.iter().enumerate() {
                let entries = self.entries(&frames[i], t, &ids, &x, &codes);
                let id = self.take_id();
                let meta = self.write_run(id, (t, 0), &entries, &frames[i], (a as u64, b as u64))?;
                temps[i].push(ERun::open(&self.dir, meta)?);
            }
        }
        let mut out = Vec::with_capacity(tables.len());
        for (i, runs) in temps.into_iter().enumerate() {
            let temp_ids: Vec<u64> = runs.iter().map(|r| r.meta.id).collect();
            out.push(self.merge_into(runs, tables[i], level)?);
            temp_ids.iter().for_each(|&id| remove_run(&self.dir, id));
        }
        Ok(out)
    }

    fn level_range(&self, t: usize, level: u32) -> (u64, u64, u64) {
        let mine = || self.meta.runs.iter().filter(|r| r.table == t && r.level == level);
        (mine().map(|r| r.lo).min().unwrap_or(0), mine().map(|r| r.hi).max().unwrap_or(0), mine().map(|r| r.rows).sum())
    }

    pub(crate) fn compact_all(&mut self, rows: &MmapColumnStore) -> Result<(), ENNError> {
        while let Some((t, level)) = (0..self.tables()).find_map(|t| self.full_level(t).map(|l| (t, l))) {
            let t0 = Instant::now();
            let range = self.level_range(t, level);
            if range.2 != range.1 - range.0 {
                return Err(io_error(format!("O1E runs of table {t} at level {level} do not cover a contiguous id range")));
            }
            let group: Vec<usize> = (0..self.tables()).filter(|&u| self.full_level(u) == Some(level) && self.level_range(u, level) == range).collect();
            let span = (range.0 as usize, range.1 as usize);
            let frames = self.fit_frames(&group, span, rows)?;
            let old: Vec<ERunMeta> = group.iter().flat_map(|&u| self.take_level(u, level)).map(|r| r.meta).collect();
            let metas = self.rekey_range(&group, span, level + 1, &frames, rows)?;
            self.install(metas, &old, t0)?;
        }
        Ok(())
    }

    pub(crate) fn rebuild_table(&mut self, t: usize, rows: &MmapColumnStore) -> Result<(), ENNError> {
        let t0 = Instant::now();
        let old: Vec<ERunMeta> = self.meta.runs.iter().filter(|r| r.table == t).cloned().collect();
        let mut fresh = Vec::new();
        let mut last = self.hashes[t].fit(&[], &self.meta.target);
        for r in &old {
            let span = (r.lo as usize, r.hi as usize);
            let frames = self.fit_frames(&[t], span, rows)?;
            fresh.extend(self.rekey_range(&[t], span, r.level, &frames, rows)?);
            last = frames.into_iter().next().unwrap_or(last);
        }
        self.meta.frames[t] = Some(last);
        self.runs.retain(|r| r.meta.table != t);
        self.meta.runs.retain(|r| r.table != t);
        for m in fresh {
            self.runs.push(ERun::open(&self.dir, m.clone())?);
            self.meta.runs.push(m);
        }
        let ns = t0.elapsed().as_nanos() as u64;
        let st = &mut self.meta.stats;
        st.rebuilds += 1;
        st.rebuild_ns += ns;
        st.max_rebuild_ns = st.max_rebuild_ns.max(ns);
        self.save()?;
        old.iter().for_each(|r| remove_run(&self.dir, r.id));
        Ok(())
    }
}
