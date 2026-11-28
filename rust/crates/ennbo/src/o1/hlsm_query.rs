use std::collections::{BinaryHeap, HashMap};

use super::disk_index::O1Occupancy;
use super::forest_query::Scored;
use super::hlsm::HlsmIndex;
use super::hlsm_hash::lut_score;
use super::hlsm_io::par_map;
use super::hlsm_run::{entry_hash, entry_id, ENTRY_HEAD};
use super::lsm_key::metric_factor;
use super::lsm_query::{profiled, QueryIo, TopK};
use super::profile::QueryProfile;
use super::row_order::Rows;
use super::search::weighted_dist2;
use crate::error::ENNError;

impl HlsmIndex {
    fn coded(&self, query: &[f64], weights: &[f64], io: &mut QueryIo) -> Result<Vec<(f64, u32)>, ENNError> {
        let (d, k) = (self.meta.d, self.meta.k);
        let mut luts: Vec<Option<Vec<f64>>> = vec![None; self.books.len()];
        let mut slot: HashMap<u32, usize> = HashMap::new();
        let mut out: Vec<(f64, f64, u32)> = Vec::new();
        let nr = self.runs.len();
        let mut zq = vec![0.0; nr * k];
        let mut keys = vec![0u64; nr];
        let mut cr = vec![0.0; nr];
        for (r, run) in self.runs.iter().enumerate() {
            keys[r] = self.hashes[run.meta.table].key(query, &run.frame, &mut zq[r * k..(r + 1) * k]);
            cr[r] = metric_factor(&run.frame.scale, weights) * (1.0 - 1e-9);
        }
        let plan: Vec<(usize, usize, usize)> = (0..nr)
            .map(|r| (r, self.runs[r].window(keys[r], self.window, self.entry)))
            .filter(|(_, (lo, hi))| hi > lo)
            .map(|(r, (lo, hi))| (r, lo, hi))
            .collect();
        let bufs = par_map(&plan, self.io, |&(r, lo, hi)| {
            let mut b = Vec::new();
            self.runs[r].read(lo, hi, self.entry, &mut b).map(|_| b)
        });
        for (&(r, lo, hi), buf) in plan.iter().zip(bufs) {
            let (buf, run) = (buf?, &self.runs[r]);
            let t = run.meta.table;
            io.reads += 1;
            io.blocks += ((hi - lo) * self.entry).div_ceil(4096) as u64;
            for e in buf.chunks_exact(self.entry) {
                let id = entry_id(e);
                let hb = cr[r] * self.hashes[t].proj_lb2(&zq[r * k..(r + 1) * k], entry_hash(e), &run.frame);
                if let Some(&i) = slot.get(&id) {
                    out[i].1 = out[i].1.max(hb);
                    continue;
                }
                let b = self.book_of(id);
                let lut = luts[b].get_or_insert_with(|| self.books[b].lut_lb(query, weights));
                slot.insert(id, out.len());
                out.push((lut_score(lut, &e[ENTRY_HEAD..], d), hb, id));
            }
        }
        io.nodes += out.len() as u64;
        Ok(out.into_iter().map(|(c, h, id)| (c.max(h), id)).collect())
    }

    pub(crate) fn scored(&self, query: &[f64], rows: Rows<'_>, weights: &[f64], need: usize) -> Result<(Vec<(f64, u32)>, QueryIo), ENNError> {
        let mut top = TopK { need: need.max(1), heap: BinaryHeap::new() };
        for i in self.meta.flushed as usize..rows.nrows() {
            let x = rows.fetch(i, false)?;
            top.offer(weighted_dist2(query, &x, weights), i as u32);
        }
        let mut io = QueryIo::default();
        let mut cands = self.coded(query, weights, &mut io)?;
        let take = self.rerank.max(need).min(cands.len());
        if take < cands.len() {
            cands.select_nth_unstable_by(take, |a, b| a.0.total_cmp(&b.0));
            cands.truncate(take);
        }
        let exact = par_map(&cands, self.io, |&(_, id)| rows.pread(id as usize).map(|x| (weighted_dist2(query, &x, weights), id)));
        for hit in exact {
            let (dist, id) = hit?;
            top.offer(dist, id);
        }
        Ok((top.sorted(), io))
    }

    pub(crate) fn scored_at(&self, query: &[f64], prof: &QueryProfile, rows: Rows<'_>, weights: &[f64], need: usize) -> Result<Scored, ENNError> {
        Ok(profiled(self.scored(query, rows, weights, need)?, prof))
    }

    pub(crate) fn candidate_ids(&self, query: &[f64], rows: Rows<'_>) -> Result<Vec<u32>, ENNError> {
        let w: Vec<f64> = self.meta.target.iter().map(|s| 1.0 / (s * s)).collect();
        let (hits, _) = self.scored(query, rows, &w, 32)?;
        let mut ids: Vec<u32> = hits.into_iter().map(|h| h.1).collect();
        ids.sort_unstable();
        Ok(ids)
    }

    pub(crate) fn occupancy(&self) -> O1Occupancy {
        O1Occupancy {
            num_rows: self.n,
            reachable_rows: self.n,
            disk_nodes: self.runs.len(),
            node_bytes: self.entry,
            splits: self.meta.stats.merges,
            rebuilt_rows: self.meta.stats.merged_rows,
            max_rebuild_rows: self.meta.stats.max_merge_rows,
            ..O1Occupancy::default()
        }
    }
}
