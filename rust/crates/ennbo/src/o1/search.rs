use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

use super::disk_index::LshIndex;
use crate::backend::MmapColumnStore;
use crate::error::ENNError;

pub(crate) const O1_EXACT_ROWS: usize = 256;
const SELF_DIST2_EPS: f64 = 1e-24;

fn top_up(ids: &mut Vec<u32>, n: usize, need: usize) {
    let mut extra = Vec::new();
    let mut row = n;
    while ids.len() + extra.len() < need && row > 0 {
        row -= 1;
        if ids.binary_search(&(row as u32)).is_err() {
            extra.push(row as u32);
        }
    }
    ids.extend(extra);
}

fn weighted_dist2(a: &[f64], b: &[f64], weights: &[f64]) -> f64 {
    let mut acc = [0.0f64; 4];
    let body = a.len() - a.len() % 4;
    for ((ca, cb), cw) in a[..body]
        .chunks_exact(4)
        .zip(b[..body].chunks_exact(4))
        .zip(weights[..body].chunks_exact(4))
    {
        for (l, slot) in acc.iter_mut().enumerate() {
            let d = ca[l] - cb[l];
            *slot += cw[l] * d * d;
        }
    }
    let tail: f64 = (body..a.len())
        .map(|j| weights[j] * (a[j] - b[j]) * (a[j] - b[j]))
        .sum();
    acc.iter().sum::<f64>() + tail
}

pub(crate) struct SearchContext<'a> {
    pub index: &'a LshIndex,
    pub rows: &'a MmapColumnStore,
    pub weights: &'a [f64],
}

impl SearchContext<'_> {
    fn ranked(&self, query: &[f64], need: usize) -> Result<Vec<(f64, u32)>, ENNError> {
        let n = self.rows.nrows;
        let mut ids = if n <= O1_EXACT_ROWS {
            (0..n as u32).collect()
        } else {
            self.index.candidate_ids(query)?
        };
        top_up(&mut ids, n, need);
        let mut scored = Vec::with_capacity(ids.len());
        for id in ids {
            let row = self.rows.mmap_row_slice(id as usize)?;
            scored.push((weighted_dist2(query, row, self.weights), id));
        }
        let keep = need.min(scored.len());
        if keep < scored.len() {
            scored.select_nth_unstable_by(keep, |a, b| a.0.total_cmp(&b.0));
            scored.truncate(keep);
        }
        scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        Ok(scored)
    }

    fn ranked_excluding(
        &self,
        query: &[f64],
        fetch: usize,
        exclude_nearest: bool,
    ) -> Result<Vec<(f64, u32)>, ENNError> {
        let mut ranked = self.ranked(query, fetch)?;
        if exclude_nearest && !ranked.is_empty() {
            let self_pos = ranked.iter().position(|r| r.0 <= SELF_DIST2_EPS);
            ranked.remove(self_pos.unwrap_or(ranked.len() - 1));
        }
        Ok(ranked)
    }

    pub(crate) fn search(
        &self,
        queries: &ArrayView2<f64>,
        search_k: usize,
        exclude_nearest: bool,
    ) -> Result<(Array2<f64>, Array2<i64>), ENNError> {
        let n = self.rows.nrows;
        let fetch = (search_k + usize::from(exclude_nearest)).min(n);
        let k_out = search_k.min(n.saturating_sub(usize::from(exclude_nearest)));
        let mut dist = Array2::from_elem((queries.nrows(), k_out), f64::INFINITY);
        let mut idx = Array2::zeros((queries.nrows(), k_out));
        let ranked: Result<Vec<Vec<(f64, u32)>>, ENNError> = (0..queries.nrows())
            .into_par_iter()
            .map(|qi| self.ranked_excluding(&queries.row(qi).to_vec(), fetch, exclude_nearest))
            .collect();
        for (qi, row) in ranked?.into_iter().enumerate() {
            for (j, (d, id)) in row.into_iter().take(k_out).enumerate() {
                dist[[qi, j]] = d;
                idx[[qi, j]] = i64::from(id);
            }
        }
        Ok((dist, idx))
    }
}
