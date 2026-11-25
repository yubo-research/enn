use std::ops::Deref;
use std::sync::Arc;
use std::time::Instant;

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

use super::cache::{CacheKey, Cached, ObjCache};
use super::forest_query::Scored;
use super::index_kind::O1Index;
use super::profile::{QueryProfile, Stage};
use super::row_order::Rows;
use crate::error::ENNError;

pub(crate) const X_COLUMN: u8 = 0;

pub(crate) enum RowRef<'a> {
    Mapped(&'a [f64]),
    Shared(Arc<[f64]>),
}

impl Deref for RowRef<'_> {
    type Target = [f64];

    fn deref(&self) -> &[f64] {
        match self {
            Self::Mapped(r) => r,
            Self::Shared(r) => r,
        }
    }
}

pub(crate) fn fetch_row<'a>(
    store: Rows<'a>,
    cache: Option<&ObjCache>,
    column: u8,
    i: usize,
) -> Result<RowRef<'a>, ENNError> {
    let Some(cache) = cache.filter(|c| c.caches_rows()) else {
        return store.mapped(i).map(RowRef::Mapped);
    };
    let key = CacheKey::Row { column, row: i as u32 };
    match cache.get_or_load(key, || store.pread(i).map(|r| Cached::Row(r.into())))? {
        Cached::Row(r) => Ok(RowRef::Shared(r)),
        Cached::Node(_) => Err(ENNError::InvalidParameter("row cache entry holds a node".into())),
    }
}

pub(crate) const O1_EXACT_ROWS: usize = 256;
const ROW_PROBE: usize = 8;
const READ_AROUND: usize = 128 << 10;
const PAGE: usize = 4096;

fn scattered(ids: &[u32], rows: Rows<'_>) -> bool {
    let row_bytes = rows.row_bytes();
    let per = (READ_AROUND / row_bytes.max(1)).max(1) as u64;
    let mut windows: Vec<u64> = ids.iter().map(|&i| rows.position(i) / per).collect();
    windows.sort_unstable();
    windows.dedup();
    ids.len() * row_bytes.max(PAGE) < windows.len() * READ_AROUND
}
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

pub(crate) fn weighted_dist2(a: &[f64], b: &[f64], weights: &[f64]) -> f64 {
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
    pub index: &'a O1Index,
    pub rows: Rows<'a>,
    pub weights: &'a [f64],
    pub cache: Option<&'a ObjCache>,
    pub profile: &'a QueryProfile,
    pub overlap: bool,
}

impl SearchContext<'_> {
    pub(crate) fn candidates(&self, query: &[f64], need: usize) -> Result<Vec<u32>, ENNError> {
        let n = self.rows.nrows();
        let mut ids = if n <= O1_EXACT_ROWS {
            (0..n as u32).collect()
        } else {
            self.index.candidate_ids_at(query, self.profile, self.rows, self.overlap)?
        };
        top_up(&mut ids, n, need);
        Ok(ids)
    }

    fn score(&self, query: &[f64], ids: &[u32]) -> Result<Vec<(f64, u32)>, ENNError> {
        ids.iter()
            .map(|&id| Ok((weighted_dist2(query, &fetch_row(self.rows, self.cache, X_COLUMN, id as usize)?, self.weights), id)))
            .collect()
    }

    fn score_read(&self, query: &[f64], ids: &[u32], pread: bool) -> Result<Vec<(f64, u32)>, ENNError> {
        if !pread {
            return self.score(query, ids);
        }
        ids.iter().map(|&id| Ok((weighted_dist2(query, &self.rows.pread(id as usize)?, self.weights), id))).collect()
    }

    fn scored(&self, query: &[f64], ids: &[u32]) -> Result<Vec<(f64, u32)>, ENNError> {
        let (width, slow_ns) = self.index.io_width();
        let head = ids.len().min(ROW_PROBE);
        let t0 = Instant::now();
        let mut out = self.score(query, &ids[..head])?;
        let rest = &ids[head..];
        if !self.overlap || width <= 1 || rest.len() < width || (t0.elapsed().as_nanos() as u64) < slow_ns {
            out.extend(self.score(query, rest)?);
            return Ok(out);
        }
        let pread = !self.cache.is_some_and(ObjCache::caches_rows) && scattered(rest, self.rows);
        let parts: Vec<Result<Vec<(f64, u32)>, ENNError>> = std::thread::scope(|scope| {
            let jobs: Vec<_> = rest.chunks(rest.len().div_ceil(width)).map(|c| scope.spawn(move || self.score_read(query, c, pread))).collect();
            jobs.into_iter().map(|j| j.join().unwrap_or_else(|_| Err(ENNError::InvalidParameter("row worker panicked".into())))).collect()
        });
        for part in parts {
            out.extend(part?);
        }
        Ok(out)
    }

    fn ranked(&self, query: &[f64], need: usize) -> Result<Vec<(f64, u32)>, ENNError> {
        let t_search = self.profile.start();
        if self.rows.nrows() > O1_EXACT_ROWS {
            if let Some(mut scored) = self.index.scored_at(query, self.profile, self.rows, self.overlap, self.weights, need)?.map(|s| self.reranked(query, s, need)).transpose()?.filter(|s| s.len() >= need) {
                self.profile.count_query(scored.len());
                let out = Self::nearest(&mut scored, need);
                self.profile.stop(Stage::Search, t_search);
                return Ok(out);
            }
        }
        let ids = self.candidates(query, need)?;
        self.profile.count_query(ids.len());
        if !self.profile.enabled() {
            let mut scored = self.scored(query, &ids)?;
            return Ok(Self::nearest(&mut scored, need));
        }
        let t_rows = self.profile.start();
        let rows = ids
            .iter()
            .map(|&id| fetch_row(self.rows, self.cache, X_COLUMN, id as usize))
            .collect::<Result<Vec<_>, _>>()?;
        if t_rows.is_some() {
            std::hint::black_box(rows.iter().map(|r| r.iter().sum::<f64>()).sum::<f64>());
        }
        self.profile.stop(Stage::Row, t_rows);
        let t_dist = self.profile.start();
        let mut scored: Vec<(f64, u32)> =
            rows.iter().zip(&ids).map(|(row, &id)| (weighted_dist2(query, row, self.weights), id)).collect();
        self.profile.stop(Stage::Distance, t_dist);
        let out = Self::nearest(&mut scored, need);
        self.profile.stop(Stage::Search, t_search);
        Ok(out)
    }

    fn reranked(&self, query: &[f64], s: Scored, need: usize) -> Result<Vec<(f64, u32)>, ENNError> {
        let Scored { mut exact, mut coded, rerank } = s;
        let total = coded.len();
        if total > 0 {
            let mut ids: Vec<u32> = Self::nearest(&mut coded, rerank.max(need)).into_iter().map(|(_, id)| id).collect();
            ids.sort_unstable();
            self.profile.count_codes(total, ids.len());
            exact.extend(self.scored(query, &ids)?);
        }
        Ok(exact)
    }

    fn nearest(scored: &mut Vec<(f64, u32)>, need: usize) -> Vec<(f64, u32)> {
        let keep = need.min(scored.len());
        if keep < scored.len() {
            scored.select_nth_unstable_by(keep, |a, b| a.0.total_cmp(&b.0));
            scored.truncate(keep);
        }
        scored.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        std::mem::take(scored)
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
        let n = self.rows.nrows();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::o1::O1Store;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use tempfile::TempDir;

    #[test]
    fn scattered_rows_are_read_one_by_one_with_the_same_answers() {
        let dummy = tempfile::TempDir::new().unwrap();
        let x100 = crate::backend::MmapColumnStore::mmap_open_or_create(dummy.path().join("x.bin"), 100, Some(0)).unwrap();
        assert!(!scattered(&(0..1280).collect::<Vec<u32>>(), Rows::plain(&x100)));
        assert!(scattered(&(0..1280).map(|i| i * 1000).collect::<Vec<u32>>(), Rows::plain(&x100)));
        let d = 600;
        let mut rng = StdRng::seed_from_u64(5);
        let x = Array2::from_shape_fn((1500, d), |_| rng.gen::<f64>());
        let y = Array2::from_shape_fn((1500, 1), |(i, _)| x[[i, 0]]);
        let dir = TempDir::new().unwrap();
        let mut store = O1Store::open_forest(dir.path().to_path_buf(), d, 1).unwrap();
        store.append_rows(&x.view(), &y.view(), None).unwrap();
        let q = Array2::from_shape_fn((4, d), |_| rng.gen::<f64>());
        let batch = store.search(&q.view(), 10, false).unwrap();
        store.set_io_width(4, 0).unwrap();
        for i in 0..q.nrows() {
            let one = store.search(&q.slice(ndarray::s![i..i + 1, ..]), 10, false).unwrap();
            assert_eq!((one.0.row(0).to_owned(), one.1.row(0).to_owned()), (batch.0.row(i).to_owned(), batch.1.row(i).to_owned()));
        }
    }
}
