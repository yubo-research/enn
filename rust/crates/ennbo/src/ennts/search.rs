use std::collections::HashSet;

use ndarray::{Array2, Array3, ArrayView2};

use crate::error::ENNError;

use super::lsh::probed_keys;
use super::metric::{self, keys_for_scale};
use super::store::{target_of, window_of, Posting};
use super::{EnnTs, QueryOptions};

#[derive(Clone, Debug)]
pub(super) struct Candidate {
    pub series_id: u64,
    pub end_ts: i64,
    pub dist2: f64,
    pub target: Vec<f64>,
}

pub struct SearchBatch {
    pub series_id: Array2<i64>,
    pub end_timestamp: Array2<i64>,
    pub distance: Array2<f64>,
    pub target: Array3<f64>,
}

pub(super) fn search(
    model: &EnnTs,
    queries: ArrayView2<f64>,
    options: &QueryOptions,
    exact: bool,
) -> Result<SearchBatch, ENNError> {
    let ranked = rank_all(model, queries, options, exact)?;
    Ok(pack(&ranked, options.k, model.metrics))
}

pub(super) fn rank_all(
    model: &EnnTs,
    queries: ArrayView2<f64>,
    options: &QueryOptions,
    exact: bool,
) -> Result<Vec<Vec<Candidate>>, ENNError> {
    check_query(model, &queries, options)?;
    if !exact && model.windows.len() > super::ann::TREE_MIN_WINDOWS {
        return super::ann::rank_large(model, queries, options);
    }
    let mut ranked = Vec::with_capacity(queries.nrows());
    for i in 0..queries.nrows() {
        ranked.push(rank_one(model, &queries.row(i).to_vec(), options, exact)?);
    }
    Ok(ranked)
}

fn check_query(
    model: &EnnTs,
    queries: &ArrayView2<f64>,
    options: &QueryOptions,
) -> Result<(), ENNError> {
    if options.k == 0 {
        return Err(ENNError::InvalidParameter("k must be >= 1".into()));
    }
    let dim = model
        .dim
        .ok_or_else(|| ENNError::InvalidParameter("add observations before searching".into()))?;
    let width = model.lookback * dim;
    if queries.ncols() != width || queries.iter().any(|v| !v.is_finite()) {
        return Err(ENNError::InvalidParameter(format!(
            "query rows must be finite and have length {width}"
        )));
    }
    Ok(())
}

fn rank_one(
    model: &EnnTs,
    window: &[f64],
    options: &QueryOptions,
    exact: bool,
) -> Result<Vec<Candidate>, ENNError> {
    let postings = if exact {
        exact_postings(model, options)?
    } else {
        lsh_postings(model, window, options)?
    };
    let mut cands = Vec::new();
    for posting in postings {
        if let Some(cand) = candidate(model, window, posting, options.require_target)? {
            cands.push(cand);
        }
    }
    cands.sort_by(|a, b| {
        a.dist2
            .total_cmp(&b.dist2)
            .then(a.series_id.cmp(&b.series_id))
            .then(a.end_ts.cmp(&b.end_ts))
    });
    cands.truncate(options.k);
    Ok(cands)
}

pub(super) fn candidate(
    model: &EnnTs,
    window: &[f64],
    posting: Posting,
    require_target: bool,
) -> Result<Option<Candidate>, ENNError> {
    let target = match target_of(model, posting) {
        Some(row) => row,
        None if require_target => return Ok(None),
        None => vec![f64::NAN; model.metrics],
    };
    let scale = metric::latest_scale(model)?;
    let dist2 = metric::dist2(window, &window_of(model, posting)?, scale)?;
    Ok(Some(Candidate {
        series_id: posting.series_id,
        end_ts: posting.end_ts,
        dist2,
        target,
    }))
}

fn exact_postings(model: &EnnTs, options: &QueryOptions) -> Result<Vec<Posting>, ENNError> {
    let max_end = max_end_ts(model, options)?;
    Ok(model
        .windows
        .iter()
        .copied()
        .filter(|p| p.end_ts <= max_end)
        .collect())
}

fn lsh_postings(
    model: &EnnTs,
    window: &[f64],
    options: &QueryOptions,
) -> Result<Vec<Posting>, ENNError> {
    let max_end = max_end_ts(model, options)?;
    let lsh = model.lsh.as_ref().ok_or_else(|| {
        ENNError::InvalidParameter("locality-sensitive hash family is unset".into())
    })?;
    let mut seen = HashSet::new();
    let mut out = Vec::new();
    for scale in &model.scale_gens {
        let keys = keys_for_scale(lsh, model.config.num_tables, window, scale)?;
        for (table, base) in keys.iter().enumerate() {
            for key in probed_keys(base, model.config.probe) {
                let Some(bucket) = model.buckets[table].get(&key) else {
                    continue;
                };
                let end = bucket.partition_point(|p| p.end_ts <= max_end);
                for posting in &bucket[..end] {
                    if seen.insert((posting.series_id, posting.end_pos)) {
                        out.push(*posting);
                    }
                }
            }
        }
    }
    Ok(out)
}

pub(super) fn max_end_ts(model: &EnnTs, options: &QueryOptions) -> Result<i64, ENNError> {
    if !options.require_target {
        return Ok(options.cutoff);
    }
    let step = model
        .config
        .horizon
        .checked_mul(model.config.cadence)
        .ok_or_else(|| ENNError::InvalidParameter("horizon overflow".into()))?;
    options
        .cutoff
        .checked_sub(step)
        .ok_or_else(|| ENNError::InvalidParameter("cutoff overflow".into()))
}

fn pack(ranked: &[Vec<Candidate>], k: usize, metrics: usize) -> SearchBatch {
    let n = ranked.len();
    let mut series_id = Array2::from_elem((n, k), -1i64);
    let mut end_timestamp = Array2::from_elem((n, k), -1i64);
    let mut distance = Array2::from_elem((n, k), f64::INFINITY);
    let mut target = Array3::from_elem((n, k, metrics), f64::NAN);
    for (i, row) in ranked.iter().enumerate() {
        for (j, cand) in row.iter().enumerate() {
            series_id[[i, j]] = cand.series_id as i64;
            end_timestamp[[i, j]] = cand.end_ts;
            distance[[i, j]] = cand.dist2.max(0.0).sqrt();
            for m in 0..metrics.min(cand.target.len()) {
                target[[i, j, m]] = cand.target[m];
            }
        }
    }
    SearchBatch {
        series_id,
        end_timestamp,
        distance,
        target,
    }
}
