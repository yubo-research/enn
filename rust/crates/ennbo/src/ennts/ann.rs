use std::cmp::Ordering;
use std::sync::Once;

use faiss::index::IndexImpl;
use faiss::{index_factory, Index, MetricType};
use ndarray::ArrayView2;
use rayon::prelude::*;

use super::metric;
use super::search::{self, Candidate};
use super::store;
use super::{EnnTs, QueryOptions};
use crate::error::ENNError;

pub(super) const TREE_MIN_WINDOWS: usize = 2_048;
const HNSW_SPEC: &str = "HNSW16";

pub(super) struct Cache {
    index: IndexImpl,
    scale_epoch: u64,
    n_windows: usize,
}

pub(super) fn sync(model: &EnnTs) -> Result<(), ENNError> {
    if model.windows.len() < TREE_MIN_WINDOWS {
        return Ok(());
    }
    let scale = metric::latest_scale(model)?.to_vec();
    let epoch = model.scale_epoch;
    let n = model.windows.len();
    let mut guard = lock(model)?;
    if let Some(cache) = guard.as_mut() {
        if cache.scale_epoch == epoch && cache.n_windows == n {
            return Ok(());
        }
        if cache.scale_epoch == epoch && cache.n_windows < n {
            let coords = scaled_range(model, cache.n_windows, n, &scale)?;
            cache.index.add(&coords).map_err(faiss_err)?;
            cache.n_windows = n;
            return Ok(());
        }
    }
    let coords = scaled_range(model, 0, n, &scale)?;
    enable_omp();
    let mut index =
        index_factory(scale.len() as u32, HNSW_SPEC, MetricType::L2).map_err(faiss_err)?;
    index.add(&coords).map_err(faiss_err)?;
    *guard = Some(Cache {
        index,
        scale_epoch: epoch,
        n_windows: n,
    });
    Ok(())
}

pub(super) fn rank_large(
    model: &EnnTs,
    queries: ArrayView2<f64>,
    options: &QueryOptions,
) -> Result<Vec<Vec<Candidate>>, ENNError> {
    sync(model)?;
    let scale = metric::latest_scale(model)?;
    let width = scale.len();
    let max_end = search::max_end_ts(model, options)?;
    let n = model.windows.len();
    let raw: Vec<Vec<f64>> = queries.rows().into_iter().map(|row| row.to_vec()).collect();
    let mut flat_q = Vec::with_capacity(raw.len().saturating_mul(width));
    for row in &raw {
        flat_q.extend(scale_query(row, scale));
    }
    let labeled = target_flags(model, options);
    let mut pool = options.k.saturating_mul(16).max(128).min(n.max(1));
    loop {
        let labels = search_labels(model, &flat_q, pool)?;
        let ranked = raw
            .par_iter()
            .enumerate()
            .map(|(qi, row)| {
                take_neighbors(
                    model,
                    row,
                    &labels[qi * pool..(qi + 1) * pool],
                    scale,
                    &labeled,
                    options,
                    max_end,
                )
            })
            .collect::<Result<Vec<_>, ENNError>>()?;
        let short = ranked.iter().any(|row| row.len() < options.k);
        if !short || pool == n {
            return Ok(ranked);
        }
        let next = pool.saturating_mul(4).min(n);
        if next == pool {
            return Ok(ranked);
        }
        pool = next;
    }
}

struct Near {
    dist: f64,
    series_id: u64,
    end_ts: i64,
    index: u32,
}

fn nearer(left: &Near, right: &Near) -> Ordering {
    left.dist
        .total_cmp(&right.dist)
        .then(left.series_id.cmp(&right.series_id))
        .then(left.end_ts.cmp(&right.end_ts))
        .then(left.index.cmp(&right.index))
}

fn target_flags(model: &EnnTs, options: &QueryOptions) -> Vec<u8> {
    if !options.require_target {
        return vec![1; model.windows.len()];
    }
    model
        .windows
        .iter()
        .map(|posting| u8::from(store::has_target(model, *posting)))
        .collect()
}

fn take_neighbors(
    model: &EnnTs,
    window: &[f64],
    labels: &[i64],
    scale: &[f64],
    labeled: &[u8],
    options: &QueryOptions,
    max_end: i64,
) -> Result<Vec<Candidate>, ENNError> {
    let width = scale.len();
    let mut best: Vec<Near> = Vec::with_capacity(options.k);
    for &label in labels {
        if label < 0 {
            continue;
        }
        let Ok(index) = usize::try_from(label) else {
            continue;
        };
        let Some(posting) = model.windows.get(index).copied() else {
            continue;
        };
        if posting.end_ts > max_end || labeled.get(index).copied() != Some(1) {
            continue;
        }
        let offset = index
            .checked_mul(width)
            .ok_or_else(|| ENNError::InvalidParameter("window index overflow".into()))?;
        let stored = model
            .window_rows
            .get(offset..offset + width)
            .ok_or_else(|| {
                ENNError::InvalidParameter("stored window is shorter than the metric scale".into())
            })?;
        consider(
            &mut best,
            options.k,
            Near {
                dist: scaled_dist2(window, stored, scale),
                series_id: posting.series_id,
                end_ts: posting.end_ts,
                index: index as u32,
            },
        );
    }
    best.sort_by(nearer);
    let mut cands = Vec::with_capacity(best.len());
    for near in best {
        let posting = model.windows[near.index as usize];
        let target = match store::target_of(model, posting) {
            Some(row) => row,
            None if options.require_target => continue,
            None => vec![f64::NAN; model.metrics],
        };
        cands.push(Candidate {
            series_id: near.series_id,
            end_ts: near.end_ts,
            dist2: near.dist,
            target,
        });
    }
    Ok(cands)
}

fn consider(best: &mut Vec<Near>, k: usize, near: Near) {
    if best.len() < k {
        best.push(near);
        return;
    }
    let worst = best
        .iter()
        .enumerate()
        .max_by(|left, right| nearer(left.1, right.1))
        .map(|(index, _)| index);
    if let Some(index) = worst {
        if nearer(&near, &best[index]).is_lt() {
            best[index] = near;
        }
    }
}

fn scaled_dist2(left: &[f64], right: &[f64], scale: &[f64]) -> f64 {
    let mut sum = 0.0;
    for ((a, b), factor) in left.iter().zip(right).zip(scale) {
        let delta = (a - b) / factor;
        sum += delta * delta;
    }
    sum.max(0.0)
}

fn search_labels(model: &EnnTs, queries: &[f32], pool: usize) -> Result<Vec<i64>, ENNError> {
    let mut guard = lock(model)?;
    let cache = guard
        .as_mut()
        .ok_or_else(|| ENNError::InvalidParameter("neighbor index was not built".into()))?;
    let found = cache.index.search(queries, pool).map_err(faiss_err)?;
    Ok(found.labels.iter().map(|label| label.to_native()).collect())
}

fn scaled_range(
    model: &EnnTs,
    start: usize,
    end: usize,
    scale: &[f64],
) -> Result<Vec<f32>, ENNError> {
    let width = scale.len();
    if model.window_rows.len() != model.windows.len().saturating_mul(width) {
        return Err(ENNError::InvalidParameter(
            "stored windows do not match the metric scale".into(),
        ));
    }
    let mut coords = Vec::with_capacity((end - start).saturating_mul(width));
    for index in start..end {
        let offset = index
            .checked_mul(width)
            .ok_or_else(|| ENNError::InvalidParameter("window index overflow".into()))?;
        let row = model
            .window_rows
            .get(offset..offset + width)
            .ok_or_else(|| {
                ENNError::InvalidParameter("stored window is shorter than the metric scale".into())
            })?;
        for (value, factor) in row.iter().zip(scale.iter()) {
            let scaled = value / factor;
            if !scaled.is_finite() {
                return Err(ENNError::InvalidParameter(
                    "scaled window is non-finite".into(),
                ));
            }
            coords.push(scaled as f32);
        }
    }
    Ok(coords)
}

fn scale_query(row: &[f64], scale: &[f64]) -> Vec<f32> {
    row.iter()
        .zip(scale.iter())
        .map(|(value, factor)| (*value / *factor) as f32)
        .collect()
}

fn lock(model: &EnnTs) -> Result<std::sync::MutexGuard<'_, Option<Cache>>, ENNError> {
    model
        .ann
        .lock()
        .map_err(|_| ENNError::InvalidParameter("neighbor index lock was poisoned".into()))
}

fn faiss_err(err: faiss::error::Error) -> ENNError {
    ENNError::InvalidParameter(format!("neighbor index failed: {err}"))
}

fn enable_omp() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        extern "C" {
            fn omp_set_num_threads(num_threads: i32);
        }
        let threads = std::thread::available_parallelism()
            .map(|count| count.get())
            .unwrap_or(1)
            .clamp(1, 255) as i32;
        unsafe { omp_set_num_threads(threads) }
    });
}

#[cfg(test)]
mod tests {
    use super::super::{EnnTs, EnnTsConfig, QueryOptions};
    use super::*;
    use ndarray::Array2;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    fn window_query(x: &Array2<f64>, end: usize, lookback: usize) -> Array2<f64> {
        let dim = x.ncols();
        let mut query = Array2::zeros((1, lookback * dim));
        for lag in 0..lookback {
            let row = end + 1 - lookback + lag;
            for column in 0..dim {
                query[[0, lag * dim + column]] = x[[row, column]];
            }
        }
        query
    }

    fn assert_same_neighbors(x: &Array2<f64>, n: usize, end: usize, k: usize) {
        let mut model = EnnTs::new(EnnTsConfig::new(5)).unwrap();
        let ts: Vec<i64> = (0..n as i64).collect();
        let y = Array2::from_shape_fn((n, 1), |(i, _)| i as f64);
        model.add(0, &ts, x.view(), Some(y.view())).unwrap();
        assert!(model.num_windows() > TREE_MIN_WINDOWS);
        let query = window_query(x, end, 5);
        let opts = QueryOptions {
            cutoff: n as i64,
            k,
            require_target: true,
        };
        let approx = model.search(query.view(), &opts).unwrap();
        let exact = model.search_exact(query.view(), &opts).unwrap();
        assert_eq!(approx.end_timestamp, exact.end_timestamp);
        assert_eq!(approx.series_id, exact.series_id);
    }

    #[test]
    fn large_model_search_matches_exact_neighbors() {
        let n = 2_200;
        let x = Array2::from_shape_fn((n, 3), |(i, j)| {
            ((i as f64) * 0.013 + (j as f64) * 0.37).sin()
        });
        assert_same_neighbors(&x, n, 40, 4);
    }

    #[test]
    fn large_gaussian_model_finds_the_same_nearest_window() {
        let n = 2_200;
        let mut rng = StdRng::seed_from_u64(19);
        let x = Array2::from_shape_fn((n, 10), |(_, _)| rng.gen::<f64>() - 0.5);
        assert_same_neighbors(&x, n, 80, 1);
    }
}
