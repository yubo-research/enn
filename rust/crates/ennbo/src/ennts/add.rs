use std::collections::HashMap;

use ndarray::ArrayView2;

use crate::error::ENNError;

use super::lsh::LshFamily;
use super::metric::{self, MORPH_PER_WINDOW};
use super::store::{complete_window, insert_bucket, target_of, Posting};
use super::EnnTs;

pub(super) fn auto_timestamps(
    model: &EnnTs,
    series_id: u64,
    n: usize,
) -> Result<Vec<i64>, ENNError> {
    let start = next_timestamp(model, series_id)?;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let step = model
            .config
            .cadence
            .checked_mul(i64::try_from(i).map_err(|_| overflow())?)
            .ok_or_else(overflow)?;
        out.push(start.checked_add(step).ok_or_else(overflow)?);
    }
    Ok(out)
}

pub(super) fn add(
    model: &mut EnnTs,
    series_id: u64,
    timestamps: &[i64],
    x: ArrayView2<f64>,
    y: Option<ArrayView2<f64>>,
) -> Result<(), ENNError> {
    check_series_id(series_id)?;
    if timestamps.is_empty() && x.nrows() == 0 {
        return Ok(());
    }
    check_batch(timestamps, &x, y)?;
    ensure_dim(model, x.ncols())?;
    if let Some(yv) = y {
        adopt_metrics(model, yv.ncols())?;
    }
    let start = append_rows(model, series_id, timestamps, &x, y)?;
    index_new_windows(model, series_id, start)
}

pub(super) fn add_y(
    model: &mut EnnTs,
    series_id: u64,
    timestamps: &[i64],
    y: ArrayView2<f64>,
) -> Result<(), ENNError> {
    check_series_id(series_id)?;
    if timestamps.len() != y.nrows() || y.ncols() == 0 {
        return Err(ENNError::InvalidShape {
            expected: vec![timestamps.len(), model.metrics.max(1)],
            got: y.shape().to_vec(),
        });
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(ENNError::InvalidParameter("targets must be finite".into()));
    }
    adopt_metrics(model, y.ncols())?;
    let positions = target_positions(model, series_id, timestamps)?;
    for (row_i, pos) in positions.iter().enumerate() {
        let row = y.row(row_i).to_vec();
        {
            let series = model
                .series
                .get_mut(&series_id)
                .ok_or_else(|| ENNError::InvalidParameter(format!("unknown series {series_id}")))?;
            series.write_y(*pos, &row)?;
        }
        for (m, v) in row.iter().enumerate() {
            model.y_sum[m] += v;
            model.y_sumsq[m] += v * v;
        }
        model.n_y += 1;
    }
    metric::observe_labeled(model, series_id, timestamps)
}

fn target_positions(
    model: &EnnTs,
    series_id: u64,
    timestamps: &[i64],
) -> Result<Vec<usize>, ENNError> {
    let series = model
        .series
        .get(&series_id)
        .ok_or_else(|| ENNError::InvalidParameter(format!("unknown series {series_id}")))?;
    timestamps
        .iter()
        .map(|ts| {
            series.pos_of(*ts).ok_or_else(|| {
                ENNError::InvalidParameter(format!("series {series_id} has no observation at {ts}"))
            })
        })
        .collect()
}

fn overflow() -> ENNError {
    ENNError::InvalidParameter("timestamp overflow".into())
}

fn next_timestamp(model: &EnnTs, series_id: u64) -> Result<i64, ENNError> {
    match model
        .series
        .get(&series_id)
        .and_then(|s| s.ts.last().copied())
    {
        Some(last) => last.checked_add(model.config.cadence).ok_or_else(overflow),
        None => Ok(0),
    }
}

fn check_series_id(series_id: u64) -> Result<(), ENNError> {
    if series_id > i64::MAX as u64 {
        return Err(ENNError::InvalidParameter(
            "series_id must fit in a signed 64-bit integer".into(),
        ));
    }
    Ok(())
}

fn check_batch(
    timestamps: &[i64],
    x: &ArrayView2<f64>,
    y: Option<ArrayView2<f64>>,
) -> Result<(), ENNError> {
    if timestamps.len() != x.nrows() || x.ncols() == 0 {
        return Err(ENNError::InvalidShape {
            expected: vec![timestamps.len(), x.ncols().max(1)],
            got: x.shape().to_vec(),
        });
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err(ENNError::InvalidParameter(
            "observations must be finite".into(),
        ));
    }
    if let Some(yv) = y {
        if yv.nrows() != x.nrows() || yv.ncols() == 0 || yv.iter().any(|v| !v.is_finite()) {
            return Err(ENNError::InvalidParameter(
                "targets must be finite and have one row per observation".into(),
            ));
        }
    }
    Ok(())
}

fn ensure_dim(model: &mut EnnTs, dim: usize) -> Result<(), ENNError> {
    if let Some(existing) = model.dim {
        if existing != dim {
            return Err(ENNError::InvalidParameter(format!(
                "expected dimension {existing}, got {dim}"
            )));
        }
        return Ok(());
    }
    let window_dim = model
        .lookback
        .checked_mul(dim)
        .ok_or_else(|| ENNError::InvalidParameter("window dimension overflow".into()))?;
    model.lsh = Some(LshFamily::create(
        window_dim,
        model.config.num_tables,
        model.config.num_projections,
        model.config.bucket_width,
        model.config.seed,
    )?);
    model.scale_gens = vec![vec![1.0; window_dim]];
    model.buckets = (0..model.config.num_tables)
        .map(|_| HashMap::new())
        .collect();
    model.dim = Some(dim);
    Ok(())
}

fn adopt_metrics(model: &mut EnnTs, metrics: usize) -> Result<(), ENNError> {
    if metrics == 0 {
        return Err(ENNError::InvalidParameter(
            "targets need at least one column".into(),
        ));
    }
    if model.metrics == metrics {
        return Ok(());
    }
    if model.metrics != 0 {
        return Err(ENNError::InvalidParameter(format!(
            "expected {} target columns, got {metrics}",
            model.metrics
        )));
    }
    model.metrics = metrics;
    model.y_sum = vec![0.0; metrics];
    model.y_sumsq = vec![0.0; metrics];
    for series in model.series.values_mut() {
        let n = series.ts.len();
        series.y = vec![0.0; n * metrics];
        series.known = vec![0; n];
    }
    Ok(())
}

fn append_rows(
    model: &mut EnnTs,
    series_id: u64,
    timestamps: &[i64],
    x: &ArrayView2<f64>,
    y: Option<ArrayView2<f64>>,
) -> Result<usize, ENNError> {
    let last = model
        .series
        .get(&series_id)
        .and_then(|s| s.ts.last().copied());
    check_increasing(last, timestamps)?;
    let start = model
        .series
        .get(&series_id)
        .map(|s| s.ts.len())
        .unwrap_or(0);
    let metrics = model.metrics;
    for (i, &ts) in timestamps.iter().enumerate() {
        let row = x.row(i).to_vec();
        {
            let series = model.series.entry(series_id).or_default();
            series.push_x(ts, &row, metrics);
        }
        model.n_obs += 1;
        if let Some(yv) = y {
            record_new_target(model, series_id, &yv.row(i).to_vec())?;
        }
    }
    Ok(start)
}

fn record_new_target(model: &mut EnnTs, series_id: u64, values: &[f64]) -> Result<(), ENNError> {
    {
        let series = model
            .series
            .get_mut(&series_id)
            .ok_or_else(|| ENNError::InvalidParameter(format!("unknown series {series_id}")))?;
        let pos = series.ts.len() - 1;
        series.write_y(pos, values)?;
    }
    for (m, v) in values.iter().enumerate() {
        model.y_sum[m] += v;
        model.y_sumsq[m] += v * v;
    }
    model.n_y += 1;
    Ok(())
}

fn check_increasing(last: Option<i64>, timestamps: &[i64]) -> Result<(), ENNError> {
    let mut prev = last;
    for &ts in timestamps {
        if let Some(p) = prev {
            if ts <= p {
                return Err(ENNError::InvalidParameter(format!(
                    "timestamps must increase within a series, got {ts} after {p}"
                )));
            }
        }
        prev = Some(ts);
    }
    Ok(())
}

fn index_new_windows(model: &mut EnnTs, series_id: u64, start: usize) -> Result<(), ENNError> {
    let pending = pending_windows(model, series_id, start)?;
    for (pos, end_ts, window) in pending {
        let probe = Posting {
            series_id,
            end_pos: pos,
            end_ts,
        };
        let y = target_of(model, probe);
        if let Some(row) = &y {
            metric::note_window(model, &window, row)?;
        }
        insert_posting(model, series_id, pos, end_ts, &window, y.is_some())?;
    }
    Ok(())
}

fn pending_windows(
    model: &EnnTs,
    series_id: u64,
    start: usize,
) -> Result<Vec<(usize, i64, Vec<f64>)>, ENNError> {
    let dim = model
        .dim
        .ok_or_else(|| ENNError::InvalidParameter("dimension is unset".into()))?;
    let series = model
        .series
        .get(&series_id)
        .ok_or_else(|| ENNError::InvalidParameter(format!("unknown series {series_id}")))?;
    let mut pending = Vec::new();
    for pos in start..series.ts.len() {
        if let Some(window) =
            complete_window(series, pos, model.lookback, model.config.cadence, dim)
        {
            pending.push((pos, series.ts[pos], window));
        }
    }
    Ok(pending)
}

fn insert_posting(
    model: &mut EnnTs,
    series_id: u64,
    pos: usize,
    end_ts: i64,
    window: &[f64],
    observed: bool,
) -> Result<(), ENNError> {
    let posting = Posting {
        series_id,
        end_pos: pos,
        end_ts,
    };
    let gen = metric::current_gen(model)?;
    let keys = metric::keys_for_gen(model, window, gen)?;
    model.windows.push(posting);
    model.window_rows.extend_from_slice(window);
    model.window_scale_gen.push(gen);
    model.metric_observed.push(u8::from(observed));
    for (table, key) in keys.into_iter().enumerate() {
        let bucket = model.buckets[table].entry(key).or_default();
        insert_bucket(bucket, posting);
    }
    metric::advance(model, MORPH_PER_WINDOW)
}
