use crate::error::ENNError;

use super::EnnTs;

#[derive(Clone, Copy, Debug)]
pub(super) struct Posting {
    pub series_id: u64,
    pub end_pos: usize,
    pub end_ts: i64,
}

#[derive(Debug, Default)]
pub(super) struct SeriesBuf {
    pub ts: Vec<i64>,
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub known: Vec<u8>,
}

impl SeriesBuf {
    pub(super) fn push_x(&mut self, ts: i64, row: &[f64], metrics: usize) {
        self.ts.push(ts);
        self.x.extend_from_slice(row);
        if metrics > 0 {
            self.y.extend(std::iter::repeat_n(0.0, metrics));
            self.known.push(0);
        }
    }

    pub(super) fn pos_of(&self, ts: i64) -> Option<usize> {
        self.ts.binary_search(&ts).ok()
    }

    pub(super) fn write_y(&mut self, pos: usize, row: &[f64]) -> Result<(), ENNError> {
        if pos >= self.ts.len() || self.known.get(pos).copied() == Some(1) {
            return Err(ENNError::InvalidParameter(
                "target timestamp is missing or already set".into(),
            ));
        }
        if row.iter().any(|v| !v.is_finite()) {
            return Err(ENNError::InvalidParameter("targets must be finite".into()));
        }
        let start = pos * row.len();
        for (m, v) in row.iter().enumerate() {
            self.y[start + m] = *v;
        }
        self.known[pos] = 1;
        Ok(())
    }

    pub(super) fn y_row(&self, pos: usize) -> Option<Vec<f64>> {
        if self.known.get(pos).copied() != Some(1) {
            return None;
        }
        let metrics = self.y.len() / self.ts.len().max(1);
        let start = pos * metrics;
        Some(self.y[start..start + metrics].to_vec())
    }

    pub(super) fn window(
        &self,
        end_pos: usize,
        lookback: usize,
        dim: usize,
    ) -> Result<Vec<f64>, ENNError> {
        if lookback == 0 || end_pos + 1 < lookback || self.x.len() < self.ts.len() * dim {
            return Err(ENNError::InvalidParameter(
                "stored window is incomplete".into(),
            ));
        }
        let mut out = Vec::with_capacity(lookback * dim);
        let start_pos = end_pos + 1 - lookback;
        for pos in start_pos..=end_pos {
            let start = pos * dim;
            out.extend_from_slice(&self.x[start..start + dim]);
        }
        Ok(out)
    }
}

pub(super) fn fill_window(
    series: &SeriesBuf,
    end_pos: usize,
    lookback: usize,
    cadence: i64,
    dim: usize,
    out: &mut [f64],
) -> bool {
    if lookback == 0 || end_pos + 1 < lookback || out.len() != lookback * dim {
        return false;
    }
    if series.x.len() < series.ts.len() * dim {
        return false;
    }
    let end_ts = series.ts[end_pos];
    for i in 0..lookback {
        let pos = end_pos + 1 - lookback + i;
        let Ok(age) = i64::try_from(lookback - 1 - i) else {
            return false;
        };
        let Some(step) = cadence.checked_mul(age) else {
            return false;
        };
        let Some(expected) = end_ts.checked_sub(step) else {
            return false;
        };
        if series.ts[pos] != expected {
            return false;
        }
    }
    let start_pos = end_pos + 1 - lookback;
    for (i, pos) in (start_pos..=end_pos).enumerate() {
        let start = pos * dim;
        out[i * dim..(i + 1) * dim].copy_from_slice(&series.x[start..start + dim]);
    }
    true
}

pub(super) fn write_target(model: &EnnTs, posting: Posting, buf: &mut Vec<f64>) -> bool {
    let Some(step) = model.config.horizon.checked_mul(model.config.cadence) else {
        return false;
    };
    let Some(target_ts) = posting.end_ts.checked_add(step) else {
        return false;
    };
    let Some(series) = model.series.get(&posting.series_id) else {
        return false;
    };
    let Some(pos) = series.pos_of(target_ts) else {
        return false;
    };
    if series.known.get(pos).copied() != Some(1) || series.ts.is_empty() {
        return false;
    }
    let metrics = series.y.len() / series.ts.len();
    if metrics == 0 || pos * metrics + metrics > series.y.len() {
        return false;
    }
    let start = pos * metrics;
    buf.clear();
    buf.extend_from_slice(&series.y[start..start + metrics]);
    true
}

pub(super) fn has_target(model: &EnnTs, posting: Posting) -> bool {
    let Some(step) = model.config.horizon.checked_mul(model.config.cadence) else {
        return false;
    };
    let Some(target_ts) = posting.end_ts.checked_add(step) else {
        return false;
    };
    let Some(series) = model.series.get(&posting.series_id) else {
        return false;
    };
    let Some(pos) = series.pos_of(target_ts) else {
        return false;
    };
    series.known.get(pos).copied() == Some(1)
}

pub(super) fn target_of(model: &EnnTs, posting: Posting) -> Option<Vec<f64>> {
    let step = model.config.horizon.checked_mul(model.config.cadence)?;
    let target_ts = posting.end_ts.checked_add(step)?;
    let series = model.series.get(&posting.series_id)?;
    let pos = series.pos_of(target_ts)?;
    series.y_row(pos)
}

pub(super) fn insert_bucket(bucket: &mut Vec<Posting>, posting: Posting) {
    if bucket
        .last()
        .map(|p| p.end_ts <= posting.end_ts)
        .unwrap_or(true)
    {
        bucket.push(posting);
        return;
    }
    let idx = bucket.partition_point(|p| p.end_ts <= posting.end_ts);
    bucket.insert(idx, posting);
}

pub(super) fn remove_bucket(bucket: &mut Vec<Posting>, posting: Posting) -> Result<(), ENNError> {
    let start = bucket.partition_point(|p| p.end_ts < posting.end_ts);
    let end = bucket.partition_point(|p| p.end_ts <= posting.end_ts);
    let Some(offset) = bucket[start..end]
        .iter()
        .position(|p| p.series_id == posting.series_id && p.end_pos == posting.end_pos)
    else {
        return Err(ENNError::InvalidParameter(
            "metric rehash missed a stored posting".into(),
        ));
    };
    bucket.remove(start + offset);
    Ok(())
}

pub(super) fn window_of(model: &EnnTs, posting: Posting) -> Result<Vec<f64>, ENNError> {
    let dim = model.dim.ok_or_else(|| {
        ENNError::InvalidParameter("add observations before reconstructing a window".into())
    })?;
    let series = model.series.get(&posting.series_id).ok_or_else(|| {
        ENNError::InvalidParameter(format!("unknown series {}", posting.series_id))
    })?;
    series.window(posting.end_pos, model.lookback, dim)
}
