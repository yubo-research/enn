use crate::error::ENNError;
use crate::metric_auto::{AutoMetric, MetricLearning};

use super::ann::hash_tables_serve;
use super::lsh::LshFamily;
use super::store::{insert_bucket, remove_bucket, target_of, window_of, Posting};
use super::EnnTs;

pub(super) const MORPH_PER_WINDOW: usize = 6;

pub(super) fn note_window(model: &mut EnnTs, window: &[f64], y: &[f64]) -> Result<(), ENNError> {
    if model.config.metric_learning != MetricLearning::Auto {
        return Ok(());
    }
    if model.metric.is_none() {
        model.metric = Some(AutoMetric::new(
            window.len(),
            y.len(),
            Vec::new(),
            model.config.seed,
        )?);
    }
    let update = {
        let metric = model
            .metric
            .as_mut()
            .ok_or_else(|| ENNError::InvalidParameter("metric learner is unset".into()))?;
        metric.observe(window, y, 1)?
    };
    if let Some(update) = update {
        adopt(model, update.x_scale)?;
    }
    Ok(())
}

pub(super) fn observe_labeled(
    model: &mut EnnTs,
    series_id: u64,
    timestamps: &[i64],
) -> Result<(), ENNError> {
    if model.config.metric_learning != MetricLearning::Auto {
        return Ok(());
    }
    let step = model
        .config
        .horizon
        .checked_mul(model.config.cadence)
        .ok_or_else(|| ENNError::InvalidParameter("horizon overflow".into()))?;
    for &ts in timestamps {
        let Some(end_ts) = ts.checked_sub(step) else {
            continue;
        };
        let Some(index) = model
            .windows
            .iter()
            .position(|p| p.series_id == series_id && p.end_ts == end_ts)
        else {
            continue;
        };
        if model.metric_observed[index] != 0 {
            continue;
        }
        let posting = model.windows[index];
        let window = window_of(model, posting)?;
        let Some(y) = target_of(model, posting) else {
            continue;
        };
        note_window(model, &window, &y)?;
        model.metric_observed[index] = 1;
        advance(model, MORPH_PER_WINDOW)?;
    }
    Ok(())
}

pub(super) fn current_gen(model: &EnnTs) -> Result<u32, ENNError> {
    let n = model.scale_gens.len();
    if n == 0 {
        return Err(ENNError::InvalidParameter("metric scale is unset".into()));
    }
    u32::try_from(n - 1).map_err(|_| ENNError::InvalidParameter("too many metric scales".into()))
}

pub(super) fn keys_for_gen(
    model: &EnnTs,
    window: &[f64],
    gen: u32,
) -> Result<Vec<Vec<i32>>, ENNError> {
    let scale = model
        .scale_gens
        .get(gen as usize)
        .ok_or_else(|| ENNError::InvalidParameter("metric scale generation is missing".into()))?;
    let lsh = model.lsh.as_ref().ok_or_else(|| {
        ENNError::InvalidParameter("locality-sensitive hash family is unset".into())
    })?;
    keys_for_scale(lsh, model.config.num_tables, window, scale)
}

pub(super) fn keys_for_scale(
    lsh: &LshFamily,
    num_tables: usize,
    window: &[f64],
    scale: &[f64],
) -> Result<Vec<Vec<i32>>, ENNError> {
    if window.len() != scale.len() {
        return Err(ENNError::InvalidParameter(
            "metric scale does not match the window".into(),
        ));
    }
    let scaled: Vec<f64> = window.iter().zip(scale).map(|(v, s)| v / s).collect();
    let mut keys = Vec::with_capacity(num_tables);
    for table in 0..num_tables {
        keys.push(lsh.hash_table(table, &scaled)?);
    }
    Ok(keys)
}

pub(super) fn latest_scale(model: &EnnTs) -> Result<&[f64], ENNError> {
    model
        .scale_gens
        .last()
        .map(Vec::as_slice)
        .ok_or_else(|| ENNError::InvalidParameter("metric scale is unset".into()))
}

pub(super) fn dist2(left: &[f64], right: &[f64], scale: &[f64]) -> Result<f64, ENNError> {
    if left.len() != right.len() || left.len() != scale.len() {
        return Err(ENNError::InvalidParameter(
            "window and metric scale lengths differ".into(),
        ));
    }
    let mut sum = 0.0;
    for ((a, b), s) in left.iter().zip(right).zip(scale) {
        let delta = (a - b) / s;
        sum += delta * delta;
    }
    Ok(sum.max(0.0))
}

pub(super) fn release_hash_tables(model: &mut EnnTs) {
    model.morph_queue.clear();
    compact(model);
}

pub(super) fn advance(model: &mut EnnTs, steps: usize) -> Result<(), ENNError> {
    if !hash_tables_serve(model.windows.len()) {
        release_hash_tables(model);
        return Ok(());
    }
    let mut left = steps;
    while left > 0 {
        let Some(index) = model.morph_queue.pop_front() else {
            break;
        };
        rehash(model, index)?;
        left -= 1;
    }
    compact(model);
    Ok(())
}

fn adopt(model: &mut EnnTs, x_scale: Vec<f64>) -> Result<(), ENNError> {
    let width = model
        .scale_gens
        .first()
        .map(Vec::len)
        .ok_or_else(|| ENNError::InvalidParameter("metric scale is unset".into()))?;
    if x_scale.len() != width || !x_scale.iter().all(|s| s.is_finite() && *s > 0.0) {
        return Err(ENNError::InvalidParameter(
            "metric scale must be finite, positive, and match the window".into(),
        ));
    }
    model.scale_gens.push(x_scale);
    model.scale_epoch = model.scale_epoch.saturating_add(1);
    if !hash_tables_serve(model.windows.len()) {
        release_hash_tables(model);
        return Ok(());
    }
    let latest = current_gen(model)?;
    model.morph_queue.clear();
    for index in 0..model.windows.len() {
        if model.window_scale_gen[index] != latest {
            model.morph_queue.push_back(index);
        }
    }
    Ok(())
}

fn rehash(model: &mut EnnTs, index: usize) -> Result<(), ENNError> {
    let latest = current_gen(model)?;
    let old_gen = model.window_scale_gen[index];
    if old_gen == latest {
        return Ok(());
    }
    let posting = model.windows[index];
    let window = window_of(model, posting)?;
    let old_keys = keys_for_gen(model, &window, old_gen)?;
    detach(model, posting, &old_keys)?;
    model.window_scale_gen[index] = latest;
    let new_keys = keys_for_gen(model, &window, latest)?;
    attach(model, posting, &new_keys);
    Ok(())
}

fn detach(model: &mut EnnTs, posting: Posting, keys: &[Vec<i32>]) -> Result<(), ENNError> {
    for (table, key) in keys.iter().enumerate() {
        let empty = {
            let bucket = model.buckets[table].get_mut(key).ok_or_else(|| {
                ENNError::InvalidParameter("metric rehash missed a bucket".into())
            })?;
            remove_bucket(bucket, posting)?;
            bucket.is_empty()
        };
        if empty {
            model.buckets[table].remove(key);
        }
    }
    Ok(())
}

fn attach(model: &mut EnnTs, posting: Posting, keys: &[Vec<i32>]) {
    for (table, key) in keys.iter().enumerate() {
        let bucket = model.buckets[table].entry(key.clone()).or_default();
        insert_bucket(bucket, posting);
    }
}

fn compact(model: &mut EnnTs) {
    if !model.morph_queue.is_empty() || model.scale_gens.len() <= 1 {
        return;
    }
    let Some(scale) = model.scale_gens.pop() else {
        return;
    };
    model.scale_gens.clear();
    model.scale_gens.push(scale);
    model.window_scale_gen.fill(0);
}

#[cfg(test)]
impl EnnTs {
    pub(super) fn adopt_scale_for_test(&mut self, scale: Vec<f64>) -> Result<(), ENNError> {
        adopt(self, scale)
    }
}
