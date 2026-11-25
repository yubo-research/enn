use ndarray::{Array1, Array2, ArrayView2};

use super::EpistemicNearestNeighbors;
use crate::backend::TrainRowsAtResult;
use crate::error::ENNError;

pub struct EnnIndexAccess<'a> {
    pub(super) model: &'a EpistemicNearestNeighbors,
}

impl<'a> EnnIndexAccess<'a> {
    pub(crate) fn new(model: &'a EpistemicNearestNeighbors) -> Self {
        Self { model }
    }

    pub fn ensure_sync(&self) -> Result<(), ENNError> {
        self.model
            .backend
            .ensure_index_sync(self.model.scale_x, &self.model.x_scale)
    }

    pub fn memory_bytes(&self) -> Result<usize, ENNError> {
        self.model.backend.index_memory_bytes()
    }

    pub fn occupancy(&self) -> Result<Option<crate::o1::O1Occupancy>, ENNError> {
        self.model.backend.with_o1(|s| s.occupancy())
    }

    pub fn candidates(&self, x: &ArrayView2<f64>, search_k: usize) -> Result<Option<Vec<Vec<u32>>>, ENNError> {
        self.model.backend.with_o1(|s| s.candidates(x, search_k))
    }

    pub fn node_io(&self) -> Result<Option<(u64, u64)>, ENNError> {
        self.model.backend.with_o1(|s| Ok(s.node_io())).map(Option::flatten)
    }

    pub fn set_probe_budget(&self, max_leaves: usize, max_candidates: usize) -> Result<(), ENNError> {
        self.model
            .backend
            .with_o1(|s| s.set_probe_budget(max_leaves, max_candidates))?
            .ok_or_else(|| ENNError::InvalidParameter("OLOGN has no probe budget".into()))
    }

    pub fn set_io_width(&self, width: usize, slow_ns: u64) -> Result<(), ENNError> {
        self.model
            .backend
            .with_o1(|s| s.set_io_width(width, slow_ns))?
            .ok_or_else(|| ENNError::InvalidParameter("OLOGN has no I/O width".into()))
    }

    pub fn set_split_rule(&self, rule: crate::o1::SplitRule) -> Result<(), ENNError> {
        self.model
            .backend
            .with_o1(|s| s.set_split_rule(rule))?
            .ok_or_else(|| ENNError::InvalidParameter("OLOGN has no split rule".into()))
    }

    pub fn set_bound(&self, bound: f64) -> Result<(), ENNError> {
        self.model
            .backend
            .with_o1(|s| s.set_bound(bound))?
            .ok_or_else(|| ENNError::InvalidParameter("OLOGN has no search bound".into()))
    }

    pub fn set_repair(&self, repair_rows: usize, stale_drift: f64, prefer_fresh: bool) -> Result<(), ENNError> {
        self.model
            .backend
            .with_o1(|s| s.set_repair(repair_rows, stale_drift, prefer_fresh))?
            .ok_or_else(|| ENNError::InvalidParameter("OLOGN has no repair budget".into()))
    }

    pub fn maintain(&self, max_rows: usize) -> Result<usize, ENNError> {
        Ok(self.model.backend.with_o1(|s| s.maintain_index(max_rows))?.unwrap_or(0))
    }

    pub fn compact(&self) -> Result<usize, ENNError> {
        Ok(self.model.backend.with_o1(|s| s.compact_index())?.unwrap_or(0))
    }

    pub fn set_cache_budget(&self, budget_bytes: usize) -> Result<(), ENNError> {
        self.model
            .backend
            .with_o1(|s| {
                s.set_cache_budget(budget_bytes);
                Ok(())
            })?
            .ok_or_else(|| ENNError::InvalidParameter("OLOGN has no object cache".into()))
    }

    pub fn set_cache_split(&self, node_bytes: usize, row_bytes: usize) -> Result<(), ENNError> {
        self.model
            .backend
            .with_o1(|s| {
                s.set_cache_split(node_bytes, row_bytes);
                Ok(())
            })?
            .ok_or_else(|| ENNError::InvalidParameter("OLOGN has no object cache".into()))
    }

    pub fn cache_stats(&self) -> Result<Option<crate::o1::CacheStats>, ENNError> {
        self.model.backend.with_o1(|s| Ok(s.cache_stats())).map(Option::flatten)
    }

    pub fn set_profile(&self, enabled: bool) -> Result<(), ENNError> {
        self.model.backend.with_o1(|s| {
            s.set_profile(enabled);
            Ok(())
        })?;
        Ok(())
    }

    pub fn take_profile(&self) -> Result<Option<crate::o1::ProfileStats>, ENNError> {
        self.model.backend.with_o1(|s| Ok(s.take_profile()))
    }

    pub fn release_observation_pages(&self) -> Result<(), ENNError> {
        self.model.backend.release_observation_pages()
    }

    pub fn len(&self) -> usize {
        self.model.backend.index_len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn neighbor_distances_and_indices(
        &self,
        x: &ArrayView2<f64>,
        search_k: i32,
        exclude_nearest: bool,
    ) -> Result<(Array2<f64>, Array2<i64>), ENNError> {
        let n_obs = self.model.num_obs();
        if !exclude_nearest || n_obs == 0 || search_k <= 0 {
            let (dist, idx) = self.model.backend.search(x, search_k, exclude_nearest)?;
            return Ok(pad_neighbor_cols_to_search_k(
                dist,
                idx,
                search_k.max(0) as usize,
            ));
        }
        let fetch_k = ((search_k as usize) + 1).min(n_obs);
        let (dist, idx) = self.model.backend.search(x, fetch_k as i32, true)?;
        let k_out = (search_k as usize).min(idx.ncols());
        Ok(trim_search_cols(dist, idx, k_out))
    }

    pub fn index_neighbor_distances_and_indices(
        &self,
        x: &ArrayView2<f64>,
        search_k: i32,
        exclude_nearest: bool,
    ) -> Result<(Array2<f64>, Array2<i64>), ENNError> {
        let n_obs = self.model.num_obs();
        if !exclude_nearest || n_obs == 0 || search_k <= 0 {
            return crate::posterior::index_search(self.model, x, search_k, exclude_nearest);
        }
        let fetch_k = ((search_k as usize) + 1).min(n_obs);
        let (dist, idx) = crate::posterior::index_search(self.model, x, fetch_k as i32, true)?;
        let k_out = (search_k as usize).min(idx.ncols());
        Ok(trim_search_cols(dist, idx, k_out))
    }
}

pub struct EnnRowAccess<'a> {
    model: &'a EpistemicNearestNeighbors,
}

impl<'a> EnnRowAccess<'a> {
    pub(crate) fn new(model: &'a EpistemicNearestNeighbors) -> Self {
        Self { model }
    }

    pub fn train_rows_at(&self, indices: &[usize]) -> Result<TrainRowsAtResult, ENNError> {
        self.model.backend.train_rows_at(indices)
    }

    pub fn row_x(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        self.model.backend.row_x(i)
    }

    pub fn row_y(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        self.model.backend.row_y(i)
    }

    pub fn row_yvar(&self, i: usize) -> Result<Option<Array1<f64>>, ENNError> {
        self.model.backend.row_yvar(i)
    }
}

fn pad_neighbor_cols_to_search_k(
    dist2s: Array2<f64>,
    idx: Array2<i64>,
    search_k: usize,
) -> (Array2<f64>, Array2<i64>) {
    use ndarray::{concatenate, s, Axis};
    let k_eff = dist2s.ncols();
    if k_eff >= search_k {
        return (dist2s, idx);
    }
    let n_query = dist2s.nrows();
    let pad_dist = Array2::from_elem((n_query, search_k - k_eff), f64::INFINITY);
    let mut pad_idx = Array2::zeros((n_query, search_k - k_eff));
    if k_eff > 0 {
        let farthest = idx.slice(s![.., k_eff - 1]).to_owned();
        for mut column in pad_idx.columns_mut() {
            column.assign(&farthest);
        }
    }
    (
        concatenate![Axis(1), dist2s.view(), pad_dist.view()],
        concatenate![Axis(1), idx.view(), pad_idx.view()],
    )
}

fn trim_search_cols(
    dist: Array2<f64>,
    idx: Array2<i64>,
    k_out: usize,
) -> (Array2<f64>, Array2<i64>) {
    if k_out == idx.ncols() {
        (dist, idx)
    } else {
        (
            dist.slice_axis(ndarray::Axis(1), ndarray::Slice::from(..k_out))
                .to_owned(),
            idx.slice_axis(ndarray::Axis(1), ndarray::Slice::from(..k_out))
                .to_owned(),
        )
    }
}

impl EpistemicNearestNeighbors {
    pub fn index_access(&self) -> EnnIndexAccess<'_> {
        EnnIndexAccess::new(self)
    }

    pub fn rows(&self) -> EnnRowAccess<'_> {
        EnnRowAccess::new(self)
    }

    pub(crate) fn ensure_index_sync(&self) -> Result<(), ENNError> {
        self.index_access().ensure_sync()
    }
}

#[cfg(test)]
mod access_tests {
    use ndarray::array;

    #[test]
    fn neighbor_distance_helpers() {
        let model = crate::test_helpers::test_model(
            array![[0.0, 0.0], [1.0, 0.0]],
            array![[0.0], [1.0]],
            None,
            false,
        );
        let query = array![[0.1, 0.1]];
        let access = model.index_access();
        let _ = access
            .neighbor_distances_and_indices(&query.view(), 1, false)
            .unwrap();
        let _ = access
            .index_neighbor_distances_and_indices(&query.view(), 1, false)
            .unwrap();
        let has_buckets = access.occupancy().unwrap().is_some();
        assert_eq!(access.candidates(&query.view(), 1).unwrap().is_some(), has_buckets);
        assert!(access.node_io().unwrap().is_none());
        assert!(access.maint_stats().unwrap().is_none());
    }
}
