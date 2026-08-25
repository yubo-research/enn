//! Focused accessors for ENN row and index operations (kiss method-count split).

use ndarray::{Array1, Array2, ArrayView2};

use super::EpistemicNearestNeighbors;
use crate::backend::TrainRowsAtResult;
use crate::error::ENNError;

/// Index search and sync operations on an ENN model.
pub struct EnnIndexAccess<'a> {
    model: &'a EpistemicNearestNeighbors,
}

impl<'a> EnnIndexAccess<'a> {
    pub(crate) fn new(model: &'a EpistemicNearestNeighbors) -> Self {
        Self { model }
    }

    pub fn ensure_sync(&self) -> Result<(), ENNError> {
        self.model
            .backend
            .ensure_index_sync(self.model.scale_x, &self.model.x_scale)?;
        self.model.persist_y_bounds_metadata()
    }

    pub fn memory_bytes(&self) -> Result<usize, ENNError> {
        if !self.model.backend.defer_index_sync_for_search() {
            self.ensure_sync()?;
        }
        self.model.backend.index_memory_bytes()
    }

    pub fn is_stale(&self) -> bool {
        self.model.backend.is_index_stale()
    }

    pub fn release_observation_pages(&self) -> Result<(), ENNError> {
        crate::backend::release_enn_observation_pages(&self.model.backend)
    }

    pub fn len(&self) -> usize {
        self.model.backend.index_len()
    }

    #[doc = "kiss-coverage-off"]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn neighbor_distances_and_indices(
        &self,
        x: &ArrayView2<f64>,
        search_k: i32,
        exclude_nearest: bool,
    ) -> Result<(Array2<f64>, Array2<i64>), ENNError> {
        if !self.model.backend.defer_index_sync_for_search() {
            self.ensure_sync()?;
        }
        let n_obs = self.model.num_obs();
        if !exclude_nearest || n_obs == 0 || search_k <= 0 {
            return self.model.backend.search(x, search_k, exclude_nearest);
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
        let (dist, idx) =
            crate::posterior::index_search(self.model, x, fetch_k as i32, true)?;
        let k_out = (search_k as usize).min(idx.ncols());
        Ok(trim_search_cols(dist, idx, k_out))
    }
}

/// Row gather operations on an ENN model.
pub struct EnnRowAccess<'a> {
    model: &'a EpistemicNearestNeighbors,
}

impl<'a> EnnRowAccess<'a> {
    pub(crate) fn new(model: &'a EpistemicNearestNeighbors) -> Self {
        Self { model }
    }

    pub fn train_rows_at(
        &self,
        indices: &[usize],
    ) -> Result<TrainRowsAtResult, ENNError> {
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


#[doc = "kiss-coverage-off"]
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
    use crate::EpistemicNearestNeighbors;
    use crate::IndexDriver;
    use ndarray::array;

    #[test]
    fn neighbor_distance_helpers() {
        let model = EpistemicNearestNeighbors::new(
            array![[0.0, 0.0], [1.0, 0.0]],
            array![[0.0], [1.0]],
            None,
            false,
            IndexDriver::Exact,
        )
        .unwrap();
        let query = array![[0.1, 0.1]];
        let access = model.index_access();
        let _ = access
            .neighbor_distances_and_indices(&query.view(), 1, false)
            .unwrap();
        let _ = access
            .index_neighbor_distances_and_indices(&query.view(), 1, false)
            .unwrap();
    }
}
