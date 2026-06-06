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
            .ensure_index_sync(self.model.scale_x, &self.model.x_scale)
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
        if !self.model.backend.defer_index_sync_for_search() {
            self.ensure_sync()?;
        }
        self.model.backend.search(x, search_k, exclude_nearest)
    }

    pub fn index_neighbor_distances_and_indices(
        &self,
        x: &ArrayView2<f64>,
        search_k: i32,
        exclude_nearest: bool,
        tie_break_neighbors: bool,
    ) -> Result<(Array2<f64>, Array2<i64>), ENNError> {
        let _ = tie_break_neighbors;
        crate::posterior::index_search(self.model, x, search_k, exclude_nearest, tie_break_neighbors)
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
