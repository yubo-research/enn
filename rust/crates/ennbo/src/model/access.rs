use ndarray::{Array1, Array2, ArrayView2};

use super::EpistemicNearestNeighbors;
use crate::backend::TrainRowsAtResult;
use crate::error::ENNError;

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
        self.model.backend.index_memory_bytes()
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
    }
}
