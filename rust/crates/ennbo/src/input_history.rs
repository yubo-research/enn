//! Original observations for input representations that cannot be inverted.
use crate::error::ENNError;
use crate::input_history_disk::DiskInputs;
use crate::model::EpistemicNearestNeighbors;
use ndarray::{Array2, ArrayView2, Axis};

/// Original rows retained beside semantic-only model coordinates.
pub struct InputHistory {
    rows: Array2<f64>,
    disk: Option<DiskInputs>,
}
impl InputHistory {
    /// Retain the original training observations.
    pub fn new(rows: Array2<f64>) -> Self {
        Self { rows, disk: None }
    }
    /// Select storage from the model, reopening original disk rows when needed.
    pub fn for_model(
        rows: Array2<f64>,
        model: &EpistemicNearestNeighbors,
    ) -> Result<Self, ENNError> {
        Self::for_named_model(rows, model, "original_inputs.bin")
    }
    /// Store a separate original observation matrix alongside a disk model.
    pub(crate) fn for_named_model(rows: Array2<f64>, model: &EpistemicNearestNeighbors, filename: &str) -> Result<Self, ENNError> {
        if let Some(directory) = model.layout.work_dir() {
            let disk = DiskInputs::open(directory, filename, &rows.view(), model.len())?;
            return Ok(Self {
                rows: Array2::zeros((0, rows.ncols())),
                disk: Some(disk),
            });
        }
        Ok(Self::new(rows))
    }
    /// Append originals after the model accepts the corresponding features.
    pub fn append(&mut self, rows: &ArrayView2<f64>) -> Result<(), ENNError> {
        if let Some(disk) = &mut self.disk {
            return disk.append(rows);
        }
        self.rows = ndarray::concatenate(Axis(0), &[self.rows.view(), rows.view()])
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
        Ok(())
    }
    /// Retrieve originals in caller-specified order.
    pub fn gather(&self, indices: &[usize]) -> Result<Array2<f64>, ENNError> {
        if let Some(disk) = &self.disk {
            return disk.gather(indices);
        }
        if indices.iter().any(|&i| i >= self.rows.nrows()) {
            return Err(ENNError::InvalidParameter(
                "observation index out of range".into(),
            ));
        }
        Ok(self.rows.select(Axis(0), indices))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    #[test]
    fn original_observations_remain_ordered_after_append() {
        let mut history = InputHistory::new(array![[1., 2.]]);
        history.append(&array![[3., 4.], [5., 6.]].view()).unwrap();
        assert_eq!(
            history.gather(&[2, 0, 2]).unwrap(),
            array![[5., 6.], [1., 2.], [5., 6.]]
        );
        assert!(history.gather(&[3]).is_err());
        assert!(history.append(&array![[7.]].view()).is_err());
        assert_eq!(history.gather(&[2]).unwrap(), array![[5., 6.]]);
    }
}
