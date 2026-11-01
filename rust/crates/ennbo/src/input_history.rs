use crate::error::ENNError;
use crate::input_history_disk::DiskInputs;
use crate::model::EpistemicNearestNeighbors;
use ndarray::{Array2, ArrayView2};
use std::path::Path;

const ORIGINAL_INPUTS_FILE: &str = "original_inputs.bin";
pub(crate) const ORIGINAL_OUTPUTS_FILE: &str = "original_outputs.bin";

pub(crate) fn remove_store(work_dir: &Path) -> Result<(), ENNError> {
    ennbo_bpann::bpann_remove_store(work_dir)
        .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
    for name in [
        ORIGINAL_INPUTS_FILE,
        ORIGINAL_OUTPUTS_FILE,
        crate::store_identity::STORE_IDENTITY_FILE,
        crate::metric_auto::METRIC_STATE_FILE,
    ] {
        match std::fs::remove_file(work_dir.join(name)) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                return Err(ENNError::InvalidParameter(format!("remove {name}: {e}")));
            }
            _ => {}
        }
    }
    Ok(())
}

pub struct InputHistory {
    disk: DiskInputs,
}
impl InputHistory {
    pub fn for_model(
        rows: Array2<f64>,
        model: &EpistemicNearestNeighbors,
    ) -> Result<Self, ENNError> {
        Self::for_named_model(rows, model, ORIGINAL_INPUTS_FILE)
    }
    pub(crate) fn for_named_model(
        rows: Array2<f64>,
        model: &EpistemicNearestNeighbors,
        filename: &str,
    ) -> Result<Self, ENNError> {
        let directory = model.layout.work_dir();
        let disk = DiskInputs::open(directory, filename, &rows.view(), model.len())?;
        Ok(Self { disk })
    }
    pub fn append(&mut self, rows: &ArrayView2<f64>) -> Result<(), ENNError> {
        self.disk.append(rows)
    }
    pub fn gather(&self, indices: &[usize]) -> Result<Array2<f64>, ENNError> {
        self.disk.gather(indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_helpers::test_model;
    use ndarray::array;
    #[test]
    fn original_observations_remain_ordered_after_append() {
        let model = test_model(array![[0.], [1.]], array![[0.], [1.]], None, false);
        let mut history = InputHistory::for_model(array![[1., 2.], [8., 9.]], &model).unwrap();
        history.append(&array![[3., 4.], [5., 6.]].view()).unwrap();
        assert_eq!(
            history.gather(&[3, 0, 3]).unwrap(),
            array![[5., 6.], [1., 2.], [5., 6.]]
        );
        assert!(history.gather(&[4]).is_err());
        assert!(history.append(&array![[7.]].view()).is_err());
        assert_eq!(history.gather(&[3]).unwrap(), array![[5., 6.]]);
    }
}
