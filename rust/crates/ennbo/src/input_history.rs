use crate::error::ENNError;
use crate::input_history_disk::DiskInputs;
use ndarray::{Array2, ArrayView2};
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

pub(crate) const ORIGINAL_INPUTS_FILE: &str = "original_inputs.bin";
pub(crate) const ORIGINAL_OUTPUTS_FILE: &str = "original_outputs.bin";
const INPUT_TRANSFORM_FILE: &str = "input_transform.npz";
const OUTPUT_PROJECTION_FILE: &str = "output_projection.npz";

#[derive(Debug, Clone, Default)]
pub struct StoreAttachments {
    pub original_inputs: Option<Array2<f64>>,
    pub input_transform: Option<Vec<u8>>,
    pub output_projection: Option<Vec<u8>>,
}

pub fn input_transform_path(work_dir: &Path) -> PathBuf {
    work_dir.join(INPUT_TRANSFORM_FILE)
}

pub fn output_projection_path(work_dir: &Path) -> PathBuf {
    work_dir.join(OUTPUT_PROJECTION_FILE)
}

fn write_artifact(path: &Path, bytes: &[u8]) -> Result<(), ENNError> {
    let failed = |e: std::io::Error| {
        ENNError::InvalidParameter(format!("write {}: {e}", path.display()))
    };
    let staged = path.with_extension("tmp");
    let mut file = File::create(&staged).map_err(failed)?;
    file.write_all(bytes).map_err(failed)?;
    file.sync_all().map_err(failed)?;
    std::fs::rename(&staged, path).map_err(failed)
}

impl StoreAttachments {
    pub(crate) fn write_artifacts(&self, work_dir: &Path) -> Result<(), ENNError> {
        if let Some(bytes) = &self.input_transform {
            write_artifact(&input_transform_path(work_dir), bytes)?;
        }
        if let Some(bytes) = &self.output_projection {
            write_artifact(&output_projection_path(work_dir), bytes)?;
        }
        Ok(())
    }
}

pub(crate) fn remove_store(work_dir: &Path) -> Result<(), ENNError> {
    ennbo_bpann::bpann_remove_store(work_dir)
        .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
    for name in [
        ORIGINAL_INPUTS_FILE,
        ORIGINAL_OUTPUTS_FILE,
        crate::store_identity::STORE_IDENTITY_FILE,
        crate::metric_auto::METRIC_STATE_FILE,
        INPUT_TRANSFORM_FILE,
        OUTPUT_PROJECTION_FILE,
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
    pub(crate) fn create(
        work_dir: &Path,
        filename: &str,
        rows: &ArrayView2<f64>,
    ) -> Result<Self, ENNError> {
        let disk = DiskInputs::create(work_dir, filename, rows)?;
        Ok(Self { disk })
    }

    pub(crate) fn recover(
        work_dir: &Path,
        filename: &str,
        committed_rows: usize,
    ) -> Result<Option<Self>, ENNError> {
        let Some((mut disk, _)) = DiskInputs::reopen(work_dir, filename)? else {
            return Ok(None);
        };
        disk.keep_rows(committed_rows)?;
        Ok(Some(Self { disk }))
    }

    pub(crate) fn write_at(
        &mut self,
        first_row: usize,
        rows: &ArrayView2<f64>,
    ) -> Result<(), ENNError> {
        self.disk.keep_rows(first_row)?;
        self.disk.append(rows)
    }

    pub fn gather(&self, indices: &[usize]) -> Result<Array2<f64>, ENNError> {
        self.disk.gather(indices)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const NAME: &str = "history.bin";

    #[test]
    fn original_observations_remain_ordered_after_a_write() {
        let dir = tempfile::tempdir().unwrap();
        let rows = array![[1., 2.], [8., 9.]];
        let mut history = InputHistory::create(dir.path(), NAME, &rows.view()).unwrap();
        history
            .write_at(2, &array![[3., 4.], [5., 6.]].view())
            .unwrap();
        assert_eq!(
            history.gather(&[3, 0, 3]).unwrap(),
            array![[5., 6.], [1., 2.], [5., 6.]]
        );
        assert!(history.gather(&[4]).is_err());
        assert!(history.write_at(4, &array![[7.]].view()).is_err());
    }

    #[test]
    fn a_write_at_a_row_replaces_rows_that_were_never_committed() {
        let dir = tempfile::tempdir().unwrap();
        let rows = array![[1.], [2.], [3.]];
        let mut history = InputHistory::create(dir.path(), NAME, &rows.view()).unwrap();
        history.write_at(1, &array![[9.]].view()).unwrap();
        assert_eq!(history.gather(&[0, 1]).unwrap(), array![[1.], [9.]]);
        assert!(history.gather(&[2]).is_err());
        assert!(history.write_at(5, &array![[7.]].view()).is_err());
    }

    #[test]
    fn recovery_drops_uncommitted_rows_and_refuses_missing_ones() {
        let dir = tempfile::tempdir().unwrap();
        let rows = array![[1.], [2.], [3.]];
        drop(InputHistory::create(dir.path(), NAME, &rows.view()).unwrap());
        assert!(InputHistory::recover(dir.path(), NAME, 4).is_err());
        let history = InputHistory::recover(dir.path(), NAME, 2).unwrap().unwrap();
        assert_eq!(history.gather(&[1]).unwrap(), array![[2.]]);
        assert!(history.gather(&[2]).is_err());
        drop(history);
        let again = InputHistory::recover(dir.path(), NAME, 2).unwrap().unwrap();
        assert_eq!(again.gather(&[0, 1]).unwrap(), array![[1.], [2.]]);
        assert!(InputHistory::recover(dir.path(), "absent.bin", 0)
            .unwrap()
            .is_none());
    }

    #[test]
    fn recovery_discards_bytes_from_an_interrupted_write() {
        let dir = tempfile::tempdir().unwrap();
        let rows = array![[1.], [2.]];
        drop(InputHistory::create(dir.path(), NAME, &rows.view()).unwrap());
        let path = dir.path().join(NAME);
        let size = std::fs::metadata(&path).unwrap().len();
        let file = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
        file.set_len(size + 3).unwrap();
        let history = InputHistory::recover(dir.path(), NAME, 2).unwrap().unwrap();
        assert_eq!(history.gather(&[1]).unwrap(), array![[2.]]);
        assert_eq!(std::fs::metadata(&path).unwrap().len(), size);
    }

    #[test]
    fn a_shortened_history_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let rows = array![[1.], [2.]];
        drop(InputHistory::create(dir.path(), NAME, &rows.view()).unwrap());
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(dir.path().join(NAME))
            .unwrap();
        file.set_len(25).unwrap();
        assert!(InputHistory::recover(dir.path(), NAME, 2).is_err());
    }

    #[test]
    fn an_artifact_write_replaces_the_file_and_leaves_no_staged_copy() {
        let dir = tempfile::tempdir().unwrap();
        let attachments = StoreAttachments {
            input_transform: Some(vec![1, 2]),
            output_projection: Some(vec![3]),
            ..Default::default()
        };
        attachments.write_artifacts(dir.path()).unwrap();
        assert_eq!(
            std::fs::read(input_transform_path(dir.path())).unwrap(),
            [1, 2]
        );
        assert_eq!(
            std::fs::read(output_projection_path(dir.path())).unwrap(),
            [3]
        );
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 2);
    }
}
