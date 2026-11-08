use crate::error::ENNError;
use crate::input_history_disk::DiskInputs;
use ndarray::{Array2, ArrayView2};
use std::path::Path;

pub(crate) const ORIGINAL_OUTPUTS_FILE: &str = "original_outputs.bin";

pub(crate) fn remove_store(work_dir: &Path) -> Result<(), ENNError> {
    ennbo_bpann::bpann_remove_store(work_dir)
        .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
    for name in [
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
}
