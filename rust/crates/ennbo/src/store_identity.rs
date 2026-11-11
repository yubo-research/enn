use std::path::{Path, PathBuf};

use ennbo_bpann::{bpann_read_json, bpann_write_json};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

use crate::error::ENNError;
use crate::layout::EnnLayout;

pub(crate) const STORE_IDENTITY_FILE: &str = "store_identity.json";
const STORE_METADATA_FILE: &str = "metadata.json";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct StoreIdentity {
    pub(crate) num_dim: usize,
    pub(crate) num_metrics: usize,
    weighting: bool,
    #[serde(default)]
    aggregate_duplicates: bool,
    pub(crate) tied_dims: Vec<Vec<usize>>,
    y_bounds: Vec<(Option<f64>, Option<f64>)>,
    #[serde(default = "committed_by_default")]
    committed: bool,
}

fn committed_by_default() -> bool {
    true
}

fn file_error(e: ennbo_bpann::BpannError) -> ENNError {
    ENNError::InvalidParameter(e.to_string())
}

fn finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

pub(crate) fn holds_committed_store(work_dir: &Path) -> Result<bool, ENNError> {
    let identity = bpann_read_json::<StoreIdentity>(&work_dir.join(STORE_IDENTITY_FILE));
    Ok(identity.map_err(file_error)?.is_some_and(|i| i.committed))
}

fn already_holds_store(work_dir: &Path) -> ENNError {
    ENNError::InvalidParameter(format!(
        "work_dir {} already holds a store; reopen it with EpistemicNearestNeighbors.open",
        work_dir.display()
    ))
}

pub(crate) fn claim_store_dir(work_dir: &Path) -> Result<(), ENNError> {
    if work_dir.join(STORE_IDENTITY_FILE).exists() {
        if holds_committed_store(work_dir)? {
            return Err(already_holds_store(work_dir));
        }
        return crate::input_history::remove_store(work_dir);
    }
    if work_dir.join(STORE_METADATA_FILE).exists() {
        return Err(already_holds_store(work_dir));
    }
    Ok(())
}

impl StoreIdentity {
    pub(crate) fn of(
        layout: &EnnLayout,
        shape: (usize, usize),
        tied_dims: &[Vec<usize>],
        y_bounds: &Array2<f64>,
    ) -> Self {
        Self {
            num_dim: shape.0,
            num_metrics: shape.1,
            weighting: layout.weighting(),
            aggregate_duplicates: layout.aggregate_duplicates(),
            tied_dims: tied_dims.to_vec(),
            y_bounds: y_bounds
                .rows()
                .into_iter()
                .map(|row| (finite(row[0]), finite(row[1])))
                .collect(),
            committed: false,
        }
    }

    pub(crate) fn load(work_dir: &Path) -> Result<Self, ENNError> {
        let identity = bpann_read_json::<Self>(&work_dir.join(STORE_IDENTITY_FILE))
            .map_err(file_error)?
            .ok_or_else(|| {
                ENNError::InvalidParameter(format!(
                    "work_dir {} has no {STORE_IDENTITY_FILE}, so it holds no store to open",
                    work_dir.display()
                ))
            })?;
        if !identity.committed {
            return Err(ENNError::InvalidParameter(format!(
                "work_dir {} holds a store whose creation never finished; create it again",
                work_dir.display()
            )));
        }
        Ok(identity)
    }

    pub(crate) fn layout(&self, work_dir: PathBuf) -> Result<EnnLayout, ENNError> {
        Ok(EnnLayout::new(work_dir, self.weighting)?
            .with_aggregate_duplicates(self.aggregate_duplicates))
    }

    pub(crate) fn y_bounds(&self) -> Array2<f64> {
        let mut bounds = Array2::zeros((self.y_bounds.len(), 2));
        for (i, (lo, hi)) in self.y_bounds.iter().enumerate() {
            bounds[[i, 0]] = lo.unwrap_or(f64::NEG_INFINITY);
            bounds[[i, 1]] = hi.unwrap_or(f64::INFINITY);
        }
        bounds
    }

    pub(crate) fn record(&self, work_dir: &Path) -> Result<(), ENNError> {
        std::fs::create_dir_all(work_dir).map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
        bpann_write_json(&work_dir.join(STORE_IDENTITY_FILE), self).map_err(file_error)
    }

    pub(crate) fn commit(mut self, work_dir: &Path) -> Result<(), ENNError> {
        self.committed = true;
        self.record(work_dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use tempfile::TempDir;

    #[test]
    fn identity_round_trips_every_setting_including_open_bounds() {
        let dir = TempDir::new().unwrap();
        let layout = EnnLayout::new(dir.path(), true)
            .unwrap()
            .with_aggregate_duplicates(true);
        let bounds = array![[0.0, 1.0], [f64::NEG_INFINITY, f64::INFINITY]];
        let recorded = StoreIdentity::of(&layout, (3, 2), &[vec![0, 1]], &bounds);
        recorded.clone().commit(dir.path()).unwrap();
        let loaded = StoreIdentity::load(dir.path()).unwrap();
        assert_eq!(loaded.tied_dims, recorded.tied_dims);
        assert_eq!(loaded.num_metrics, 2);
        assert_eq!(loaded.y_bounds(), bounds);
        assert_eq!(loaded.layout(dir.path().into()).unwrap(), layout);
    }

    #[test]
    fn identity_written_before_aggregate_duplicates_existed_reopens_with_it_off() {
        let dir = TempDir::new().unwrap();
        let layout = EnnLayout::new(dir.path(), false).unwrap();
        StoreIdentity::of(&layout, (1, 1), &[], &array![[0.0, 1.0]])
            .commit(dir.path())
            .unwrap();
        let path = work_file(dir.path(), STORE_IDENTITY_FILE);
        let text = std::fs::read_to_string(&path).unwrap();
        let older = text.replace("\"aggregate_duplicates\":false,", "");
        assert_ne!(older, text);
        std::fs::write(&path, older).unwrap();
        let loaded = StoreIdentity::load(dir.path()).unwrap();
        assert!(!loaded
            .layout(dir.path().into())
            .unwrap()
            .aggregate_duplicates());
    }

    #[test]
    fn missing_identity_is_a_clear_error() {
        let dir = TempDir::new().unwrap();
        let err = StoreIdentity::load(dir.path()).unwrap_err();
        assert!(err.to_string().contains("store_identity.json"), "{err}");
    }

    fn identity(dir: &Path) -> StoreIdentity {
        let layout = EnnLayout::new(dir, false).unwrap();
        StoreIdentity::of(&layout, (1, 1), &[], &array![[0.0, 1.0]])
    }

    #[test]
    fn a_finished_store_or_a_foreign_one_is_rejected() {
        let dir = TempDir::new().unwrap();
        claim_store_dir(dir.path()).unwrap();
        identity(dir.path()).commit(dir.path()).unwrap();
        let err = claim_store_dir(dir.path()).unwrap_err();
        assert!(err.to_string().contains(".open"), "{err}");
        assert!(work_file(dir.path(), STORE_IDENTITY_FILE).exists());
        std::fs::remove_file(work_file(dir.path(), STORE_IDENTITY_FILE)).unwrap();
        std::fs::write(work_file(dir.path(), STORE_METADATA_FILE), "{}").unwrap();
        let err = claim_store_dir(dir.path()).unwrap_err();
        assert!(err.to_string().contains(".open"), "{err}");
    }

    fn work_file(dir: &Path, name: &str) -> PathBuf {
        dir.join(name)
    }

    #[test]
    fn an_unfinished_store_cannot_be_opened_and_is_replaced_by_a_new_one() {
        let dir = TempDir::new().unwrap();
        identity(dir.path()).record(dir.path()).unwrap();
        std::fs::write(work_file(dir.path(), STORE_METADATA_FILE), "{}").unwrap();
        let err = StoreIdentity::load(dir.path()).unwrap_err();
        assert!(err.to_string().contains("never finished"), "{err}");
        assert!(!holds_committed_store(dir.path()).unwrap());
        claim_store_dir(dir.path()).unwrap();
        assert!(!work_file(dir.path(), STORE_IDENTITY_FILE).exists());
        assert!(!work_file(dir.path(), STORE_METADATA_FILE).exists());
    }

    #[test]
    fn identity_written_before_the_commit_flag_existed_counts_as_committed() {
        let dir = TempDir::new().unwrap();
        identity(dir.path()).record(dir.path()).unwrap();
        let path = work_file(dir.path(), STORE_IDENTITY_FILE);
        let text = std::fs::read_to_string(&path).unwrap();
        let older = text.replace(",\"committed\":false", "");
        assert_ne!(older, text);
        std::fs::write(&path, older).unwrap();
        assert!(holds_committed_store(dir.path()).unwrap());
    }
}
