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
    input_fingerprint: Option<String>,
    semantic_only: bool,
    output_fingerprint: Option<String>,
    output_alpha: Option<f64>,
    pub(crate) tied_dims: Vec<Vec<usize>>,
    y_bounds: Vec<(Option<f64>, Option<f64>)>,
}

fn file_error(e: ennbo_bpann::BpannError) -> ENNError {
    ENNError::InvalidParameter(e.to_string())
}

fn finite(value: f64) -> Option<f64> {
    value.is_finite().then_some(value)
}

pub(crate) fn reject_existing_store(work_dir: &Path) -> Result<(), ENNError> {
    let held = [STORE_IDENTITY_FILE, STORE_METADATA_FILE]
        .iter()
        .any(|name| work_dir.join(name).exists());
    if held {
        return Err(ENNError::InvalidParameter(format!(
            "work_dir {} already holds a store; reopen it with EpistemicNearestNeighbors.open",
            work_dir.display()
        )));
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
            input_fingerprint: layout.input_fingerprint().map(str::to_owned),
            semantic_only: layout.semantic_only(),
            output_fingerprint: layout.output_fingerprint().map(str::to_owned),
            output_alpha: layout.output_alpha(),
            tied_dims: tied_dims.to_vec(),
            y_bounds: y_bounds
                .rows()
                .into_iter()
                .map(|row| (finite(row[0]), finite(row[1])))
                .collect(),
        }
    }

    pub(crate) fn load(work_dir: &Path) -> Result<Self, ENNError> {
        bpann_read_json::<Self>(&work_dir.join(STORE_IDENTITY_FILE))
            .map_err(file_error)?
            .ok_or_else(|| {
                ENNError::InvalidParameter(format!(
                    "work_dir {} has no {STORE_IDENTITY_FILE}, so it holds no store to open",
                    work_dir.display()
                ))
            })
    }

    pub(crate) fn layout(&self, work_dir: PathBuf) -> Result<EnnLayout, ENNError> {
        Ok(EnnLayout::new(work_dir, self.weighting)?
            .with_input_fingerprint(self.input_fingerprint.clone())
            .with_semantic_only(self.semantic_only)
            .with_output_identity(self.output_fingerprint.clone(), self.output_alpha))
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
        bpann_write_json(&work_dir.join(STORE_IDENTITY_FILE), self).map_err(file_error)
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
            .with_input_fingerprint(Some("in".into()))
            .with_semantic_only(true)
            .with_output_identity(Some("out".into()), Some(0.25));
        let bounds = array![[0.0, 1.0], [f64::NEG_INFINITY, f64::INFINITY]];
        let recorded = StoreIdentity::of(&layout, (3, 2), &[vec![0, 1]], &bounds);
        recorded.record(dir.path()).unwrap();
        let loaded = StoreIdentity::load(dir.path()).unwrap();
        assert_eq!(loaded, recorded);
        assert_eq!(loaded.y_bounds(), bounds);
        assert_eq!(loaded.layout(dir.path().into()).unwrap(), layout);
    }

    #[test]
    fn missing_identity_is_a_clear_error() {
        let dir = TempDir::new().unwrap();
        let err = StoreIdentity::load(dir.path()).unwrap_err();
        assert!(err.to_string().contains("store_identity.json"), "{err}");
    }

    #[test]
    fn an_existing_store_is_rejected_by_either_marker_file() {
        let dir = TempDir::new().unwrap();
        reject_existing_store(dir.path()).unwrap();
        for name in [STORE_METADATA_FILE, STORE_IDENTITY_FILE] {
            std::fs::write(dir.path().join(name), "{}").unwrap();
            let err = reject_existing_store(dir.path()).unwrap_err();
            assert!(err.to_string().contains(".open"), "{err}");
            std::fs::remove_file(dir.path().join(name)).unwrap();
        }
    }
}
