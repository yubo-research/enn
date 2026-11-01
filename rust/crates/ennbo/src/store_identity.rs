use std::path::Path;

use ennbo_bpann::{bpann_read_json, bpann_write_json};
use serde::{Deserialize, Serialize};

use crate::backend::disk_observation::parse_json_usize_field;
use crate::error::ENNError;
use crate::layout::EnnLayout;

pub(crate) const STORE_IDENTITY_FILE: &str = "store_identity.json";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct StoreIdentity {
    num_dim: usize,
    num_metrics: usize,
    weighting: bool,
    input_fingerprint: Option<String>,
}

fn file_error(e: ennbo_bpann::BpannError) -> ENNError {
    ENNError::InvalidParameter(e.to_string())
}

impl StoreIdentity {
    pub(crate) fn of(layout: &EnnLayout, num_dim: usize, num_metrics: usize) -> Self {
        Self {
            num_dim,
            num_metrics,
            weighting: layout.weighting(),
            input_fingerprint: layout.input_fingerprint().map(str::to_owned),
        }
    }

    fn mismatch(&self, work_dir: &Path, recorded: impl std::fmt::Debug) -> ENNError {
        ENNError::InvalidParameter(format!(
            "work_dir {} holds {recorded:?}, which does not match the requested {self:?}; \
             reopen with the original settings or use a fresh work_dir",
            work_dir.display()
        ))
    }

    fn verify_store_shape(&self, work_dir: &Path) -> Result<(), ENNError> {
        let Ok(meta) = std::fs::read_to_string(work_dir.join("metadata.json")) else {
            return Ok(());
        };
        let recorded = (
            parse_json_usize_field(&meta, "num_dim"),
            parse_json_usize_field(&meta, "num_metrics"),
        );
        if recorded == (Some(self.num_dim), Some(self.num_metrics)) {
            return Ok(());
        }
        Err(self.mismatch(work_dir, recorded))
    }

    pub(crate) fn verify(&self, work_dir: &Path) -> Result<(), ENNError> {
        let path = work_dir.join(STORE_IDENTITY_FILE);
        match bpann_read_json::<Self>(&path).map_err(file_error)? {
            Some(recorded) if recorded == *self => Ok(()),
            Some(recorded) => Err(self.mismatch(work_dir, recorded)),
            None => self.verify_store_shape(work_dir),
        }
    }

    pub(crate) fn record(&self, work_dir: &Path) -> Result<(), ENNError> {
        let path = work_dir.join(STORE_IDENTITY_FILE);
        if path.exists() {
            return Ok(());
        }
        bpann_write_json(&path, self).map_err(file_error)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn identity(dir: &TempDir, weighting: bool, fingerprint: Option<&str>) -> StoreIdentity {
        let layout = EnnLayout::new(dir.path(), weighting)
            .unwrap()
            .with_input_fingerprint(fingerprint.map(str::to_owned));
        StoreIdentity::of(&layout, 3, 1)
    }

    #[test]
    fn a_recorded_identity_accepts_itself_and_rejects_changes() {
        let dir = TempDir::new().unwrap();
        let recorded = identity(&dir, false, Some("a"));
        recorded.record(dir.path()).unwrap();
        recorded.verify(dir.path()).unwrap();
        for other in [
            identity(&dir, true, Some("a")),
            identity(&dir, false, Some("b")),
            identity(&dir, false, None),
            StoreIdentity {
                num_dim: 5,
                ..recorded.clone()
            },
            StoreIdentity {
                num_metrics: 4,
                ..recorded.clone()
            },
        ] {
            let err = other.verify(dir.path()).unwrap_err();
            assert!(err.to_string().contains("work_dir"), "{err}");
        }
    }

    #[test]
    fn record_keeps_the_first_identity() {
        let dir = TempDir::new().unwrap();
        let first = identity(&dir, false, None);
        first.record(dir.path()).unwrap();
        identity(&dir, true, None).record(dir.path()).unwrap();
        first.verify(dir.path()).unwrap();
    }

    #[test]
    fn a_store_without_a_recorded_identity_is_checked_against_its_metadata() {
        let dir = TempDir::new().unwrap();
        let requested = identity(&dir, false, None);
        requested.verify(dir.path()).unwrap();
        std::fs::write(
            dir.path().join("metadata.json"),
            r#"{"num_dim":3,"num_metrics":1}"#,
        )
        .unwrap();
        requested.verify(dir.path()).unwrap();
        std::fs::write(
            dir.path().join("metadata.json"),
            r#"{"num_dim":5,"num_metrics":1}"#,
        )
        .unwrap();
        assert!(requested.verify(dir.path()).is_err());
    }
}
