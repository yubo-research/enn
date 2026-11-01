use std::fs::{self, File, OpenOptions, TryLockError};
use std::path::Path;

use serde::de::DeserializeOwned;
use serde::Serialize;

use crate::error::BpannError;

const STORE_LOCK_FILE: &str = "store.lock";

fn invalid(context: &Path, e: impl std::fmt::Display) -> BpannError {
    BpannError::InvalidParameter(format!("{}: {e}", context.display()))
}

pub fn bpann_lock_store(work_dir: &Path) -> Result<File, BpannError> {
    let path = work_dir.join(STORE_LOCK_FILE);
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(&path)
        .map_err(|e| invalid(&path, e))?;
    match file.try_lock() {
        Ok(()) => Ok(file),
        Err(TryLockError::WouldBlock) => Err(BpannError::InvalidParameter(format!(
            "work_dir {} is already open by another model; release that model or use a different work_dir",
            work_dir.display()
        ))),
        Err(TryLockError::Error(e)) => Err(invalid(&path, e)),
    }
}

pub fn bpann_write_json<T: Serialize>(path: &Path, value: &T) -> Result<(), BpannError> {
    let body = serde_json::to_string(value).map_err(|e| invalid(path, e))?;
    let staged = path.with_extension("json.tmp");
    fs::write(&staged, body).map_err(|e| invalid(&staged, e))?;
    fs::rename(&staged, path).map_err(|e| invalid(path, e))
}

pub fn bpann_read_json<T: DeserializeOwned>(path: &Path) -> Result<Option<T>, BpannError> {
    match fs::read_to_string(path) {
        Ok(text) => serde_json::from_str(&text)
            .map(Some)
            .map_err(|e| invalid(path, e)),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(invalid(path, e)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn lock_is_exclusive_until_the_holder_is_dropped() {
        let dir = TempDir::new().unwrap();
        let holder = bpann_lock_store(dir.path()).unwrap();
        let err = bpann_lock_store(dir.path()).unwrap_err();
        assert!(err.to_string().contains("already open"), "{err}");
        drop(holder);
        assert!(bpann_lock_store(dir.path()).is_ok());
    }

    #[test]
    fn json_round_trips_and_a_missing_file_reads_as_none() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("state.json");
        assert_eq!(bpann_read_json::<Vec<f64>>(&path).unwrap(), None);
        bpann_write_json(&path, &vec![1.0, 2.5]).unwrap();
        assert_eq!(
            bpann_read_json::<Vec<f64>>(&path).unwrap(),
            Some(vec![1.0, 2.5])
        );
        assert!(!path.with_extension("json.tmp").exists());
    }

    #[test]
    fn corrupt_json_is_an_error_not_a_missing_file() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("state.json");
        fs::write(&path, "[1.0,").unwrap();
        assert!(bpann_read_json::<Vec<f64>>(&path).is_err());
    }
}
