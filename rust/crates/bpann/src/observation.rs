use std::fs::{self, OpenOptions};
use std::path::Path;
use std::sync::Mutex;

use memmap2::MmapMut;
use ndarray::{Array2, ArrayView2};

use crate::error::BpannError;
use crate::mmap_store::MmapColumnStore;

pub const FORMAT_VERSION: u32 = 1;
pub const MAX_NUM_DIM: usize = 1024;
pub const MAX_RECORD_STRIDE: usize = 8 * 1024 * 1024;
pub const INDEX_BACKEND: &str = "bpann_disk";

pub type TrainRowsAt = (Array2<f64>, Array2<f64>, Option<Array2<f64>>);

pub fn bpann_validate_dim_limits(num_dim: usize) -> Result<(), BpannError> {
    let record_stride = num_dim * std::mem::size_of::<f64>();
    if num_dim > MAX_NUM_DIM {
        return Err(BpannError::InvalidParameter(format!(
            "num_dim {num_dim} exceeds maximum {MAX_NUM_DIM}"
        )));
    }
    if record_stride > MAX_RECORD_STRIDE {
        return Err(BpannError::InvalidParameter(format!(
            "record_stride {record_stride} exceeds maximum {MAX_RECORD_STRIDE}"
        )));
    }
    Ok(())
}

pub fn bpann_check_append_row_limit(new_n: usize) -> Result<(), BpannError> {
    if new_n >= u32::MAX as usize {
        return Err(BpannError::InvalidParameter(
            "row count exceeds u32::MAX".to_string(),
        ));
    }
    Ok(())
}

pub fn bpann_validate_index_backend(work_dir: &Path, expected: &str) -> Result<(), BpannError> {
    if let Some(backend) = bpann_load_index_backend(work_dir) {
        if backend != expected {
            return Err(BpannError::InvalidParameter(format!(
                "work_dir index_backend is {backend}, expected {expected}"
            )));
        }
    }
    Ok(())
}

pub fn bpann_load_num_obs(work_dir: &Path) -> Option<usize> {
    let sidecar = work_dir.join("num_obs.bin");
    if let Ok(data) = fs::read(&sidecar) {
        if data.len() == 8 {
            return Some(u64::from_le_bytes(data.try_into().ok()?) as usize);
        }
    }
    let text = fs::read_to_string(work_dir.join("metadata.json")).ok()?;
    parse_json_usize_field(&text, "num_obs")
}

pub fn write_num_obs(work_dir: &Path, num_obs: usize) -> Result<(), BpannError> {
    fs::write(
        work_dir.join("num_obs.bin"),
        (num_obs as u64).to_le_bytes(),
    )
    .map_err(|e| BpannError::InvalidParameter(e.to_string()))
}

pub struct NumObsCounter {
    mmap: MmapMut,
}

impl NumObsCounter {
    pub fn open(work_dir: &Path) -> Result<Self, BpannError> {
        let path = work_dir.join("num_obs.bin");
        if !path.exists() {
            fs::write(&path, 0u64.to_le_bytes())
                .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        }
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        if file
            .metadata()
            .map_err(|e| BpannError::InvalidParameter(e.to_string()))?
            .len()
            < 8
        {
            file.set_len(8)
                .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        }
        let mmap = unsafe {
            MmapMut::map_mut(&file).map_err(|e| BpannError::InvalidParameter(e.to_string()))?
        };
        Ok(Self { mmap })
    }

    pub fn set(&mut self, num_obs: usize) {
        self.mmap[0..8].copy_from_slice(&(num_obs as u64).to_le_bytes());
    }
}

pub fn write_indexed_rows(work_dir: &Path, indexed_rows: usize) -> Result<(), BpannError> {
    fs::write(
        work_dir.join("indexed_rows.bin"),
        (indexed_rows as u64).to_le_bytes(),
    )
    .map_err(|e| BpannError::InvalidParameter(e.to_string()))
}

pub fn bpann_load_indexed_rows(work_dir: &Path) -> Option<usize> {
    let sidecar = work_dir.join("indexed_rows.bin");
    if let Ok(data) = fs::read(&sidecar) {
        if data.len() == 8 {
            return Some(u64::from_le_bytes(data.try_into().ok()?) as usize);
        }
    }
    let text = fs::read_to_string(work_dir.join("metadata.json")).ok()?;
    parse_json_usize_field(&text, "indexed_rows")
}

pub fn bpann_load_index_backend(work_dir: &Path) -> Option<String> {
    let text = fs::read_to_string(work_dir.join("metadata.json")).ok()?;
    bpann_parse_json_string_field(&text, "index_backend")
}

pub fn bpann_write_metadata(
    work_dir: &Path,
    num_obs: usize,
    num_dim: usize,
    num_metrics: usize,
    scale_x: bool,
    indexed_rows: usize,
) -> Result<(), BpannError> {
    let json = format!(
        "{{\"format_version\":{FORMAT_VERSION},\"num_obs\":{num_obs},\"num_dim\":{num_dim},\"num_metrics\":{num_metrics},\"scale_x\":{scale_x},\"index_backend\":\"{INDEX_BACKEND}\",\"indexed_rows\":{indexed_rows}}}"
    );
    fs::write(work_dir.join("metadata.json"), json)
        .map_err(|e| BpannError::InvalidParameter(e.to_string()))
}

pub fn bpann_open_or_append_yvar(
    work_dir: &Path,
    num_metrics: usize,
    train_yvar: Option<&Array2<f64>>,
) -> Result<Option<MmapColumnStore>, BpannError> {
    if let Some(yv) = train_yvar {
        let yv_path = work_dir.join("train_yvar.bin");
        let known_nrows = bpann_load_num_obs(work_dir);
        let mut store = MmapColumnStore::mmap_open_or_create(yv_path, num_metrics, known_nrows)?;
        if store.nrows == 0 {
            store.mmap_append(&yv.view())?;
        }
        Ok(Some(store))
    } else {
        Ok(None)
    }
}

pub fn bpann_append_yvar_on_add(
    work_dir: &Path,
    num_metrics: usize,
    train_yvar: &mut Option<MmapColumnStore>,
    yvar: Option<&ArrayView2<f64>>,
) -> Result<(), BpannError> {
    match (train_yvar.as_mut(), yvar) {
        (Some(store), Some(yv)) => store.mmap_append(yv)?,
        (None, Some(yv)) => {
            let yv_path = work_dir.join("train_yvar.bin");
            let known_nrows = bpann_load_num_obs(work_dir);
            let mut store =
                MmapColumnStore::mmap_open_or_create(yv_path, num_metrics, known_nrows)?;
            store.mmap_append(yv)?;
            *train_yvar = Some(store);
        }
        _ => {}
    }
    Ok(())
}

pub fn bpann_train_rows_at(
    n: usize,
    train_x: &MmapColumnStore,
    train_y: &MmapColumnStore,
    train_yvar: Option<&MmapColumnStore>,
    indices: &[usize],
) -> Result<TrainRowsAt, BpannError> {
    for &i in indices {
        if i >= n {
            return Err(BpannError::InvalidParameter(format!(
                "bpann_train_rows_at index {i} out of range [0, {n})"
            )));
        }
    }
    let x = train_x.mmap_gather(indices)?;
    let y = train_y.mmap_gather(indices)?;
    let yvar = train_yvar
        .map(|s| s.mmap_gather(indices))
        .transpose()?;
    Ok((x, y, yvar))
}

pub fn bpann_mark_index_dirty(index_dirty: &Mutex<bool>) {
    *index_dirty.lock().expect("index_dirty mutex poisoned") = true;
}

pub(crate) fn parse_json_usize_field(text: &str, field: &str) -> Option<usize> {
    let needle = format!("\"{field}\":");
    text.split_once(&needle).and_then(|(_, tail)| {
        tail.trim_start()
            .split(|c: char| !c.is_ascii_digit())
            .next()
            .and_then(|digits| digits.parse().ok())
    })
}

pub fn bpann_parse_json_string_field(text: &str, field: &str) -> Option<String> {
    let marker = format!("\"{field}\":\"");
    text.split_once(&marker).and_then(|(_, tail)| {
        tail.split_once('"').map(|(value, _)| value.to_string())
    })
}

#[cfg(test)]
mod kiss_coverage_tests {
    use super::*;
    use crate::mmap_store::MmapColumnStore;
    use ndarray::array;
    use std::sync::Mutex;
    use tempfile::TempDir;

    #[test]
    fn observation_units_are_linked() {
        let _ = (
            crate::observation::bpann_validate_dim_limits,
            crate::observation::bpann_check_append_row_limit,
            crate::observation::bpann_validate_index_backend,
            crate::observation::bpann_load_num_obs,
            crate::observation::write_num_obs,
            crate::observation::write_indexed_rows,
            crate::observation::bpann_load_indexed_rows,
            crate::observation::bpann_load_index_backend,
            crate::observation::bpann_write_metadata,
            crate::observation::bpann_open_or_append_yvar,
            crate::observation::bpann_append_yvar_on_add,
            crate::observation::bpann_train_rows_at,
            crate::observation::bpann_mark_index_dirty,
            crate::observation::bpann_parse_json_string_field,
            crate::observation::parse_json_usize_field,
        );
    }

    #[test]
    fn observation_helpers_called() {
        let dir = TempDir::new().unwrap();
        crate::observation::bpann_validate_dim_limits(4).unwrap();
        assert!(crate::observation::bpann_validate_dim_limits(crate::observation::MAX_NUM_DIM + 1).is_err());
        crate::observation::bpann_check_append_row_limit(10).unwrap();
        crate::observation::bpann_write_metadata(dir.path(), 0, 4, 1, false, 0).unwrap();
        crate::observation::write_num_obs(dir.path(), 0).unwrap();
        crate::observation::write_indexed_rows(dir.path(), 0).unwrap();
        let mut counter = crate::observation::NumObsCounter::open(dir.path()).unwrap();
        counter.set(0);
        assert_eq!(crate::observation::bpann_load_num_obs(dir.path()), Some(0));
        assert_eq!(crate::observation::bpann_load_indexed_rows(dir.path()), Some(0));
        assert_eq!(
            crate::observation::bpann_load_index_backend(dir.path()).as_deref(),
            Some(INDEX_BACKEND)
        );
        crate::observation::bpann_validate_index_backend(dir.path(), INDEX_BACKEND).unwrap();
        let mut yvar =
            crate::observation::bpann_open_or_append_yvar(dir.path(), 1, Some(&array![[0.1]])).unwrap();
        crate::observation::bpann_append_yvar_on_add(
            dir.path(),
            1,
            &mut yvar,
            Some(&array![[0.2]].view()),
        )
        .unwrap();
        let mut x =
            MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), 2, None).unwrap();
        let mut y =
            MmapColumnStore::mmap_open_or_create(dir.path().join("y.bin"), 1, None).unwrap();
        x.mmap_append(&array![[0.0, 0.0]].view()).unwrap();
        y.mmap_append(&array![[0.0]].view()).unwrap();
        crate::observation::bpann_train_rows_at(1, &x, &y, None, &[0]).unwrap();
        let dirty = Mutex::new(false);
        crate::observation::bpann_mark_index_dirty(&dirty);
        assert_eq!(
            crate::observation::bpann_parse_json_string_field(
                r#"{"index_backend":"bpann_disk"}"#,
                "index_backend"
            )
            .as_deref(),
            Some("bpann_disk")
        );
        assert_eq!(
            crate::observation::parse_json_usize_field(r#"{"num_obs":42}"#, "num_obs"),
            Some(42)
        );
    }
}
