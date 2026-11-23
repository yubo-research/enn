mod lsh;
mod search;
#[cfg(test)]
mod tests;

use ennbo_bpann::{bpann_read_json, bpann_write_json};
use ndarray::{Array1, Array2, ArrayView2};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::backend::{MmapColumnStore, TrainRowsAtResult};
use crate::error::ENNError;
use lsh::LshTables;
pub use lsh::O1Params;
use search::SearchContext;

pub(crate) const O1_META_FILE: &str = "o1_meta.json";
const O1_SEED: u64 = 0x5eed_0001;

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct O1Meta {
    num_rows: usize,
    has_yvar: bool,
    x_scale: Option<Vec<f64>>,
    hash_scale: Option<Vec<f64>>,
}

pub struct O1Store {
    work_dir: PathBuf,
    x: MmapColumnStore,
    y: MmapColumnStore,
    yvar: Option<MmapColumnStore>,
    x_scale: Option<Array1<f64>>,
    tables: LshTables,
}

fn io_error(e: impl std::fmt::Display) -> ENNError {
    ENNError::InvalidParameter(e.to_string())
}

fn store_path(work_dir: &Path, name: &str) -> PathBuf {
    work_dir.join(format!("o1_{name}.bin"))
}

impl O1Store {
    pub fn open_or_create(
        work_dir: PathBuf,
        num_dim: usize,
        num_metrics: usize,
    ) -> Result<Self, ENNError> {
        Self::open_with_params(work_dir, (num_dim, num_metrics), O1Params::default())
    }

    pub fn open_with_params(
        work_dir: PathBuf,
        shape: (usize, usize),
        params: O1Params,
    ) -> Result<Self, ENNError> {
        let (num_dim, num_metrics) = shape;
        std::fs::create_dir_all(&work_dir).map_err(io_error)?;
        let meta: O1Meta = bpann_read_json(&work_dir.join(O1_META_FILE))
            .map_err(io_error)?
            .unwrap_or_default();
        let n = Some(meta.num_rows);
        let x = MmapColumnStore::mmap_open_or_create(store_path(&work_dir, "x"), num_dim, n)?;
        let y = MmapColumnStore::mmap_open_or_create(store_path(&work_dir, "y"), num_metrics, n)?;
        let yvar = meta
            .has_yvar
            .then(|| MmapColumnStore::mmap_open_or_create(store_path(&work_dir, "yvar"), num_metrics, n))
            .transpose()?;
        let mut tables = LshTables::new(num_dim, O1_SEED, params);
        let hash_scale = meta.hash_scale.unwrap_or_else(|| vec![1.0; num_dim]);
        tables.rehash(&x, &hash_scale)?;
        tables.insert_rows(&x, 0)?;
        Ok(Self {
            work_dir,
            x,
            y,
            yvar,
            x_scale: meta.x_scale.map(Array1::from),
            tables,
        })
    }

    fn write_meta(&self) -> Result<(), ENNError> {
        let meta = O1Meta {
            num_rows: self.x.nrows,
            has_yvar: self.yvar.is_some(),
            x_scale: self.x_scale.as_ref().map(|s| s.to_vec()),
            hash_scale: Some(self.tables.hash_scale().to_vec()),
        };
        bpann_write_json(&self.work_dir.join(O1_META_FILE), &meta).map_err(io_error)
    }

    pub fn len(&self) -> usize {
        self.x.nrows
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn num_dim(&self) -> usize {
        self.x.ncols
    }

    pub fn num_metrics(&self) -> usize {
        self.y.ncols
    }

    pub fn hash_bits(&self) -> usize {
        self.tables.bits()
    }

    fn append_yvar(&mut self, yvar: Option<&ArrayView2<f64>>, rows: usize) -> Result<(), ENNError> {
        match (yvar, self.yvar.as_mut()) {
            (Some(v), Some(store)) => store.mmap_append(v),
            (None, None) => Ok(()),
            (Some(v), None) if self.x.nrows == 0 => {
                let path = store_path(&self.work_dir, "yvar");
                let mut store = MmapColumnStore::mmap_open_or_create(path, self.y.ncols, Some(0))?;
                store.mmap_append(v)?;
                self.yvar = Some(store);
                Ok(())
            }
            _ if rows == 0 => Ok(()),
            _ => Err(ENNError::InvalidParameter(
                "train_yvar must be given for every add or for none".into(),
            )),
        }
    }

    pub fn append_rows(
        &mut self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Result<(), ENNError> {
        if x.nrows() != y.nrows() || yvar.is_some_and(|v| v.nrows() != x.nrows()) {
            return Err(ENNError::InvalidParameter(
                "train_x, train_y and train_yvar must have the same number of rows".into(),
            ));
        }
        let start = self.x.nrows;
        self.append_yvar(yvar, x.nrows())?;
        self.y.mmap_append(y)?;
        self.x.mmap_append(x)?;
        self.tables.insert_rows(&self.x, start)?;
        self.write_meta()
    }

    pub fn metric_scale(&self) -> Option<Array1<f64>> {
        self.x_scale.clone()
    }

    pub fn set_metric_scale(&mut self, x_scale: &Array1<f64>, rebuild: bool) -> Result<(), ENNError> {
        if x_scale.len() != self.num_dim() || !x_scale.iter().all(|s| s.is_finite() && *s > 0.0) {
            return Err(ENNError::InvalidParameter(
                "metric x_scale must be finite, positive and match num_dim".into(),
            ));
        }
        self.x_scale = Some(x_scale.clone());
        if rebuild || self.tables.bits() == 0 {
            self.tables.rehash(&self.x, &x_scale.to_vec())?;
        }
        self.write_meta()
    }

    pub fn search(
        &self,
        x: &ArrayView2<f64>,
        search_k: usize,
        exclude_nearest: bool,
    ) -> Result<(Array2<f64>, Array2<i64>), ENNError> {
        let weights: Vec<f64> = match &self.x_scale {
            Some(s) => s.iter().map(|v| 1.0 / (v * v)).collect(),
            None => vec![1.0; self.num_dim()],
        };
        let ctx = SearchContext {
            tables: &self.tables,
            rows: &self.x,
            weights: &weights,
        };
        ctx.search(x, search_k, exclude_nearest)
    }

    pub fn train_rows_at(&self, indices: &[usize]) -> Result<TrainRowsAtResult, ENNError> {
        crate::backend::disk_observation::mmap_train_rows_at(
            self.len(),
            &self.x,
            &self.y,
            self.yvar.as_ref(),
            indices,
        )
    }

    pub fn row_x(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        Ok(Array1::from(self.x.mmap_row_slice(i)?.to_vec()))
    }

    pub fn row_y(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        Ok(Array1::from(self.y.mmap_row_slice(i)?.to_vec()))
    }

    pub fn row_yvar(&self, i: usize) -> Result<Option<Array1<f64>>, ENNError> {
        self.yvar
            .as_ref()
            .map(|s| s.mmap_row_slice(i).map(|r| Array1::from(r.to_vec())))
            .transpose()
    }

    pub fn index_memory_bytes(&self) -> usize {
        self.tables.memory_bytes()
    }
}
