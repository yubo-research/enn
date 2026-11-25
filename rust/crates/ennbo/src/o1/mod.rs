mod buckets;
mod cache;
mod cache_shard;
mod disk_index;
mod forest;
mod forest_bound;
mod forest_node;
mod forest_knobs;
mod forest_order;
mod forest_pack;
mod forest_plan;
mod forest_query;
mod forest_rules;
mod forest_seg;
mod forest_repair;
mod forest_split;
mod forest_stage;
mod forest_tree;
mod index_kind;
mod index_order;
mod keys;
mod lsh;
mod pending;
mod profile;
mod search;
mod row_codes;
mod row_order;
pub(crate) mod store_rows;
#[cfg(test)]
mod cache_tests;
#[cfg(test)]
mod disk_tests;
#[cfg(test)]
mod forest_tests;
#[cfg(test)]
mod forest_repair_tests;
#[cfg(test)]
mod forest_pivot_tests;
#[cfg(test)]
mod tests;

use ennbo_bpann::{bpann_read_json, bpann_write_json};
use ndarray::{Array1, Array2, ArrayView2};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use crate::backend::{MmapColumnStore, TrainRowsAtResult};
use crate::error::ENNError;
use cache::SharedCache;
pub use cache::CacheStats;
pub use disk_index::O1Occupancy;
pub use forest::{ForestParams, MaintStats};
pub use row_order::OrderParams;
pub use forest_seg::SegParams;
pub use forest_rules::{ForestRules, ProbeRule, SelectRule, SplitConfig, SplitRule};
use index_kind::O1Index;
pub use index_kind::O1IndexParams;
use lsh::io_error;
pub use lsh::O1Params;
use profile::QueryProfile;
pub use profile::ProfileStats;
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

pub const DEFAULT_NODE_CACHE_BYTES: usize = 1 << 20;

pub struct O1Store {
    work_dir: PathBuf,
    x: MmapColumnStore,
    y: MmapColumnStore,
    yvar: Option<MmapColumnStore>,
    x_scale: Option<Array1<f64>>,
    index: O1Index,
    cache: SharedCache,
    profile: QueryProfile,
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

    pub fn open_for(
        store: crate::layout::StoreType,
        work_dir: PathBuf,
        num_dim: usize,
        num_metrics: usize,
    ) -> Result<Self, ENNError> {
        if store == crate::layout::StoreType::O1Forest {
            Self::open_forest(work_dir, num_dim, num_metrics)
        } else {
            Self::open_or_create(work_dir, num_dim, num_metrics)
        }
    }

    pub fn open_forest(work_dir: PathBuf, num_dim: usize, num_metrics: usize) -> Result<Self, ENNError> {
        let params = O1IndexParams::Forest(ForestParams::default());
        Self::open_with_index(work_dir, (num_dim, num_metrics), params)
    }

    pub fn open_with_params(
        work_dir: PathBuf,
        shape: (usize, usize),
        params: O1Params,
    ) -> Result<Self, ENNError> {
        Self::open_with_index(work_dir, shape, O1IndexParams::Lsh(params))
    }

    pub fn open_with_index(
        work_dir: PathBuf,
        shape: (usize, usize),
        params: O1IndexParams,
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
        let hash_scale = meta.hash_scale.unwrap_or_else(|| vec![1.0; num_dim]);
        let index = O1Index::open(&work_dir, O1_SEED, params, &hash_scale, &x)?;
        let mut store = Self {
            work_dir,
            x,
            y,
            yvar,
            x_scale: meta.x_scale.map(Array1::from),
            index,
            cache: None,
            profile: QueryProfile::default(),
        };
        store.set_cache_split(DEFAULT_NODE_CACHE_BYTES, 0);
        Ok(store)
    }

    fn write_meta(&self) -> Result<(), ENNError> {
        let meta = O1Meta {
            num_rows: self.x.nrows,
            has_yvar: self.yvar.is_some(),
            x_scale: self.x_scale.as_ref().map(|s| s.to_vec()),
            hash_scale: Some(self.index.hash_scale().to_vec()),
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
        self.index.bits()
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
        self.write_meta()?;
        self.index.insert_rows(&self.x, start)
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
        if rebuild || self.index.wants_first_metric() {
            self.index.rehash(&self.x, &x_scale.to_vec())?;
        }
        self.write_meta()
    }

    fn metric_weights(&self) -> Vec<f64> {
        match &self.x_scale {
            Some(s) => s.iter().map(|v| 1.0 / (v * v)).collect(),
            None => vec![1.0; self.num_dim()],
        }
    }

    pub fn search(
        &self,
        x: &ArrayView2<f64>,
        search_k: usize,
        exclude_nearest: bool,
    ) -> Result<(Array2<f64>, Array2<i64>), ENNError> {
        let weights = self.metric_weights();
        let ctx = SearchContext {
            index: &self.index,
            rows: self.index.rows(&self.x),
            weights: &weights,
            cache: self.cache.as_deref(),
            profile: &self.profile,
            overlap: x.nrows() == 1,
        };
        ctx.search(x, search_k, exclude_nearest)
    }

    pub fn candidates(&self, x: &ArrayView2<f64>, search_k: usize) -> Result<Vec<Vec<u32>>, ENNError> {
        let weights = self.metric_weights();
        let ctx = SearchContext {
            index: &self.index,
            rows: self.index.rows(&self.x),
            weights: &weights,
            cache: self.cache.as_deref(),
            profile: &self.profile,
            overlap: true,
        };
        x.rows()
            .into_iter()
            .map(|q| ctx.candidates(&q.to_vec(), search_k.min(self.len())))
            .collect()
    }

    pub fn occupancy(&self) -> Result<O1Occupancy, ENNError> {
        self.index.occupancy()
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

    pub fn node_io(&self) -> Option<(u64, u64)> {
        self.index.node_io()
    }

    pub fn set_probe_budget(&mut self, max_leaves: usize, max_candidates: usize) -> Result<(), ENNError> {
        self.index.set_budget(max_leaves, max_candidates)
    }

    pub fn set_io_width(&mut self, width: usize, slow_ns: u64) -> Result<(), ENNError> {
        self.index.set_io_width(width, slow_ns)
    }

    pub fn set_split_candidates(&mut self, candidates: u32) -> Result<(), ENNError> {
        self.index.set_split(None, Some(candidates), &self.x)
    }

    pub fn set_split_rule(&mut self, rule: SplitRule) -> Result<(), ENNError> {
        self.index.set_split(Some(rule), None, &self.x)
    }

    pub fn set_probe_rule(&mut self, rule: ProbeRule, select: SelectRule, stop: (f64, u32)) -> Result<(), ENNError> {
        self.index.set_probe_rule(rule, select, stop)
    }

    pub fn set_repair(&mut self, repair_rows: usize, stale_drift: f64, prefer_fresh: bool) -> Result<(), ENNError> {
        self.index.set_repair(repair_rows, stale_drift, prefer_fresh)
    }

    pub fn maintain_index(&mut self, max_rows: usize) -> Result<usize, ENNError> {
        self.index.maintain(&self.x, max_rows)
    }

    pub fn compact_index(&mut self) -> Result<usize, ENNError> {
        self.index.compact()
    }

    pub fn index_memory_bytes(&self) -> usize {
        self.index.memory_bytes()
    }
}
