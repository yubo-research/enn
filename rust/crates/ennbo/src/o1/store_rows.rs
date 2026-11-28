use std::sync::Arc;

use ndarray::{Array1, Array2};

use super::cache::{CacheStats, ObjCache};
use super::forest::MaintStats;
use super::profile::ProfileStats;
use super::row_order::Rows;
use super::search::{fetch_row, X_COLUMN};
use super::O1Store;
use crate::backend::MmapColumnStore;
use crate::error::ENNError;

const Y_COLUMN: u8 = 1;
const YVAR_COLUMN: u8 = 2;

impl O1Store {
    fn cached_row(
        &self,
        store: &MmapColumnStore,
        column: u8,
        i: usize,
    ) -> Result<Array1<f64>, ENNError> {
        let rows = if column == X_COLUMN {
            self.index.rows(store)
        } else {
            Rows::plain(store)
        };
        Ok(Array1::from(
            fetch_row(rows, self.cache.as_deref(), column, i)?.to_vec(),
        ))
    }

    pub fn row_x(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        self.cached_row(&self.x, X_COLUMN, i)
    }

    pub fn row_y(&self, i: usize) -> Result<Array1<f64>, ENNError> {
        self.cached_row(&self.y, Y_COLUMN, i)
    }

    pub fn row_yvar(&self, i: usize) -> Result<Option<Array1<f64>>, ENNError> {
        self.yvar
            .as_ref()
            .map(|s| self.cached_row(s, YVAR_COLUMN, i))
            .transpose()
    }

    pub fn set_cache_budget(&mut self, budget_bytes: usize) {
        self.cache = (budget_bytes > 0).then(|| Arc::new(ObjCache::new(budget_bytes)));
        self.index.set_cache(self.cache.clone());
    }

    pub fn set_cache_split(&mut self, node_bytes: usize, row_bytes: usize) {
        self.cache = Some(Arc::new(ObjCache::split(node_bytes, row_bytes)));
        self.index.set_cache(self.cache.clone());
    }

    pub fn cache_stats(&self) -> Option<CacheStats> {
        self.cache.as_ref().map(|c| c.stats())
    }

    pub fn set_profile(&mut self, enabled: bool) {
        self.profile.set_enabled(enabled);
    }

    pub fn take_profile(&mut self) -> ProfileStats {
        self.profile.take()
    }

    pub fn maint_stats(&self) -> Option<MaintStats> {
        self.index.maint_stats()
    }

    pub fn set_pread_pivots(&mut self, on: bool) -> Result<(), ENNError> {
        self.index.set_pread_pivots(on)
    }

    pub fn set_row_order(&mut self, params: super::OrderParams) -> Result<(), ENNError> {
        self.index.set_row_order(params)
    }

    pub fn set_par_rebuild(&mut self, above: u32) -> Result<(), ENNError> {
        self.index.set_par_rebuild(above)
    }

    pub fn set_rerank(&mut self, rerank: u32) -> Result<(), ENNError> {
        self.index.set_rerank(rerank)
    }

    pub fn set_trees(&mut self, n: usize) -> Result<(), ENNError> {
        self.index.set_trees(n, &self.x)
    }

    pub fn order_rows(&mut self) -> Result<usize, ENNError> {
        self.index.order_now(&self.x)
    }
}

pub fn scattered_rows(
    store: &O1Store,
    indices: &[usize],
) -> Result<(Array2<f64>, Array2<f64>), ENNError> {
    Ok((
        store.x.pread_gather(indices)?,
        store.y.pread_gather(indices)?,
    ))
}

pub fn y_rows(store: &O1Store, start: usize, end: usize) -> Result<Array2<f64>, ENNError> {
    store.y.mmap_row_range(start, end)
}
