use std::path::Path;

use super::cache::SharedCache;
use super::disk_index::{LshIndex, O1Occupancy};
use super::forest::{ForestIndex, ForestParams, MaintStats};
use super::forest_rules::{ProbeRule, SelectRule, SplitRule};
use super::lsh::O1Params;
use super::profile::QueryProfile;
use super::row_order::Rows;
use crate::backend::MmapColumnStore;
use crate::error::ENNError;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum O1IndexParams {
    Lsh(O1Params),
    Forest(ForestParams),
}

pub(crate) enum O1Index {
    Lsh(Box<LshIndex>),
    Forest(Box<ForestIndex>),
}

impl O1Index {
    pub(crate) fn open(
        work_dir: &Path,
        seed: u64,
        params: O1IndexParams,
        fallback_scale: &[f64],
        rows: &MmapColumnStore,
    ) -> Result<Self, ENNError> {
        Ok(match params {
            O1IndexParams::Lsh(p) => Self::Lsh(Box::new(LshIndex::open(work_dir, seed, p, fallback_scale, rows)?)),
            O1IndexParams::Forest(p) => Self::Forest(Box::new(ForestIndex::open(work_dir, seed, p, fallback_scale, rows)?)),
        })
    }

    pub(crate) fn insert_rows(&mut self, rows: &MmapColumnStore, start: usize) -> Result<(), ENNError> {
        match self {
            Self::Lsh(i) => i.insert_rows(rows, start),
            Self::Forest(f) => f.insert_rows(rows, start),
        }
    }

    pub(crate) fn rehash(&mut self, rows: &MmapColumnStore, hash_scale: &[f64]) -> Result<(), ENNError> {
        match self {
            Self::Lsh(i) => i.rehash(rows, hash_scale),
            Self::Forest(f) => f.rehash(rows, hash_scale),
        }
    }

    pub(crate) fn bits(&self) -> usize {
        match self {
            Self::Lsh(i) => i.bits(),
            Self::Forest(_) => 0,
        }
    }

    pub(crate) fn wants_first_metric(&self) -> bool {
        match self {
            Self::Lsh(i) => i.bits() == 0,
            Self::Forest(f) => f.is_trivial(),
        }
    }

    pub(crate) fn hash_scale(&self) -> &[f64] {
        match self {
            Self::Lsh(i) => i.hash_scale(),
            Self::Forest(f) => f.hash_scale(),
        }
    }

    pub(crate) fn memory_bytes(&self) -> usize {
        match self {
            Self::Lsh(i) => i.memory_bytes(),
            Self::Forest(f) => f.memory_bytes(),
        }
    }

    pub(crate) fn occupancy(&self) -> Result<O1Occupancy, ENNError> {
        match self {
            Self::Lsh(i) => i.occupancy(),
            Self::Forest(f) => f.occupancy(),
        }
    }

    #[cfg(test)]
    pub(crate) fn candidate_ids(&self, query: &[f64], prof: &QueryProfile, rows: &MmapColumnStore) -> Result<Vec<u32>, ENNError> {
        self.candidate_ids_at(query, prof, self.rows(rows), true)
    }

    pub(crate) fn candidate_ids_at(&self, query: &[f64], prof: &QueryProfile, rows: Rows<'_>, overlap: bool) -> Result<Vec<u32>, ENNError> {
        match self {
            Self::Lsh(i) => i.candidate_ids(query),
            Self::Forest(f) => f.candidate_ids_at(query, prof, rows, overlap),
        }
    }

    pub(crate) fn scored_at(&self, query: &[f64], prof: &QueryProfile, rows: Rows<'_>, overlap: bool, weights: &[f64], need: usize) -> Result<Option<super::forest_query::Scored>, ENNError> {
        match self {
            Self::Lsh(_) => Ok(None),
            Self::Forest(f) => f.scored_at(query, prof, rows, overlap, weights, need),
        }
    }

    pub(crate) fn set_cache(&mut self, cache: SharedCache) {
        if let Self::Forest(f) = self {
            f.set_cache(cache);
        }
    }

    pub(crate) fn set_pread_pivots(&mut self, on: bool) -> Result<(), ENNError> {
        match self {
            Self::Lsh(_) => Err(ENNError::InvalidParameter("the O1 LSH index has no pivot reads".into())),
            Self::Forest(f) => {
                f.set_pread_pivots(on);
                Ok(())
            }
        }
    }

    pub(crate) fn io_width(&self) -> (usize, u64) {
        match self {
            Self::Lsh(_) => (1, 0),
            Self::Forest(f) => (f.params.io_width, f.params.io_slow_ns),
        }
    }

    pub(crate) fn maint_stats(&self) -> Option<MaintStats> {
        match self {
            Self::Lsh(_) => None,
            Self::Forest(f) => Some(f.maint_stats()),
        }
    }

    pub(crate) fn node_io(&self) -> Option<(u64, u64)> {
        match self {
            Self::Lsh(_) => None,
            Self::Forest(f) => Some(f.node_io()),
        }
    }

    pub(crate) fn set_budget(&mut self, max_leaves: usize, max_candidates: usize) -> Result<(), ENNError> {
        match self {
            Self::Lsh(_) => Err(ENNError::InvalidParameter("the O1 LSH index has no probe budget setter".into())),
            Self::Forest(f) => {
                f.set_budget(max_leaves, max_candidates);
                Ok(())
            }
        }
    }

    pub(crate) fn set_io_width(&mut self, width: usize, slow_ns: u64) -> Result<(), ENNError> {
        match self {
            Self::Lsh(_) => Err(ENNError::InvalidParameter("the O1 LSH index has no I/O width".into())),
            Self::Forest(f) => {
                f.set_io_width(width, slow_ns);
                Ok(())
            }
        }
    }

    pub(crate) fn set_trees(&mut self, n: usize, rows: &MmapColumnStore) -> Result<(), ENNError> {
        match self {
            Self::Lsh(_) => Err(ENNError::InvalidParameter("the O1 LSH index has no trees".into())),
            Self::Forest(f) => f.set_trees(n, rows),
        }
    }

    pub(crate) fn set_split(&mut self, rule: Option<SplitRule>, candidates: Option<u32>, rows: &MmapColumnStore) -> Result<(), ENNError> {
        match self {
            Self::Lsh(_) => Err(ENNError::InvalidParameter("the O1 LSH index has no split rule".into())),
            Self::Forest(f) => f.set_split(rule, candidates, rows),
        }
    }

    pub(crate) fn set_probe_rule(&mut self, rule: ProbeRule, select: SelectRule, stop: (f64, u32)) -> Result<(), ENNError> {
        match self {
            Self::Lsh(_) => Err(ENNError::InvalidParameter("the O1 LSH index has no probe rule".into())),
            Self::Forest(f) => {
                f.set_probe_rule(rule, select, stop);
                Ok(())
            }
        }
    }

    pub(crate) fn set_staging(&mut self, above: u32, work: u32) -> Result<(), ENNError> {
        match self {
            Self::Lsh(_) => Err(ENNError::InvalidParameter("the O1 LSH index has no staged rebuilds".into())),
            Self::Forest(f) => {
                f.set_staging(above, work);
                Ok(())
            }
        }
    }

    pub(crate) fn set_lean(&mut self, lean: u32) -> Result<(), ENNError> {
        match self {
            Self::Lsh(_) => Err(ENNError::InvalidParameter("the O1 LSH index has no split lean".into())),
            Self::Forest(f) => f.set_lean(lean),
        }
    }

    pub(crate) fn set_repair(&mut self, repair_rows: usize, stale_drift: f64, prefer_fresh: bool) -> Result<(), ENNError> {
        match self {
            Self::Lsh(_) => Err(ENNError::InvalidParameter("the O1 LSH index has no repair budget".into())),
            Self::Forest(f) => {
                f.set_repair(repair_rows, stale_drift, prefer_fresh);
                Ok(())
            }
        }
    }

    pub(crate) fn compact(&mut self) -> Result<usize, ENNError> {
        match self {
            Self::Lsh(_) => Ok(0),
            Self::Forest(f) => f.compact(),
        }
    }

    pub(crate) fn maintain(&mut self, rows: &MmapColumnStore, max_rows: usize) -> Result<usize, ENNError> {
        match self {
            Self::Lsh(_) => Ok(0),
            Self::Forest(f) => f.maintain_now(rows, max_rows),
        }
    }

    #[cfg(test)]
    pub(crate) fn lsh_mut(&mut self) -> &mut LshIndex {
        match self {
            Self::Lsh(i) => i,
            Self::Forest(_) => panic!("not an LSH index"),
        }
    }

    #[cfg(test)]
    pub(crate) fn lsh(&self) -> &LshIndex {
        match self {
            Self::Lsh(i) => i,
            Self::Forest(_) => panic!("not an LSH index"),
        }
    }

    #[cfg(test)]
    pub(crate) fn forest_mut(&mut self) -> &mut ForestIndex {
        match self {
            Self::Forest(f) => f,
            Self::Lsh(_) => panic!("not a forest index"),
        }
    }

    #[cfg(test)]
    pub(crate) fn forest(&self) -> &ForestIndex {
        match self {
            Self::Forest(f) => f,
            Self::Lsh(_) => panic!("not a forest index"),
        }
    }
}
