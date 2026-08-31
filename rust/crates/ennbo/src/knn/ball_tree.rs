//! In-memory FastMem KNN: data-append on add (cheap segment_s), Faiss Flat for mid-N
//! search, AABB ball tree for large-N search.
//!
//! MuyGPyS builds a neighbor index when querying. Adds only append into `data`;
//! mid-N search lazily catches Faiss up, and large-N search builds the AABB tree.

use ndarray::{Array2, ArrayView2};
use rayon::prelude::*;

use super::{pad_neighbor_cols_to_search_k, FaissBackend};
use crate::index::{IndexDriver, IndexError};

pub(super) const LEAF_SIZE: usize = 64;
#[cfg(not(test))]
pub(super) const AABB_MODE_MIN_N: usize = 5000;
#[cfg(test)]
pub(super) const AABB_MODE_MIN_N: usize = 500;
/// Below this N, FastMem searches via Faiss Flat; at/above, AABB/ball tree.
#[cfg(not(test))]
pub(super) const TREE_SEARCH_MIN_N: usize = 250_000;
#[cfg(test)]
pub(super) const TREE_SEARCH_MIN_N: usize = 100;
/// If the uncovered tail somehow grows past this, rebuild on search.
#[cfg(not(test))]
pub(super) const TAIL_REBUILD_MAX: usize = 50_000;
#[cfg(test)]
pub(super) const TAIL_REBUILD_MAX: usize = 40;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum SearchMode {
    Brute,
    Ball,
    Aabb,
}

#[derive(Clone)]
pub(crate) struct BallNode {
    pub(crate) center: Vec<f64>,
    pub(crate) radius: f64,
    pub(crate) bbox_min: Vec<f64>,
    pub(crate) bbox_max: Vec<f64>,
    pub(crate) left: usize,
    pub(crate) right: usize,
    /// Index into `leaves` when this is a leaf; `usize::MAX` otherwise.
    pub(crate) leaf_idx: usize,
    /// Start row in `leaf_pack` (`leaf_pack_start * num_dim` is the float offset).
    pub(crate) leaf_pack_start: usize,
    pub(crate) leaf_pack_len: usize,
}

pub(crate) struct BallTreeBackend {
    pub(super) num_dim: usize,
    pub(super) data: Vec<f64>,
    pub(super) n: usize,
    /// Points `[0, tree_n)` are covered by `nodes`; `[tree_n, n)` are a brute-force tail.
    pub(super) tree_n: usize,
    pub(crate) nodes: Vec<BallNode>,
    pub(crate) leaves: Vec<Vec<usize>>,
    /// Contiguous leaf-id buffer for kiss/source compatibility with older leaf layouts.
    pub(crate) leaf_ids: Vec<usize>,
    /// Leaf-packed coordinates in DFS leaf order for cache-friendly search.
    pub(crate) leaf_pack: Vec<f64>,
    pub(crate) root: usize,
    pub(super) search_mode: SearchMode,
    pub(super) faiss_flat: Option<FaissBackend>,
    pub(super) tree_pending: bool,
}

impl BallTreeBackend {
    pub(crate) fn new(num_dim: usize, train_scaled: &ArrayView2<f64>) -> Result<Self, IndexError> {
        let mut backend = Self {
            num_dim,
            data: Vec::new(),
            n: 0,
            tree_n: 0,
            nodes: Vec::new(),
            leaves: Vec::new(),
            leaf_ids: Vec::new(),
            leaf_pack: Vec::new(),
            root: 0,
            search_mode: SearchMode::Brute,
            faiss_flat: None,
            tree_pending: false,
        };
        backend.rebuild(train_scaled)?;
        Ok(backend)
    }

    pub(crate) fn len(&self) -> usize {
        self.n
    }

    pub(crate) fn memory_usage_bytes(&self) -> usize {
        let faiss_bytes = self
            .faiss_flat
            .as_ref()
            .map(|f| f.memory_usage_bytes())
            .unwrap_or(0);
        let leaf_bytes: usize = self
            .leaves
            .iter()
            .map(|v| v.capacity() * std::mem::size_of::<usize>())
            .sum();
        self.data.len() * std::mem::size_of::<f64>()
            + self.leaf_ids.len() * std::mem::size_of::<usize>()
            + self.leaf_pack.len() * std::mem::size_of::<f64>()
            + leaf_bytes
            + self.nodes.capacity() * std::mem::size_of::<BallNode>()
            + faiss_bytes
    }

    pub(crate) fn rebuild(&mut self, train_scaled: &ArrayView2<f64>) -> Result<(), IndexError> {
        if train_scaled.ncols() != self.num_dim {
            return Err(IndexError::InvalidShape {
                expected: self.num_dim,
                got: train_scaled.ncols(),
            });
        }
        self.n = train_scaled.nrows();
        self.data.clear();
        self.data.reserve(self.n * self.num_dim);
        for row in train_scaled.rows() {
            self.data.extend(row.iter().copied());
        }
        self.nodes.clear();
        self.leaves.clear();
        self.leaf_ids.clear();
        self.leaf_pack.clear();
        self.root = 0;
        self.tree_n = 0;
        self.faiss_flat = None;
        self.tree_pending = false;
        if self.n == 0 {
            self.search_mode = SearchMode::Brute;
            return Ok(());
        }
        self.search_mode = SearchMode::Brute;
        if self.n < TREE_SEARCH_MIN_N {
            let train = Array2::from_shape_vec((self.n, self.num_dim), self.data.clone())
                .expect("data length matches n * dim");
            self.data.clear();
            self.faiss_flat = Some(FaissBackend::new(
                self.num_dim,
                IndexDriver::Exact,
                &train.view(),
            )?);
            self.tree_pending = false;
        } else {
            self.tree_pending = true;
        }
        Ok(())
    }

    /// Below [`TREE_SEARCH_MIN_N`]: Faiss Flat (sync cost); at/above: data append, tree on search.
    pub(crate) fn add(
        &mut self,
        rows_scaled: &ArrayView2<f64>,
        start_key: u64,
    ) -> Result<(), IndexError> {
        if rows_scaled.ncols() != self.num_dim {
            return Err(IndexError::InvalidShape {
                expected: self.num_dim,
                got: rows_scaled.ncols(),
            });
        }
        if rows_scaled.nrows() == 0 {
            return Ok(());
        }
        let old_n = self.n;
        let new_n = old_n + rows_scaled.nrows();
        if new_n < TREE_SEARCH_MIN_N {
            self.n = new_n;
            if let Some(faiss) = self.faiss_flat.as_mut() {
                faiss.add(rows_scaled, start_key)?;
                return Ok(());
            }
            self.search_mode = SearchMode::Brute;
            self.tree_pending = false;
            self.tree_n = 0;
            self.faiss_flat = Some(FaissBackend::new(
                self.num_dim,
                IndexDriver::Exact,
                rows_scaled,
            )?);
            return Ok(());
        }
        if self.data.len() < old_n * self.num_dim {
            self.materialize_data_from_faiss()?;
        }
        self.data.reserve(rows_scaled.nrows() * self.num_dim);
        for row in rows_scaled.rows() {
            self.data.extend(row.iter().copied());
        }
        self.n = new_n;
        self.faiss_flat = None;
        self.tree_pending = true;
        self.search_mode = SearchMode::Brute;
        Ok(())
    }

    fn materialize_data_from_faiss(&mut self) -> Result<(), IndexError> {
        let Some(faiss) = self.faiss_flat.as_ref() else {
            return Ok(());
        };
        let n = faiss.len();
        if n == 0 {
            return Ok(());
        }
        let mut buf = vec![0.0_f32; n * self.num_dim];
        faiss.reconstruct_all(&mut buf)?;
        self.data.clear();
        self.data.reserve(n * self.num_dim);
        self.data.extend(buf.into_iter().map(|v| v as f64));
        Ok(())
    }

    fn ensure_faiss_caught_up(&mut self) -> Result<(), IndexError> {
        let faiss_n = self.faiss_flat.as_ref().map(|f| f.len()).unwrap_or(0);
        if faiss_n >= self.n {
            return Ok(());
        }
        if self.data.len() >= self.n * self.num_dim {
            let pending = Array2::from_shape_vec(
                (self.n - faiss_n, self.num_dim),
                self.data[faiss_n * self.num_dim..self.n * self.num_dim].to_vec(),
            )
            .expect("pending rows match shape");
            if let Some(faiss) = self.faiss_flat.as_mut() {
                faiss.add(&pending.view(), faiss_n as u64)?;
            } else {
                self.faiss_flat = Some(FaissBackend::new(
                    self.num_dim,
                    IndexDriver::Exact,
                    &pending.view(),
                )?);
            }
            return Ok(());
        }
        if self.faiss_flat.is_none() {
            return Err(IndexError::InvalidParameter(
                "fast_mem mid-N search without Faiss or data".into(),
            ));
        }
        Ok(())
    }

    /// Mid-N: catch Faiss up from `data` then search. Large-N: build AABB from `data`.
    pub(crate) fn search(
        &mut self,
        queries_scaled: &ArrayView2<f64>,
        k_eff: usize,
        search_k: usize,
    ) -> Result<(Array2<f64>, Array2<i64>), IndexError> {
        if queries_scaled.ncols() != self.num_dim {
            return Err(IndexError::InvalidShape {
                expected: self.num_dim,
                got: queries_scaled.ncols(),
            });
        }
        let n_query = queries_scaled.nrows();
        if self.n == 0 || k_eff == 0 {
            return Ok((
                Array2::from_elem((n_query, search_k), f64::INFINITY),
                Array2::zeros((n_query, search_k)),
            ));
        }
        if self.n >= TREE_SEARCH_MIN_N {
            let need_tree = self.tree_pending
                || self.nodes.is_empty()
                || self.tree_n == 0
                || self.n.saturating_sub(self.tree_n) > TAIL_REBUILD_MAX;
            if need_tree {
                self.build_tree_now();
            }
        } else {
            self.search_mode = SearchMode::Brute;
            self.ensure_faiss_caught_up()?;
            return self
                .faiss_flat
                .as_mut()
                .expect("faiss mid-N")
                .search(queries_scaled, k_eff, search_k);
        }
        let mut dist2s = Array2::from_elem((n_query, k_eff), f64::INFINITY);
        let mut indices = Array2::zeros((n_query, k_eff));
        const PARALLEL_MIN_N: usize = 128;
        let use_parallel =
            self.n >= PARALLEL_MIN_N && n_query > 1 && rayon::current_num_threads() > 1;
        if use_parallel {
            let hits: Vec<Vec<(f64, usize)>> = (0..n_query)
                .into_par_iter()
                .map(|qi| {
                    let q = queries_scaled.row(qi);
                    self.search_one(q.as_slice().unwrap(), k_eff)
                })
                .collect();
            for (qi, row_hits) in hits.into_iter().enumerate() {
                for (j, (dist2, id)) in row_hits.into_iter().enumerate() {
                    dist2s[[qi, j]] = dist2;
                    indices[[qi, j]] = id as i64;
                }
            }
        } else {
            for qi in 0..n_query {
                let q = queries_scaled.row(qi);
                let hits = self.search_one(q.as_slice().unwrap(), k_eff);
                for (j, (dist2, id)) in hits.into_iter().enumerate() {
                    dist2s[[qi, j]] = dist2;
                    indices[[qi, j]] = id as i64;
                }
            }
        }
        let out = pad_neighbor_cols_to_search_k(dist2s, indices, search_k);
        #[cfg(not(test))]
        {
            self.nodes.clear();
            self.leaves.clear();
            self.leaf_ids.clear();
            self.leaf_pack.clear();
            self.tree_n = 0;
            self.tree_pending = true;
            self.root = 0;
        }
        Ok(out)
    }

    pub(crate) fn point(&self, id: usize) -> &[f64] {
        let start = id * self.num_dim;
        &self.data[start..start + self.num_dim]
    }

    #[inline]
    pub(crate) fn dist2(a: &[f64], b: &[f64]) -> f64 {
        let mut s = 0.0_f64;
        for (x, y) in a.iter().zip(b.iter()) {
            let d = x - y;
            s += d * d;
        }
        s
    }

    #[inline]
    pub(crate) fn aabb_dist2(query: &[f64], bbox_min: &[f64], bbox_max: &[f64]) -> f64 {
        let mut s = 0.0_f64;
        for i in 0..query.len() {
            let q = query[i];
            let v = if q < bbox_min[i] {
                bbox_min[i] - q
            } else if q > bbox_max[i] {
                q - bbox_max[i]
            } else {
                0.0
            };
            s += v * v;
        }
        s
    }

    fn search_one(&self, query: &[f64], k: usize) -> Vec<(f64, usize)> {
        match self.search_mode {
            SearchMode::Brute => self.search_one_brute(query, k),
            SearchMode::Ball => self.search_one_ball(query, k),
            SearchMode::Aabb => self.search_one_aabb(query, k),
        }
    }
}

#[cfg(test)]
#[path = "ball_tree_tests.rs"]
mod ball_tree_tests;
