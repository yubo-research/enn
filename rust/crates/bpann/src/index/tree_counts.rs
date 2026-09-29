//! Per-page state of the incremental tree that is not stored in its pages.

use crate::distance::l2_sq_f32;
use crate::error::BpannError;
use crate::index::build::BpannIndex;
use crate::index::page::Page;

pub fn dist(a: &[f32], b: &[f32]) -> f32 {
    l2_sq_f32(a, b).sqrt()
}

/// Per-page subtree row counts (running-mean weights), a contiguous `f32` block per
/// page (a leaf's row coordinates in `row_ids` order, or an internal page's child
/// centroids in `child_page_ids` order), and a radius per page: an upper bound on the
/// distance from the page's centroid (its entry in the parent) to any row below it.
/// All are indexed by page id; a missing radius is infinite.
#[derive(Clone, Debug, Default)]
pub struct TreeCounts {
    counts: Vec<usize>,
    blocks: Vec<Vec<f32>>,
    radii: Vec<f32>,
    rows_cached: bool,
}

impl TreeCounts {
    pub fn count(&self, page_id: u32) -> usize {
        self.counts.get(page_id as usize).copied().unwrap_or(0)
    }

    pub(crate) fn set(&mut self, page_id: u32, n: usize) {
        let i = page_id as usize;
        if i >= self.counts.len() {
            self.counts.resize(i + 1, 0);
        }
        self.counts[i] = n;
    }

    pub(crate) fn bump(&mut self, page_id: u32) -> usize {
        let n = self.count(page_id) + 1;
        self.set(page_id, n);
        n
    }

    pub(crate) fn alloc(&mut self) -> u32 {
        self.counts.push(0);
        (self.counts.len() - 1) as u32
    }

    pub fn radius(&self, page_id: u32) -> f32 {
        self.radii.get(page_id as usize).copied().unwrap_or(f32::INFINITY)
    }

    pub(crate) fn set_radius(&mut self, page_id: u32, r: f32) {
        let i = page_id as usize;
        if i >= self.radii.len() {
            self.radii.resize(i + 1, f32::INFINITY);
        }
        self.radii[i] = r;
    }

    /// Whether every page's block is filled (false after [`Self::from_index`]).
    pub fn rows_cached(&self) -> bool {
        self.rows_cached
    }

    pub(crate) fn mark_rows_cached(&mut self) {
        self.rows_cached = true;
    }

    /// Block of page `page_id` (`entries * num_dim` values).
    pub fn block(&self, page_id: u32) -> &[f32] {
        self.blocks.get(page_id as usize).map_or(&[], Vec::as_slice)
    }

    pub(crate) fn block_mut(&mut self, page_id: u32) -> &mut Vec<f32> {
        let i = page_id as usize;
        if i >= self.blocks.len() {
            self.blocks.resize_with(i + 1, Vec::new);
        }
        &mut self.blocks[i]
    }

    pub(crate) fn set_centroid_block(&mut self, page_id: u32, centroids: &[Vec<f32>]) {
        *self.block_mut(page_id) = centroids.concat();
    }

    /// Fill every page block, reading leaf rows with `coords(row)`, and recompute
    /// every radius; used after adopting a persisted tree.
    pub fn fill_blocks(
        &mut self,
        index: &BpannIndex,
        mut coords: impl FnMut(u32, &mut Vec<f32>) -> Result<(), BpannError>,
    ) -> Result<(), BpannError> {
        let mut v = Vec::new();
        for page in &index.pages {
            match page {
                Page::Leaf { page_id, row_ids, .. } => {
                    let mut block = Vec::with_capacity(row_ids.len() * index.header.num_dim);
                    for &row in row_ids {
                        coords(row, &mut v)?;
                        block.extend_from_slice(&v);
                    }
                    *self.block_mut(*page_id) = block;
                }
                Page::Internal { page_id, centroids, .. } => self.set_centroid_block(*page_id, centroids),
            }
        }
        let root = index.header.root_page_id;
        if let Some(page) = index.page_by_id(root) {
            let r = self.subtree_radius(index, root, &page.centroid(), 0);
            self.set_radius(root, r);
        }
        self.rows_cached = true;
        Ok(())
    }

    fn subtree_radius(&mut self, index: &BpannIndex, page_id: u32, centroid: &[f32], depth: usize) -> f32 {
        match index.page_by_id(page_id) {
            Some(Page::Leaf { .. }) if !centroid.is_empty() => self
                .block(page_id)
                .chunks_exact(centroid.len())
                .map(|row| dist(row, centroid))
                .fold(0.0, f32::max),
            Some(Page::Internal {
                centroids,
                child_page_ids,
                ..
            }) if depth < 64 => {
                let mut r = 0.0f32;
                for (c, &child) in centroids.iter().zip(child_page_ids) {
                    let rc = self.subtree_radius(index, child, c, depth + 1);
                    self.set_radius(child, rc);
                    r = r.max(dist(c, centroid) + rc);
                }
                r
            }
            _ => f32::INFINITY,
        }
    }

    /// Copy the running-mean centroids from the blocks into the pages: each internal
    /// page's child centroids and each leaf's stored centroid (the mean of its rows for
    /// a lone root leaf). Does nothing until the blocks are filled.
    pub fn write_page_centroids(&self, index: &mut BpannIndex) {
        if !self.rows_cached {
            return;
        }
        let dim = index.header.num_dim.max(1);
        let mut leaf_centroids: Vec<(u32, Vec<f32>)> = Vec::new();
        for page in &mut index.pages {
            if let Page::Internal {
                page_id,
                centroids,
                child_page_ids,
            } = page
            {
                let block = self.block(*page_id);
                for ((c, &child), b) in centroids.iter_mut().zip(child_page_ids.iter()).zip(block.chunks_exact(dim)) {
                    c.copy_from_slice(b);
                    leaf_centroids.push((child, b.to_vec()));
                }
            }
        }
        let root = index.header.root_page_id;
        for page in &mut index.pages {
            if let Page::Leaf {
                page_id,
                stored_centroid,
                ..
            } = page
            {
                if *page_id == root {
                    let rows: Vec<&[f32]> = self.block(root).chunks_exact(dim).collect();
                    let all: Vec<usize> = (0..rows.len()).collect();
                    *stored_centroid = Some(crate::index::split::weighted_mean(&rows, &vec![1.0; rows.len()], &all));
                }
            }
        }
        for (leaf, c) in leaf_centroids {
            if let Some(Page::Leaf { stored_centroid, .. }) = index.page_by_id_mut(leaf) {
                *stored_centroid = Some(c);
            }
        }
    }

    /// Multiply every cached coordinate by `ratio` (one factor per dimension); radii
    /// grow by the largest factor so they stay upper bounds.
    pub fn scale_rows(&mut self, ratio: &[f64]) {
        let dim = ratio.len().max(1);
        for row in self.blocks.iter_mut().flat_map(|b| b.chunks_mut(dim)) {
            for (v, &r) in row.iter_mut().zip(ratio) {
                *v = (f64::from(*v) * r) as f32;
            }
        }
        let grow = ratio.iter().copied().fold(0.0f64, f64::max) as f32;
        self.radii.iter_mut().for_each(|r| *r *= grow);
    }

    /// Rebuild counts for a persisted tree. `None` if it is not a row-id tree
    /// (e.g. an index written by an older fragment layout). Rows are not cached.
    pub fn from_index(index: &BpannIndex) -> Option<Self> {
        let mut out = Self {
            counts: vec![usize::MAX; index.pages.iter().map(Page::page_id).max()? as usize + 1],
            ..Self::default()
        };
        let total = out.count_subtree(index, index.header.root_page_id, 0)?;
        let seen = out.counts.iter().filter(|&&n| n != usize::MAX).count();
        (total == index.header.indexed_rows && seen == index.pages.len()).then(|| {
            out.counts.iter_mut().filter(|n| **n == usize::MAX).for_each(|n| *n = 0);
            out
        })
    }

    fn count_subtree(&mut self, index: &BpannIndex, page_id: u32, depth: usize) -> Option<usize> {
        if depth > 64 || self.counts.get(page_id as usize) != Some(&usize::MAX) {
            return None;
        }
        let n = match index.page_by_id(page_id)? {
            Page::Leaf {
                row_ids,
                row_range: None,
                vectors,
                stored_centroid: Some(_),
                ..
            } if vectors.is_empty() => row_ids.len(),
            Page::Internal { child_page_ids, .. } if !child_page_ids.is_empty() => {
                let mut sum = 0;
                for &child in child_page_ids {
                    sum += self.count_subtree(index, child, depth + 1)?;
                }
                sum
            }
            _ => return None,
        };
        self.set(page_id, n);
        Some(n)
    }
}
