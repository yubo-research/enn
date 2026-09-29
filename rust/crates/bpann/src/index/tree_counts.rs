//! Derived per-page state of the incremental tree: leaf blocks, radii and subtree
//! counts, recomputed when a tree is read back from disk or its metric changes.

use crate::distance::l2_sq_f32;
use crate::error::BpannError;
use crate::index::split::weighted_mean;
use crate::index::tree::Tree;

/// Recursion limit for a tree read from disk (a grown tree is far shallower).
const MAX_DEPTH: usize = 64;

pub fn dist(a: &[f32], b: &[f32]) -> f32 {
    l2_sq_f32(a, b).sqrt()
}

impl Tree {
    /// Fill every leaf block, reading leaf rows with `coords(row)`, and recompute
    /// every radius; used after adopting a persisted tree.
    pub fn fill_blocks(
        &mut self,
        mut coords: impl FnMut(u32, &mut Vec<f32>) -> Result<(), BpannError>,
    ) -> Result<(), BpannError> {
        let (mut v, mut block) = (Vec::new(), Vec::new());
        for page in 0..self.num_pages() as u32 {
            if !self.store.is_leaf(page) {
                continue;
            }
            let ids = self.store.ids(page).to_vec();
            block.clear();
            for &row in &ids {
                coords(row, &mut v)?;
                block.extend_from_slice(&v);
            }
            self.store.set_entries(page, &ids, &block);
        }
        if self.num_pages() > 0 {
            let root = self.header.root_page_id;
            let c = self.page_centroid(root);
            let r = self.subtree_radius(root, &c, 0);
            self.store.set_radius(root, r);
        }
        self.rows_cached = true;
        Ok(())
    }

    /// Mean of a page's rows (a leaf) or count-weighted mean of its child centroids.
    pub fn page_centroid(&self, page: u32) -> Vec<f32> {
        let s = &self.store;
        let points: Vec<&[f32]> = s.block(page).chunks_exact(s.num_dim()).collect();
        let weights: Vec<f64> = if s.is_leaf(page) {
            vec![1.0; points.len()]
        } else {
            s.ids(page).iter().map(|&k| s.count(k) as f64).collect()
        };
        weighted_mean(&points, &weights, &(0..points.len()).collect::<Vec<_>>())
    }

    fn subtree_radius(&mut self, page: u32, centroid: &[f32], depth: usize) -> f32 {
        if self.store.is_leaf(page) {
            let d = self.store.num_dim();
            return self.store.block(page).chunks_exact(d).map(|row| dist(row, centroid)).fold(0.0, f32::max);
        }
        if depth >= MAX_DEPTH {
            return f32::INFINITY;
        }
        let kids = self.store.ids(page).to_vec();
        let cents = self.store.block(page).to_vec();
        let mut r = 0.0f32;
        for (c, &child) in cents.chunks_exact(self.store.num_dim()).zip(&kids) {
            let rc = self.subtree_radius(child, c, depth + 1);
            self.store.set_radius(child, rc);
            r = r.max(dist(c, centroid) + rc);
        }
        r
    }

    /// Multiply every stored coordinate by `ratio` (one factor per dimension); radii
    /// grow by the largest factor so they stay upper bounds.
    pub fn scale_rows(&mut self, ratio: &[f64]) {
        let dim = ratio.len().max(1);
        let grow = ratio.iter().copied().fold(0.0f64, f64::max) as f32;
        for page in 0..self.num_pages() as u32 {
            for row in self.store.block_mut(page).chunks_mut(dim) {
                for (v, &r) in row.iter_mut().zip(ratio) {
                    *v = (f64::from(*v) * r) as f32;
                }
            }
            let r = self.store.radius(page);
            self.store.set_radius(page, r * grow);
        }
    }

    /// Recompute every subtree count and parent from the root, for a tree whose pages
    /// were all just loaded with count `u32::MAX` (unvisited). False unless every page
    /// is reached exactly once and the leaves hold `indexed_rows` rows.
    pub(crate) fn recount(&mut self) -> bool {
        let root = self.header.root_page_id;
        if root as usize >= self.num_pages() {
            return false;
        }
        self.store.set_parent(root, None);
        let total = self.count_subtree(root, 0);
        let all_seen = (0..self.num_pages() as u32).all(|p| self.store.count(p) != u32::MAX as usize);
        total == Some(self.header.indexed_rows) && all_seen
    }

    fn count_subtree(&mut self, page: u32, depth: usize) -> Option<usize> {
        if depth > MAX_DEPTH || self.store.count(page) != u32::MAX as usize {
            return None;
        }
        let n = if self.store.is_leaf(page) {
            self.store.len(page)
        } else {
            let mut sum = 0;
            for child in self.store.ids(page).to_vec() {
                if child as usize >= self.num_pages() {
                    return None;
                }
                sum += self.count_subtree(child, depth + 1)?;
                self.store.set_parent(child, Some(page));
            }
            sum
        };
        self.store.set_count(page, n);
        Some(n)
    }

    /// Panic unless every page's count and parent match a recount from the root.
    #[cfg(test)]
    pub(crate) fn assert_counts_consistent(&self) {
        let mut copy = self.try_clone().expect("clone tree");
        (0..copy.num_pages() as u32).for_each(|p| copy.store.set_count(p, u32::MAX as usize));
        assert!(copy.recount(), "tree does not recount");
        for p in 0..self.num_pages() as u32 {
            assert_eq!(self.store.count(p), copy.store.count(p), "count of page {p}");
            assert_eq!(self.store.parent(p), copy.store.parent(p), "parent of page {p}");
        }
    }
}
