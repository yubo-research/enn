//! Incremental B+ANN-style tree: one BPANN index that grows a row at a time.
//!
//! A new row descends from the root to the leaf whose centroid is nearest,
//! updating the running-mean centroid of every page on the way. A leaf holding
//! more than [`TREE_LEAF_CAPACITY`] rows splits into equal halves at the median of a
//! projection ([`median_split`]), and an internal page with more than [`TREE_FANOUT`]
//! children splits by weighted 2-means, pushing the new sibling into its parent as in
//! a B+ tree. Depth stays `O(log N)`, so an insert costs `O(log N)` and the index is
//! ready for a query after every insert.
//!
//! Every page lives in a disk-backed [`TreeStore`]: its row ids (a leaf) or child page
//! ids (an internal page), the scaled `f32` coordinates of a leaf's rows (4 bytes per
//! coordinate, half the `f64` observation file) or an internal page's running-mean
//! child centroids, and its subtree row count, radius and parent. Splits never fault
//! observation pages back in, scoring a leaf reads one slot, and resident memory is
//! bounded by the store's budget rather than growing with the number of rows.

use std::path::PathBuf;

use crate::distance::l2_sq_f32;
use crate::error::BpannError;
use crate::index::build::{IndexHeader, DEFAULT_SKIP_NEIGHBORS};
use crate::index::split::{median_split, two_means_split, weighted_mean};
use crate::index::tree_counts::dist;
use crate::index::tree_store::{Kind, TreeStore};

pub const TREE_LEAF_CAPACITY: usize = 64;
pub const TREE_FANOUT: usize = 16;

fn mean_update(centroid: &mut [f32], v: &[f32], n: usize) {
    let inv = 1.0 / n as f32;
    for (c, &x) in centroid.iter_mut().zip(v) {
        *c += (x - *c) * inv;
    }
}

fn nearest(block: &[f32], v: &[f32]) -> usize {
    let mut best = (0usize, f32::INFINITY);
    for (i, c) in block.chunks_exact(v.len()).enumerate() {
        let d = l2_sq_f32(v, c);
        if d < best.1 {
            best = (i, d);
        }
    }
    best.0
}

/// Largest distance from `c` to the points `g` (positions into `points`).
fn max_dist(points: &[&[f32]], g: &[usize], c: &[f32]) -> f32 {
    g.iter().map(|&i| dist(points[i], c)).fold(0.0, f32::max)
}

/// The incremental tree: its header and the store holding every page.
pub struct Tree {
    pub header: IndexHeader,
    pub store: TreeStore,
    pub index_dir: PathBuf,
    /// Whether every leaf block holds its rows' coordinates (false for a tree read
    /// back from `pages.bin` until [`Tree::fill_blocks`] runs).
    pub(crate) rows_cached: bool,
}

impl Tree {
    /// A tree with no pages yet; its scratch files live in `index_dir`.
    pub fn empty(num_dim: usize, index_dir: PathBuf) -> Result<Self, BpannError> {
        Ok(Self {
            header: IndexHeader {
                num_dim,
                indexed_rows: 0,
                root_page_id: 0,
                leaf_capacity: TREE_LEAF_CAPACITY,
                skip_neighbors: DEFAULT_SKIP_NEIGHBORS,
            },
            store: TreeStore::create(&index_dir, num_dim)?,
            index_dir,
            rows_cached: true,
        })
    }

    /// A one-leaf tree holding `row` with coordinates `v`.
    pub fn new_tree(row: u32, v: &[f32], num_dim: usize, index_dir: PathBuf) -> Result<Self, BpannError> {
        let mut tree = Self::empty(num_dim, index_dir)?;
        let id = tree.store.alloc(Kind::Leaf)?;
        tree.store.push_entry(id, row, v);
        tree.store.set_count(id, 1);
        tree.store.set_radius(id, 0.0);
        tree.header.root_page_id = id;
        tree.header.indexed_rows = 1;
        Ok(tree)
    }

    pub fn rows_cached(&self) -> bool {
        self.rows_cached
    }

    pub fn num_pages(&self) -> usize {
        self.store.num_pages()
    }

    /// A copy of the tree in new scratch files.
    pub fn try_clone(&self) -> Result<Self, BpannError> {
        Ok(Self {
            header: self.header.clone(),
            store: self.store.try_clone()?,
            index_dir: self.index_dir.clone(),
            rows_cached: self.rows_cached,
        })
    }

    /// Every indexed row id, ascending (reads every leaf).
    pub fn leaf_row_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = (0..self.num_pages() as u32)
            .filter(|&p| self.store.is_leaf(p))
            .flat_map(|p| self.store.ids(p).to_vec())
            .collect();
        ids.sort_unstable();
        ids
    }

    /// Insert `row` (scaled coordinates `v`) into the tree.
    pub fn insert_row(&mut self, row: u32, v: &[f32]) -> Result<(), BpannError> {
        let mut path: Vec<(u32, usize)> = Vec::new();
        let mut cur = self.header.root_page_id;
        loop {
            let s = &mut self.store;
            s.set_count(cur, s.count(cur) + 1);
            if s.is_leaf(cur) {
                s.push_entry(cur, row, v);
                self.header.indexed_rows += 1;
                if self.store.len(cur) > TREE_LEAF_CAPACITY {
                    self.split_leaf(cur, &mut path)?;
                }
                return Ok(());
            }
            let slot = nearest(s.block(cur), v);
            let child = s.ids(cur)[slot];
            let n_child = s.count(child) + 1;
            let c = s.entry_mut(cur, slot);
            let moved = dist(v, c) / n_child as f32;
            mean_update(c, v, n_child);
            let r = (s.radius(child) + moved).max(moved * (n_child - 1) as f32);
            s.set_radius(child, r);
            path.push((cur, slot));
            cur = child;
        }
    }

    fn split_leaf(&mut self, leaf: u32, path: &mut Vec<(u32, usize)>) -> Result<(), BpannError> {
        let row_ids = self.store.ids(leaf).to_vec();
        let block = self.store.block(leaf).to_vec();
        let vectors: Vec<&[f32]> = block.chunks(self.store.num_dim()).collect();
        let weights = vec![1.0; vectors.len()];
        let (left, right) = median_split(&vectors);
        let ca = weighted_mean(&vectors, &weights, &left);
        let cb = weighted_mean(&vectors, &weights, &right);
        let pick_ids = |g: &[usize]| -> Vec<u32> { g.iter().map(|&i| row_ids[i]).collect() };
        let gather = |g: &[usize]| -> Vec<f32> { g.iter().flat_map(|&i| vectors[i].iter().copied()).collect() };
        let s = &mut self.store;
        s.set_radius(leaf, max_dist(&vectors, &left, &ca));
        s.set_entries(leaf, &pick_ids(&left), &gather(&left));
        s.set_count(leaf, left.len());
        let new_id = s.alloc(Kind::Leaf)?;
        s.set_count(new_id, right.len());
        s.set_radius(new_id, max_dist(&vectors, &right, &cb));
        s.set_entries(new_id, &pick_ids(&right), &gather(&right));
        self.attach_sibling(path, (leaf, ca), (new_id, cb))
    }

    /// Record that `old` was split into `old` and `new`, updating the parent (or growing a new root).
    fn attach_sibling(
        &mut self,
        path: &mut Vec<(u32, usize)>,
        old: (u32, Vec<f32>),
        new: (u32, Vec<f32>),
    ) -> Result<(), BpannError> {
        let s = &mut self.store;
        let Some((parent, slot)) = path.pop() else {
            let root = s.alloc(Kind::Internal)?;
            s.set_count(root, s.count(old.0) + s.count(new.0));
            s.set_entries(root, &[old.0, new.0], &[old.1.as_slice(), new.1.as_slice()].concat());
            s.set_parent(root, None);
            s.set_parent(old.0, Some(root));
            s.set_parent(new.0, Some(root));
            self.header.root_page_id = root;
            return Ok(());
        };
        s.set_parent(new.0, Some(parent));
        s.entry_mut(parent, slot).copy_from_slice(&old.1);
        s.push_entry(parent, new.0, &new.1);
        if s.len(parent) > TREE_FANOUT {
            self.split_internal(path, parent)?;
        }
        Ok(())
    }

    fn split_internal(&mut self, path: &mut Vec<(u32, usize)>, page_id: u32) -> Result<(), BpannError> {
        let s = &self.store;
        let all_children = s.ids(page_id).to_vec();
        let flat = s.block(page_id).to_vec();
        let all_centroids: Vec<&[f32]> = flat.chunks_exact(s.num_dim()).collect();
        let weights: Vec<f64> = all_children.iter().map(|&c| s.count(c) as f64).collect();
        let (left, right) = two_means_split(&all_centroids, &weights);
        let ca = weighted_mean(&all_centroids, &weights, &left);
        let cb = weighted_mean(&all_centroids, &weights, &right);
        let radius = |g: &[usize], c: &[f32]| {
            g.iter().map(|&i| dist(all_centroids[i], c) + s.radius(all_children[i])).fold(0.0, f32::max)
        };
        let (r_a, r_b) = (radius(&left, &ca), radius(&right, &cb));
        let pick = |g: &[usize]| -> (Vec<f32>, Vec<u32>) {
            (g.iter().flat_map(|&i| all_centroids[i].iter().copied()).collect(), g.iter().map(|&i| all_children[i]).collect())
        };
        let ((cents_a, kids_a), (cents_b, kids_b)) = (pick(&left), pick(&right));
        let n_a: usize = kids_a.iter().map(|&c| s.count(c)).sum();
        let n_b: usize = kids_b.iter().map(|&c| s.count(c)).sum();
        let s = &mut self.store;
        s.set_radius(page_id, r_a);
        s.set_count(page_id, n_a);
        let new_id = s.alloc(Kind::Internal)?;
        s.set_count(new_id, n_b);
        s.set_radius(new_id, r_b);
        kids_b.iter().for_each(|&kid| s.set_parent(kid, Some(new_id)));
        s.set_entries(page_id, &kids_a, &cents_a);
        s.set_entries(new_id, &kids_b, &cents_b);
        self.attach_sibling(path, (page_id, ca), (new_id, cb))
    }
}
