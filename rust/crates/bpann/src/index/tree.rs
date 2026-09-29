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
//! Leaf pages store row ids only; the scaled `f32` coordinates of each leaf's rows
//! are kept in memory in one contiguous block per leaf (4 bytes per coordinate,
//! half the `f64` observation file). Splits never fault observation pages back in,
//! and scoring a leaf reads one block instead of rows scattered across the store.
//!
//! The running-mean centroids live in the [`TreeCounts`] blocks; the centroids held
//! in the pages themselves are refreshed from the blocks only when a page's children
//! change and by [`TreeCounts::write_page_centroids`] before the pages are persisted.

use crate::distance::l2_sq_f32;
use crate::error::BpannError;
use crate::index::build::BpannIndex;
use crate::index::page::Page;
use crate::index::split::{median_split, two_means_split, weighted_mean};
pub use crate::index::tree_counts::TreeCounts;
use crate::index::tree_counts::dist;

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

/// A one-leaf tree holding `row` with centroid `v`.
pub fn new_tree(
    row: u32,
    v: &[f32],
    num_dim: usize,
    index_dir: std::path::PathBuf,
) -> Result<(BpannIndex, TreeCounts), BpannError> {
    let mut index =
        BpannIndex::build_row_ids_leaf_with_persist(&[row], v.to_vec(), num_dim, index_dir, false)?;
    index.header.leaf_capacity = TREE_LEAF_CAPACITY;
    let mut counts = TreeCounts::default();
    let id = counts.alloc();
    counts.set(id, 1);
    *counts.block_mut(id) = v.to_vec();
    counts.set_radius(id, 0.0);
    counts.mark_rows_cached();
    Ok((index, counts))
}

/// Insert `row` (scaled coordinates `v`) into the tree.
pub fn insert_row(index: &mut BpannIndex, counts: &mut TreeCounts, row: u32, v: &[f32]) {
    let mut path: Vec<(u32, usize)> = Vec::new();
    let mut cur = index.header.root_page_id;
    loop {
        counts.bump(cur);
        let page = index.page_by_id_mut(cur).expect("tree page");
        match page {
            Page::Internal { child_page_ids, .. } => {
                let block = counts.block_mut(cur);
                let slot = nearest(block, v);
                let child = child_page_ids[slot];
                let d = v.len();
                let n_child = counts.count(child) + 1;
                let c = &mut counts.block_mut(cur)[slot * d..(slot + 1) * d];
                let moved = dist(v, c) / n_child as f32;
                mean_update(c, v, n_child);
                let r = (counts.radius(child) + moved).max(moved * (n_child - 1) as f32);
                counts.set_radius(child, r);
                path.push((cur, slot));
                cur = child;
            }
            Page::Leaf { row_ids, .. } => {
                row_ids.push(row);
                counts.block_mut(cur).extend_from_slice(v);
                let overflow = row_ids.len() > TREE_LEAF_CAPACITY;
                index.header.indexed_rows += 1;
                if overflow {
                    split_leaf(index, counts, cur, &mut path);
                }
                return;
            }
        }
    }
}

fn split_leaf(index: &mut BpannIndex, counts: &mut TreeCounts, leaf_id: u32, path: &mut Vec<(u32, usize)>) {
    let Some(Page::Leaf { row_ids, .. }) = index.page_by_id(leaf_id) else {
        unreachable!("split_leaf on a non-leaf page");
    };
    let row_ids = row_ids.clone();
    let block = std::mem::take(counts.block_mut(leaf_id));
    let dim = block.len() / row_ids.len();
    let vectors: Vec<&[f32]> = block.chunks(dim).collect();
    let weights = vec![1.0; vectors.len()];
    let (left, right) = median_split(&vectors);
    let ca = weighted_mean(&vectors, &weights, &left);
    let cb = weighted_mean(&vectors, &weights, &right);
    let full = TREE_LEAF_CAPACITY + 1;
    let pick_ids = |g: &[usize]| -> Vec<u32> {
        let mut ids = Vec::with_capacity(full);
        ids.extend(g.iter().map(|&i| row_ids[i]));
        ids
    };
    let (ids_a, ids_b) = (pick_ids(&left), pick_ids(&right));
    let gather = |g: &[usize]| -> Vec<f32> {
        let mut out = Vec::with_capacity(full * dim);
        g.iter().for_each(|&i| out.extend_from_slice(vectors[i]));
        out
    };
    let radius = |g: &[usize], c: &[f32]| g.iter().map(|&i| dist(vectors[i], c)).fold(0.0, f32::max);
    counts.set_radius(leaf_id, radius(&left, &ca));
    *counts.block_mut(leaf_id) = gather(&left);
    counts.set(leaf_id, ids_a.len());
    if let Some(Page::Leaf {
        row_ids,
        stored_centroid,
        ..
    }) = index.page_by_id_mut(leaf_id)
    {
        *row_ids = ids_a;
        *stored_centroid = Some(ca.clone());
    }
    let new_id = counts.alloc();
    counts.set(new_id, ids_b.len());
    counts.set_radius(new_id, radius(&right, &cb));
    *counts.block_mut(new_id) = gather(&right);
    index.push_page(Page::Leaf {
        page_id: new_id,
        row_ids: ids_b,
        row_range: None,
        vectors: Vec::new(),
        stored_centroid: Some(cb.clone()),
    });
    attach_sibling(index, counts, path, (leaf_id, ca), (new_id, cb));
}

/// Record that `old` was split into `old` and `new`, updating the parent (or growing a new root).
fn attach_sibling(
    index: &mut BpannIndex,
    counts: &mut TreeCounts,
    path: &mut Vec<(u32, usize)>,
    old: (u32, Vec<f32>),
    new: (u32, Vec<f32>),
) {
    let Some((parent, slot)) = path.pop() else {
        let root = counts.alloc();
        counts.set(root, counts.count(old.0) + counts.count(new.0));
        *counts.block_mut(root) = [old.1.as_slice(), new.1.as_slice()].concat();
        index.push_page(Page::Internal {
            page_id: root,
            centroids: vec![old.1, new.1],
            child_page_ids: vec![old.0, new.0],
        });
        index.header.root_page_id = root;
        return;
    };
    let Some(Page::Internal {
        centroids,
        child_page_ids,
        ..
    }) = index.page_by_id_mut(parent)
    else {
        unreachable!("tree parent must be internal");
    };
    for (c, b) in centroids.iter_mut().zip(counts.block(parent).chunks_exact(old.1.len())) {
        c.copy_from_slice(b);
    }
    centroids[slot] = old.1;
    centroids.push(new.1);
    child_page_ids.push(new.0);
    counts.set_centroid_block(parent, centroids);
    if child_page_ids.len() > TREE_FANOUT {
        split_internal(index, counts, path, parent);
    }
}

fn split_internal(index: &mut BpannIndex, counts: &mut TreeCounts, path: &mut Vec<(u32, usize)>, page_id: u32) {
    let Some(Page::Internal {
        centroids,
        child_page_ids,
        ..
    }) = index.page_by_id_mut(page_id)
    else {
        unreachable!("split_internal on a non-internal page");
    };
    let all_centroids = std::mem::take(centroids);
    let all_children = std::mem::take(child_page_ids);
    let weights: Vec<f64> = all_children.iter().map(|&c| counts.count(c) as f64).collect();
    let (left, right) = two_means_split(&all_centroids, &weights);
    let ca = weighted_mean(&all_centroids, &weights, &left);
    let cb = weighted_mean(&all_centroids, &weights, &right);
    let pick = |g: &[usize]| -> (Vec<Vec<f32>>, Vec<u32>) {
        (g.iter().map(|&i| all_centroids[i].clone()).collect(), g.iter().map(|&i| all_children[i]).collect())
    };
    let radius = |g: &[usize], c: &[f32]| {
        g.iter().map(|&i| dist(&all_centroids[i], c) + counts.radius(all_children[i])).fold(0.0, f32::max)
    };
    let (r_a, r_b) = (radius(&left, &ca), radius(&right, &cb));
    let (cents_a, kids_a) = pick(&left);
    let (cents_b, kids_b) = pick(&right);
    counts.set_radius(page_id, r_a);
    counts.set(page_id, kids_a.iter().map(|&c| counts.count(c)).sum());
    let new_id = counts.alloc();
    counts.set(new_id, kids_b.iter().map(|&c| counts.count(c)).sum());
    counts.set_radius(new_id, r_b);
    if let Some(Page::Internal {
        centroids,
        child_page_ids,
        ..
    }) = index.page_by_id_mut(page_id)
    {
        counts.set_centroid_block(page_id, &cents_a);
        *centroids = cents_a;
        *child_page_ids = kids_a;
    }
    counts.set_centroid_block(new_id, &cents_b);
    index.push_page(Page::Internal {
        page_id: new_id,
        centroids: cents_b,
        child_page_ids: kids_b,
    });
    attach_sibling(index, counts, path, (page_id, ca), (new_id, cb));
}
