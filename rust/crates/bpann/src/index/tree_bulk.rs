//! One-pass (bulk) construction of the incremental tree from many rows at once,
//! used when an empty tree must index many rows (a metric rebuild or a reopen
//! without a usable persisted tree).
//!
//! Rows are split recursively, in parallel, by [`median_split`] until at most
//! [`TREE_LEAF_CAPACITY`] remain, the split rule an overflowing leaf uses. Each
//! internal page takes up to [`TREE_FANOUT`] nodes of that binary tree as children
//! (always expanding the largest), so the result has the same page limits, counts,
//! blocks and radii as a tree grown row by row and keeps growing incrementally.

use std::cmp::{Ordering, Reverse};

use rayon::prelude::*;

use crate::distance::l2_sq_f32;
use crate::error::BpannError;
use crate::index::build::BpannIndex;
use crate::index::page::Page;
use crate::index::split::{median_split, weighted_mean};
use crate::index::tree::{TreeCounts, TREE_FANOUT, TREE_LEAF_CAPACITY};
use crate::index::tree_counts::dist;
use crate::small_n_search::OrderedF32;

/// An empty tree indexes at least this many rows in one bulk build, not row by row.
/// Fewer rows insert row by row in well under 0.1 s, so only large rebuilds use it.
pub const BULK_BUILD_MIN_ROWS: usize = 65_536;

enum Node {
    Leaf { start: usize, len: usize },
    Split(Box<Node>, Box<Node>, usize),
}

impl Node {
    fn count(&self) -> usize {
        match self {
            Node::Leaf { len, .. } => *len,
            Node::Split(_, _, n) => *n,
        }
    }
}

/// Nodes with at least this many rows are split with parallel passes.
const PARALLEL_SPLIT_MIN_ROWS: usize = 32_768;

/// Put rows `perm` (positions into `order` / `buf`) first-to-last, moving the
/// coordinates in `buf` along with the row ids in `order`.
fn apply_perm(order: &mut [usize], buf: &mut [f32], dim: usize, perm: &[usize]) {
    if perm.len() < PARALLEL_SPLIT_MIN_ROWS {
        let new_order: Vec<usize> = perm.iter().map(|&p| order[p]).collect();
        let mut new_buf = Vec::with_capacity(buf.len());
        perm.iter().for_each(|&p| new_buf.extend_from_slice(&buf[p * dim..(p + 1) * dim]));
        order.copy_from_slice(&new_order);
        buf.copy_from_slice(&new_buf);
        return;
    }
    let new_order: Vec<usize> = perm.par_iter().map(|&p| order[p]).collect();
    let mut new_buf = vec![0.0f32; buf.len()];
    new_buf
        .par_chunks_mut(dim)
        .zip(perm.par_iter())
        .for_each(|(dst, &p)| dst.copy_from_slice(&buf[p * dim..(p + 1) * dim]));
    order.par_iter_mut().zip(new_order.par_iter()).for_each(|(o, &n)| *o = n);
    buf.par_chunks_mut(dim).zip(new_buf.par_chunks(dim)).for_each(|(b, n)| b.copy_from_slice(n));
}

/// [`median_split`] of the rows in `buf` (row-major, one row per entry of `order`;
/// same result), reordering both so the left half comes first; returns its size.
fn split_rows(order: &mut [usize], buf: &mut [f32], dim: usize) -> usize {
    if order.len() < PARALLEL_SPLIT_MIN_ROWS {
        let points: Vec<&[f32]> = buf.chunks_exact(dim).collect();
        let (left, right) = median_split(&points);
        let perm: Vec<usize> = left.iter().chain(&right).copied().collect();
        apply_perm(order, buf, dim, &perm);
        return left.len();
    }
    let rows = &*buf;
    let farthest = |anchor: &[f32]| {
        let (_, pos) = rows
            .par_chunks_exact(dim)
            .enumerate()
            .map(|(pos, x)| (OrderedF32(l2_sq_f32(x, anchor)), Reverse(pos)))
            .max()
            .expect("non-empty");
        &rows[pos.0 * dim..(pos.0 + 1) * dim]
    };
    let a = farthest(&rows[..dim]);
    let b = farthest(a);
    let dir: Vec<f32> = b.iter().zip(a).map(|(&x, &y)| x - y).collect();
    let mut proj: Vec<(f32, usize)> = rows
        .par_chunks_exact(dim)
        .enumerate()
        .map(|(pos, x)| (x.iter().zip(&dir).map(|(&v, &d)| v * d).sum(), pos))
        .collect();
    let half = order.len() / 2;
    proj.select_nth_unstable_by(half, |x, y| x.0.partial_cmp(&y.0).unwrap_or(Ordering::Equal).then(x.1.cmp(&y.1)));
    let perm: Vec<usize> = proj.iter().map(|&(_, pos)| pos).collect();
    apply_perm(order, buf, dim, &perm);
    half
}

fn partition(order: &mut [usize], buf: &mut [f32], start: usize, dim: usize) -> Node {
    let n = order.len();
    if n <= TREE_LEAF_CAPACITY {
        return Node::Leaf { start, len: n };
    }
    let half = split_rows(order, buf, dim);
    let (lo, hi) = order.split_at_mut(half);
    let (buf_lo, buf_hi) = buf.split_at_mut(half * dim);
    let (a, b) = rayon::join(|| partition(lo, buf_lo, start, dim), || partition(hi, buf_hi, start + half, dim));
    Node::Split(Box::new(a), Box::new(b), n)
}

/// Up to [`TREE_FANOUT`] descendants of `node` covering all its rows.
fn children(node: &Node) -> Vec<&Node> {
    let mut kids = vec![node];
    while kids.len() < TREE_FANOUT {
        let largest = (0..kids.len())
            .filter(|&i| matches!(kids[i], Node::Split(..)))
            .max_by_key(|&i| (kids[i].count(), Reverse(i)));
        let Some(i) = largest else { break };
        let Node::Split(a, b, _) = kids[i] else { unreachable!("filtered to splits") };
        kids.splice(i..=i, [a.as_ref(), b.as_ref()]);
    }
    kids
}

/// A page just built: its id, centroid, radius and row count.
#[derive(Clone, Debug)]
struct Emitted {
    page_id: u32,
    centroid: Vec<f32>,
    radius: f32,
    count: usize,
}

/// Build the internal page over `kids` (holding `count` rows), append it to `pages`.
fn push_internal(pages: &mut Vec<Page>, counts: &mut TreeCounts, kids: &[Emitted], count: usize) -> Emitted {
    let centroids: Vec<Vec<f32>> = kids.iter().map(|k| k.centroid.clone()).collect();
    let weights: Vec<f64> = kids.iter().map(|k| k.count as f64).collect();
    let all: Vec<usize> = (0..kids.len()).collect();
    let centroid = weighted_mean(&centroids, &weights, &all);
    let radius = kids.iter().map(|k| dist(&k.centroid, &centroid) + k.radius).fold(0.0, f32::max);
    let page_id = counts.alloc();
    counts.set_centroid_block(page_id, &centroids);
    counts.set(page_id, count);
    counts.set_radius(page_id, radius);
    pages.push(Page::Internal {
        page_id,
        centroids,
        child_page_ids: kids.iter().map(|k| k.page_id).collect(),
    });
    Emitted {
        page_id,
        centroid,
        radius,
        count,
    }
}

struct Builder<'a> {
    rows: &'a [u32],
    vs: &'a [f32],
    dim: usize,
    order: &'a [usize],
    pages: Vec<Page>,
    counts: TreeCounts,
}

impl Builder<'_> {
    /// Emit `node`'s pages in post-order, so page ids are positions in `pages` and
    /// page 0 is a leaf.
    fn emit(&mut self, node: &Node) -> Emitted {
        let Node::Leaf { start, len } = node else {
            let kids: Vec<Emitted> = children(node).into_iter().map(|k| self.emit(k)).collect();
            return push_internal(&mut self.pages, &mut self.counts, &kids, node.count());
        };
        let full = TREE_LEAF_CAPACITY + 1;
        let members = &self.order[*start..start + len];
        let mut block = Vec::with_capacity(full * self.dim);
        block.extend_from_slice(&self.vs[start * self.dim..(start + len) * self.dim]);
        let points: Vec<&[f32]> = block.chunks_exact(self.dim).collect();
        let all: Vec<usize> = (0..*len).collect();
        let centroid = weighted_mean(&points, &vec![1.0; *len], &all);
        let radius = points.iter().map(|p| dist(p, &centroid)).fold(0.0, f32::max);
        let mut row_ids = Vec::with_capacity(full);
        row_ids.extend(members.iter().map(|&i| self.rows[i]));
        let page_id = self.counts.alloc();
        *self.counts.block_mut(page_id) = block;
        self.counts.set(page_id, *len);
        self.counts.set_radius(page_id, radius);
        self.pages.push(Page::Leaf {
            page_id,
            row_ids,
            row_range: None,
            vectors: Vec::new(),
            stored_centroid: Some(centroid.clone()),
        });
        Emitted {
            page_id,
            centroid,
            radius,
            count: *len,
        }
    }
}

/// Index over `pages` (page ids are positions; page 0 is a leaf) rooted at `root_id`.
pub(crate) fn into_index(
    pages: Vec<Page>,
    mut counts: TreeCounts,
    root_id: u32,
    indexed_rows: usize,
    num_dim: usize,
    index_dir: std::path::PathBuf,
) -> Result<(BpannIndex, TreeCounts), BpannError> {
    let mut pages = pages.into_iter();
    let Some(Page::Leaf {
        row_ids,
        stored_centroid: Some(centroid),
        ..
    }) = pages.next()
    else {
        unreachable!("the first page built is a leaf");
    };
    let mut index = BpannIndex::build_row_ids_leaf_with_persist(&row_ids, centroid, num_dim, index_dir, false)?;
    pages.for_each(|page| index.push_page(page));
    index.header.root_page_id = root_id;
    index.header.indexed_rows = indexed_rows;
    index.header.leaf_capacity = TREE_LEAF_CAPACITY;
    counts.mark_rows_cached();
    Ok((index, counts))
}

/// Tree over `rows` (row-major scaled coordinates `vs`, reordered in place), with
/// every block filled.
pub fn bulk_build(
    rows: &[u32],
    vs: &mut [f32],
    num_dim: usize,
    index_dir: std::path::PathBuf,
) -> Result<(BpannIndex, TreeCounts), BpannError> {
    let dim = num_dim.max(1);
    let mut order: Vec<usize> = (0..rows.len()).collect();
    let root = partition(&mut order, vs, 0, dim);
    let mut builder = Builder {
        rows,
        vs: &*vs,
        dim,
        order: &order,
        pages: Vec::new(),
        counts: TreeCounts::default(),
    };
    let root_id = builder.emit(&root).page_id;
    into_index(builder.pages, builder.counts, root_id, rows.len(), num_dim, index_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn parallel_split_matches_median_split() {
        let (n, dim) = (PARALLEL_SPLIT_MIN_ROWS + 1001, 4usize);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(5);
        let vs: Vec<f32> = (0..n * dim).map(|_| rng.gen::<f32>()).collect();
        let mut order: Vec<usize> = (0..n).collect();
        let points: Vec<&[f32]> = vs.chunks_exact(dim).collect();
        let (left, right) = median_split(&points);
        let expect: Vec<usize> = left.iter().chain(&right).copied().collect();
        let mut buf = vs.clone();
        assert_eq!(split_rows(&mut order, &mut buf, dim), left.len());
        assert_eq!(order, expect);
        assert_eq!(&buf[..dim], &vs[expect[0] * dim..(expect[0] + 1) * dim]);
    }

    #[test]
    fn bulk_tree_then_inserts_keep_page_limits_rows_counts_and_radii() {
        let (n, dim) = (10_000usize, 3usize);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(3);
        let vs: Vec<f32> = (0..n * dim).map(|_| rng.gen::<f32>()).collect();
        let rows: Vec<u32> = (0..n as u32).collect();
        let dir = tempfile::TempDir::new().unwrap();
        let (mut index, mut counts) = bulk_build(&rows[..8000], &mut vs[..8000 * dim].to_vec(), dim, dir.path().join("i")).unwrap();
        for &row in &rows[8000..] {
            let r = row as usize;
            crate::index::tree::insert_row(&mut index, &mut counts, row, &vs[r * dim..(r + 1) * dim]);
        }
        counts.write_page_centroids(&mut index);
        let mut ids = index.leaf_row_ids();
        ids.sort_unstable();
        assert_eq!(ids, rows);
        let recount = TreeCounts::from_index(&index).expect("valid tree");
        for page in &index.pages {
            assert_eq!(counts.count(page.page_id()), recount.count(page.page_id()));
            match page {
                Page::Leaf { row_ids, .. } => assert!((32..=TREE_LEAF_CAPACITY).contains(&row_ids.len())),
                Page::Internal {
                    centroids,
                    child_page_ids,
                    ..
                } => {
                    assert!((2..=TREE_FANOUT).contains(&child_page_ids.len()));
                    for (c, &child) in centroids.iter().zip(child_page_ids) {
                        if let Some(Page::Leaf { row_ids, .. }) = index.page_by_id(child) {
                            for &row in row_ids {
                                let x = &vs[row as usize * dim..(row as usize + 1) * dim];
                                assert!(dist(x, c) <= counts.radius(child) * 1.0001 + 1e-6);
                            }
                        }
                    }
                }
            }
        }
    }
}

