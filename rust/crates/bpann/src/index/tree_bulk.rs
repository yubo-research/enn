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
use crate::index::split::{median_split, weighted_mean};
use crate::index::tree::{Tree, TREE_FANOUT, TREE_LEAF_CAPACITY};
use crate::index::tree_counts::dist;
use crate::index::tree_store::Kind;
use crate::small_n_search::OrderedF32;

/// An empty tree indexes at least this many rows in one bulk build, not row by row.
/// Fewer rows insert row by row in well under 0.1 s, so only large rebuilds use it.
pub const BULK_BUILD_MIN_ROWS: usize = 65_536;

/// Coordinate bytes a bulk build may hold at once; later rows insert one at a time,
/// so a rebuild's memory does not grow with the number of rows.
pub const BULK_BUILD_MAX_BYTES: usize = 256 << 20;

/// Rows one bulk build takes: [`BULK_BUILD_MAX_BYTES`] of `f32` coordinates, and
/// never fewer than [`BULK_BUILD_MIN_ROWS`].
pub fn bulk_build_max_rows(num_dim: usize) -> usize {
    (BULK_BUILD_MAX_BYTES / (4 * num_dim.max(1))).max(BULK_BUILD_MIN_ROWS)
}

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

/// Build the internal page over `kids` (holding `count` rows) in `tree`.
fn push_internal(tree: &mut Tree, kids: &[Emitted], count: usize) -> Result<Emitted, BpannError> {
    let centroids: Vec<&[f32]> = kids.iter().map(|k| k.centroid.as_slice()).collect();
    let weights: Vec<f64> = kids.iter().map(|k| k.count as f64).collect();
    let all: Vec<usize> = (0..kids.len()).collect();
    let centroid = weighted_mean(&centroids, &weights, &all);
    let radius = kids.iter().map(|k| dist(&k.centroid, &centroid) + k.radius).fold(0.0, f32::max);
    let s = &mut tree.store;
    let page_id = s.alloc(Kind::Internal)?;
    let ids: Vec<u32> = kids.iter().map(|k| k.page_id).collect();
    s.set_entries(page_id, &ids, &centroids.concat());
    s.set_count(page_id, count);
    s.set_radius(page_id, radius);
    kids.iter().for_each(|k| s.set_parent(k.page_id, Some(page_id)));
    Ok(Emitted {
        page_id,
        centroid,
        radius,
        count,
    })
}

struct Builder<'a> {
    rows: &'a [u32],
    vs: &'a [f32],
    dim: usize,
    order: &'a [usize],
    tree: Tree,
}

impl Builder<'_> {
    /// Emit `node`'s pages in post-order.
    fn emit(&mut self, node: &Node) -> Result<Emitted, BpannError> {
        let Node::Leaf { start, len } = node else {
            let kids = children(node).into_iter().map(|k| self.emit(k)).collect::<Result<Vec<_>, _>>()?;
            return push_internal(&mut self.tree, &kids, node.count());
        };
        let block = &self.vs[start * self.dim..(start + len) * self.dim];
        let points: Vec<&[f32]> = block.chunks_exact(self.dim).collect();
        let all: Vec<usize> = (0..*len).collect();
        let centroid = weighted_mean(&points, &vec![1.0; *len], &all);
        let radius = points.iter().map(|p| dist(p, &centroid)).fold(0.0, f32::max);
        let row_ids: Vec<u32> = self.order[*start..start + len].iter().map(|&i| self.rows[i]).collect();
        let s = &mut self.tree.store;
        let page_id = s.alloc(Kind::Leaf)?;
        s.set_entries(page_id, &row_ids, block);
        s.set_count(page_id, *len);
        s.set_radius(page_id, radius);
        Ok(Emitted {
            page_id,
            centroid,
            radius,
            count: *len,
        })
    }
}

/// Tree over `rows` (row-major scaled coordinates `vs`, reordered in place), with
/// every block filled. Holds `vs` and a few words per row in memory, so callers
/// bound the number of rows ([`bulk_build_max_rows`]).
pub fn bulk_build(
    rows: &[u32],
    vs: &mut [f32],
    num_dim: usize,
    index_dir: std::path::PathBuf,
) -> Result<Tree, BpannError> {
    let dim = num_dim.max(1);
    let mut order: Vec<usize> = (0..rows.len()).collect();
    let root = partition(&mut order, vs, 0, dim);
    let mut builder = Builder {
        rows,
        vs: &*vs,
        dim,
        order: &order,
        tree: Tree::empty(num_dim, index_dir)?,
    };
    let root_id = builder.emit(&root)?.page_id;
    let mut tree = builder.tree;
    tree.header.root_page_id = root_id;
    tree.header.indexed_rows = rows.len();
    Ok(tree)
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
        let mut tree = bulk_build(&rows[..8000], &mut vs[..8000 * dim].to_vec(), dim, dir.path().join("i")).unwrap();
        assert_eq!(bulk_build_max_rows(1_000_000), BULK_BUILD_MIN_ROWS);
        assert_eq!(bulk_build_max_rows(4), BULK_BUILD_MAX_BYTES / 16);
        for &row in &rows[8000..] {
            let r = row as usize;
            tree.insert_row(row, &vs[r * dim..(r + 1) * dim]).unwrap();
        }
        assert_eq!(tree.leaf_row_ids(), rows);
        tree.assert_counts_consistent();
        let s = &tree.store;
        for page in 0..tree.num_pages() as u32 {
            if s.is_leaf(page) {
                assert!((32..=TREE_LEAF_CAPACITY).contains(&s.len(page)));
                continue;
            }
            assert!((2..=TREE_FANOUT).contains(&s.len(page)));
            for (c, &child) in s.block(page).chunks_exact(dim).zip(s.ids(page)) {
                for &row in s.ids(child).iter().filter(|_| s.is_leaf(child)) {
                    let x = &vs[row as usize * dim..(row as usize + 1) * dim];
                    assert!(dist(x, c) <= s.radius(child) * 1.0001 + 1e-6);
                }
            }
        }
    }
}

