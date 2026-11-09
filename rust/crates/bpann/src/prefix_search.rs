use std::collections::BinaryHeap;
use std::sync::atomic::{AtomicBool, Ordering};

use ndarray::{Array2, ArrayView1, ArrayView2};
use rayon::prelude::*;

use crate::backend::BpannBackend;
use crate::distance::row_sq_l2;
use crate::error::BpannError;
use crate::small_n_search::{
    load_or_build_small_n_cache, scaled_row, OrderedFloat, SMALL_N_INCORE_SEARCH_LIMIT,
};

static PREFIX_REUSE: AtomicBool = AtomicBool::new(true);

const MAX_UNIT_ROWS: usize = 64;
const MIN_UNIT_ROWS: usize = 4;
const MAX_TAIL_WORK: usize = 30_000;

pub fn prefix_reuse() -> bool {
    PREFIX_REUSE.load(Ordering::Relaxed)
}

pub fn set_prefix_reuse(value: bool) {
    PREFIX_REUSE.store(value, Ordering::Relaxed);
}

pub fn shared_prefix_len(queries: &ArrayView2<f64>) -> usize {
    let (n, d) = queries.dim();
    if n < 2 || d < 2 {
        return 0;
    }
    let mut at_least = vec![0usize; d];
    for i in 1..n {
        let common = queries
            .row(i)
            .iter()
            .zip(queries.row(i - 1).iter())
            .take(d - 1)
            .take_while(|(a, b)| a == b)
            .count();
        for slot in at_least.iter_mut().take(common) {
            *slot += 1;
        }
    }
    let pairs = n - 1;
    at_least.iter().take_while(|&&c| 2 * c >= pairs).count()
}

pub fn use_prefix_search(backend: &BpannBackend, queries: &ArrayView2<f64>) -> Option<usize> {
    let total = backend.len();
    let d = backend.num_dim;
    if !prefix_reuse() || total == 0 || total > SMALL_N_INCORE_SEARCH_LIMIT {
        return None;
    }
    if !queries.iter().all(|v| v.is_finite()) {
        return None;
    }
    let p = shared_prefix_len(queries);
    (p > 0 && p < d && total * (d - p) <= MAX_TAIL_WORK).then_some(p)
}

impl Shared<'_> {
    fn slack(&self) -> f64 {
        8.0 * (self.d as f64 + 2.0) * f64::EPSILON
    }
}

struct Shared<'a> {
    flat: &'a [f64],
    raw: &'a [f64],
    scaled: &'a [f64],
    reps: Option<&'a [bool]>,
    backend: &'a BpannBackend,
    total: usize,
    d: usize,
    p: usize,
    k: usize,
}

pub fn search_shared_prefix(
    backend: &BpannBackend,
    queries: &ArrayView2<f64>,
    k: usize,
    p: usize,
) -> Result<(Array2<f64>, Array2<i64>), BpannError> {
    let (n_query, d) = queries.dim();
    let total = backend.len();
    let flat = load_or_build_small_n_cache(backend, total)?;
    let x_scale = backend.x_scale.as_slice().expect("x_scale contiguous");
    let raw: Vec<f64> = queries.iter().copied().collect();
    let mut scaled = Vec::with_capacity(raw.len());
    let mut buf = Vec::with_capacity(d);
    for row in raw.chunks(d) {
        scaled_row(row, backend.scale_x, x_scale, &mut buf);
        scaled.extend_from_slice(&buf);
    }
    let reps: Option<Vec<bool>> = backend
        .index
        .row_filter
        .as_ref()
        .map(|_| (0..total).map(|r| backend.index.indexable(r)).collect());
    let shared = Shared {
        flat: &flat,
        raw: &raw,
        scaled: &scaled,
        reps: reps.as_deref(),
        backend,
        total,
        d,
        p,
        k,
    };
    let unit_rows = n_query
        .div_ceil(2 * rayon::current_num_threads())
        .clamp(MIN_UNIT_ROWS, MAX_UNIT_ROWS);
    let units = work_units(&raw, d, (p, unit_rows));
    let per_unit: Vec<Vec<Vec<(u32, f64)>>> = units
        .par_iter()
        .map(|&(start, end)| search_unit(&shared, start, end))
        .collect::<Result<_, _>>()?;
    let mut dist2s = Array2::from_elem((n_query, k), f64::INFINITY);
    let mut indices = Array2::from_elem((n_query, k), -1i64);
    let rows = per_unit.into_iter().flatten();
    for (q, hits) in rows.enumerate() {
        for (j, (id, dist)) in hits.into_iter().enumerate() {
            dist2s[[q, j]] = dist;
            indices[[q, j]] = i64::from(id);
        }
    }
    Ok((dist2s, indices))
}

fn work_units(raw: &[f64], d: usize, (p, unit_rows): (usize, usize)) -> Vec<(usize, usize)> {
    let n = raw.len() / d;
    let mut units = Vec::new();
    let mut start = 0;
    for r in 1..=n {
        let split = r == n
            || r - start == unit_rows
            || raw[r * d..r * d + p] != raw[(r - 1) * d..(r - 1) * d + p];
        if split {
            units.push((start, r));
            start = r;
        }
    }
    units
}

fn search_unit(
    s: &Shared<'_>,
    start: usize,
    end: usize,
) -> Result<Vec<Vec<(u32, f64)>>, BpannError> {
    let (d, p) = (s.d, s.p);
    let head = &s.scaled[start * d..start * d + p];
    let pre: Vec<f64> = s
        .flat
        .chunks_exact(d)
        .map(|train| sq_dist(head, &train[..p]))
        .collect();
    let mut approx = vec![f64::INFINITY; s.total];
    (start..end)
        .map(|r| {
            let tail = &s.scaled[r * d + p..(r + 1) * d];
            let mut best = BinaryHeap::with_capacity(s.k + 1);
            for (j, ((a, &base), train)) in approx
                .iter_mut()
                .zip(&pre)
                .zip(s.flat.chunks_exact(d))
                .enumerate()
            {
                if s.reps.is_some_and(|reps| !reps[j]) {
                    *a = f64::INFINITY;
                    continue;
                }
                *a = base + sq_dist(tail, &train[p..]);
                if best.len() < s.k {
                    best.push(OrderedFloat(*a));
                } else if best.peek().is_some_and(|worst| *a < worst.0) {
                    best.pop();
                    best.push(OrderedFloat(*a));
                }
            }
            let limit = best
                .peek()
                .map_or(f64::NEG_INFINITY, |w| w.0 * (1.0 + s.slack()));
            exact_candidates(s, r, &approx, limit)
        })
        .collect()
}

#[inline]
fn sq_dist(a: &[f64], b: &[f64]) -> f64 {
    const LANES: usize = 4;
    let mut acc = [0.0f64; LANES];
    let (a_chunks, a_rem) = a.as_chunks::<LANES>();
    let (b_chunks, b_rem) = b.as_chunks::<LANES>();
    for (ca, cb) in a_chunks.iter().zip(b_chunks) {
        for ((lane, &x), &y) in acc.iter_mut().zip(ca).zip(cb) {
            let delta = x - y;
            *lane += delta * delta;
        }
    }
    let mut sum: f64 = acc.iter().sum();
    for (&x, &y) in a_rem.iter().zip(b_rem) {
        let delta = x - y;
        sum += delta * delta;
    }
    sum
}

fn exact_candidates(
    s: &Shared<'_>,
    query_row: usize,
    approx: &[f64],
    limit: f64,
) -> Result<Vec<(u32, f64)>, BpannError> {
    let query = &s.raw[query_row * s.d..(query_row + 1) * s.d];
    let x_scale = ArrayView1::from(s.backend.x_scale.as_slice().expect("x_scale contiguous"));
    let mut ranked = Vec::with_capacity(s.k + 8);
    for j in (0..s.total).filter(|&j| approx[j] <= limit) {
        let train = s.backend.train_x.mmap_row_slice(j)?;
        let dist = row_sq_l2(
            ArrayView1::from(query),
            ArrayView1::from(train),
            s.backend.scale_x,
            x_scale,
        );
        ranked.push((j as u32, dist));
    }
    ranked.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
    ranked.truncate(s.k);
    Ok(ranked)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use tempfile::TempDir;

    fn random_matrix(rng: &mut ChaCha8Rng, n: usize, d: usize, lattice: bool) -> Array2<f64> {
        Array2::from_shape_fn((n, d), |_| {
            if lattice {
                f64::from(rng.gen_range(0..3))
            } else {
                rng.gen_range(-1.0..1.0)
            }
        })
    }

    fn backend(dir: &TempDir, x: Array2<f64>, scale: Option<Array1<f64>>) -> BpannBackend {
        let n = x.nrows();
        let d = x.ncols();
        let y = Array2::zeros((n, 1));
        let x_scale = scale.clone().unwrap_or_else(|| Array1::ones(d));
        let mut b = BpannBackend::new(
            dir.path().to_path_buf(),
            x,
            y,
            None,
            scale.is_some(),
            x_scale,
        )
        .unwrap();
        b.ensure_index_sync().unwrap();
        b
    }

    fn reference(b: &BpannBackend, q: ArrayView1<f64>, k: usize) -> Vec<(i64, f64)> {
        let xs = ArrayView1::from(b.x_scale.as_slice().unwrap());
        let mut all: Vec<(i64, f64)> = (0..b.len())
            .map(|j| {
                let row = ArrayView1::from(b.train_x.mmap_row_slice(j).unwrap());
                (j as i64, row_sq_l2(q, row, b.scale_x, xs))
            })
            .collect();
        all.sort_by(|a, c| a.1.total_cmp(&c.1).then(a.0.cmp(&c.0)));
        all.truncate(k);
        all
    }

    fn prefix_queries(
        rng: &mut ChaCha8Rng,
        rows: usize,
        d: usize,
        p: usize,
        run: usize,
    ) -> Array2<f64> {
        let mut q = random_matrix(rng, rows, d, false);
        for r in 0..rows {
            let head = (r / run) * run;
            for c in 0..p {
                q[[r, c]] = q[[head, c]];
            }
        }
        q
    }

    fn assert_matches_reference(b: &BpannBackend, q: &Array2<f64>, k: usize, p: usize) {
        let (dist, idx) = search_shared_prefix(b, &q.view(), k, p).unwrap();
        for r in 0..q.nrows() {
            let want = reference(b, q.row(r), k);
            let got: Vec<(i64, f64)> = (0..k).map(|j| (idx[[r, j]], dist[[r, j]])).collect();
            assert_eq!(got.len(), want.len());
            for (g, w) in got.iter().zip(&want) {
                assert_eq!(g.0, w.0, "row {r}");
                assert_eq!(g.1.to_bits(), w.1.to_bits(), "row {r}");
            }
        }
    }

    #[test]
    fn shared_prefix_len_counts_leading_columns() {
        assert_eq!(shared_prefix_len(&array![[1.0, 2.0, 3.0]].view()), 0);
        let same2 = array![[1.0, 2.0, 3.0], [1.0, 2.0, 4.0], [1.0, 2.0, 5.0]];
        assert_eq!(shared_prefix_len(&same2.view()), 2);
        let all_same = array![[1.0, 2.0], [1.0, 2.0]];
        assert_eq!(
            shared_prefix_len(&all_same.view()),
            1,
            "a tail column is kept"
        );
        let mixed = array![
            [1.0, 2.0, 3.0],
            [1.0, 9.0, 3.0],
            [7.0, 9.0, 3.0],
            [7.0, 9.0, 4.0]
        ];
        assert_eq!(
            shared_prefix_len(&mixed.view()),
            1,
            "2 of 3 pairs share column 0 only"
        );
        let half = array![
            [1.0, 2.0, 3.0],
            [1.0, 2.0, 4.0],
            [5.0, 2.0, 4.0],
            [5.0, 2.0, 6.0]
        ];
        assert_eq!(
            shared_prefix_len(&half.view()),
            2,
            "pairs share 2, 0, 2 columns"
        );
    }

    #[test]
    fn matches_exhaustive_ranking_unscaled_and_scaled() {
        let mut rng = ChaCha8Rng::seed_from_u64(7);
        let (n, d) = (300, 6);
        let x = random_matrix(&mut rng, n, d, false);
        let q = prefix_queries(&mut rng, 40, d, 4, 8);
        let dir = TempDir::new().unwrap();
        assert_matches_reference(&backend(&dir, x.clone(), None), &q, 7, 4);
        let scale = Array1::from_vec(vec![0.5, 2.0, 1.0, 3.0, 0.7, 1.3]);
        let dir = TempDir::new().unwrap();
        assert_matches_reference(&backend(&dir, x, Some(scale)), &q, 7, 4);
    }

    #[test]
    fn ties_and_duplicates_use_index_tiebreak() {
        let mut rng = ChaCha8Rng::seed_from_u64(11);
        let (n, d) = (200, 5);
        let x = random_matrix(&mut rng, n, d, true);
        let q = random_matrix(&mut rng, 30, d, true);
        let dir = TempDir::new().unwrap();
        let b = backend(&dir, x, None);
        for p in 1..d {
            assert_matches_reference(&b, &q, 10, p);
        }
    }

    #[test]
    fn k_equal_to_n_returns_every_row() {
        let mut rng = ChaCha8Rng::seed_from_u64(3);
        let x = random_matrix(&mut rng, 12, 4, false);
        let q = prefix_queries(&mut rng, 6, 4, 2, 3);
        let dir = TempDir::new().unwrap();
        assert_matches_reference(&backend(&dir, x, None), &q, 12, 2);
    }

    #[test]
    fn near_tie_at_kth_neighbour_is_resolved_exactly() {
        let d = 8;
        let mut x = Array2::zeros((4, d));
        x[[0, 0]] = 1.0;
        x[[1, 0]] = -1.0;
        x[[2, 0]] = 1.0 + 1e-15;
        x[[3, 0]] = 5.0;
        let mut q = Array2::zeros((4, d));
        for r in 0..4 {
            q[[r, d - 1]] = 0.1 * r as f64;
        }
        let dir = TempDir::new().unwrap();
        let b = backend(&dir, x, None);
        for k in 1..=3 {
            assert_matches_reference(&b, &q, k, 7);
        }
    }

    #[test]
    fn backend_search_routes_prefix_batches_and_agrees_with_single_rows() {
        let mut rng = ChaCha8Rng::seed_from_u64(5);
        let x = random_matrix(&mut rng, 500, 6, false);
        let q = prefix_queries(&mut rng, 24, 6, 4, 6);
        let dir = TempDir::new().unwrap();
        let b = backend(&dir, x, None);
        assert_eq!(use_prefix_search(&b, &q.view()), Some(4));
        let (dist, idx) = b.search(&q.view(), 5, false).unwrap();
        for r in 0..q.nrows() {
            let one = q.slice(ndarray::s![r..r + 1, ..]);
            let (d1, i1) = b.search(&one, 5, false).unwrap();
            assert_eq!(i1.row(0), idx.row(r));
            assert_eq!(d1.row(0).mapv(f64::to_bits), dist.row(r).mapv(f64::to_bits));
        }
    }

    #[test]
    fn exclude_nearest_and_non_finite_queries_skip_the_prefix_path() {
        let mut rng = ChaCha8Rng::seed_from_u64(9);
        let x = random_matrix(&mut rng, 50, 4, false);
        let mut q = prefix_queries(&mut rng, 8, 4, 2, 4);
        let dir = TempDir::new().unwrap();
        let b = backend(&dir, x, None);
        assert!(use_prefix_search(&b, &q.view()).is_some());
        q[[3, 3]] = f64::NAN;
        assert!(use_prefix_search(&b, &q.view()).is_none());
    }

    #[test]
    fn toggle_round_trips() {
        assert!(prefix_reuse());
        set_prefix_reuse(false);
        assert!(!prefix_reuse());
        set_prefix_reuse(true);
        assert!(prefix_reuse());
    }
}
