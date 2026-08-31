use super::{BallTreeBackend, SearchMode, AABB_MODE_MIN_N, TAIL_REBUILD_MAX, TREE_SEARCH_MIN_N};
use crate::index::IndexError;
use ndarray::{array, Array2};

#[test]
fn ball_tree_search_matches_bruteforce_nn() {
    let train = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.5],
    ];
    let mut backend = BallTreeBackend::new(2, &train.view()).unwrap();
    assert_eq!(backend.len(), 5);
    assert_eq!(backend.search_mode, SearchMode::Brute);
    let (d, i) = backend.search(&array![[0.1, 0.1]].view(), 2, 2).unwrap();
    assert_eq!(i[[0, 0]], 0);
    assert!(d[[0, 0]] < d[[0, 1]]);
}

#[test]
fn ball_tree_faiss_add_creates_index_on_first_batch() {
    let empty = Array2::<f64>::zeros((0, 2));
    let mut backend = BallTreeBackend::new(2, &empty.view()).unwrap();
    assert!(backend.faiss_flat.is_none());
    backend
        .add(&array![[0.0, 0.0], [1.0, 0.0]].view(), 0)
        .unwrap();
    assert_eq!(backend.len(), 2);
    assert!(backend.faiss_flat.is_some());
    let (_d, i) = backend.search(&array![[0.05, 0.0]].view(), 1, 1).unwrap();
    assert_eq!(i[[0, 0]], 0);
}

#[test]
fn ball_tree_faiss_incremental_add_fast_path() {
    let train = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let mut backend = BallTreeBackend::new(2, &train.view()).unwrap();
    assert!(backend.faiss_flat.is_some());
    backend.add(&array![[0.5, 0.5]].view(), 3).unwrap();
    backend.add(&array![[0.25, 0.75]].view(), 4).unwrap();
    assert_eq!(backend.len(), 5);
    let (_d, i) = backend.search(&array![[0.24, 0.76]].view(), 1, 1).unwrap();
    assert_eq!(i[[0, 0]], 4);
}

#[test]
fn ball_tree_add_rebuilds_lazily() {
    let train = array![[0.0, 0.0]];
    let mut backend = BallTreeBackend::new(2, &train.view()).unwrap();
    backend.add(&array![[1.0, 0.0]].view(), 1).unwrap();
    assert_eq!(backend.len(), 2);
    let (_d, i) = backend.search(&array![[0.9, 0.0]].view(), 1, 1).unwrap();
    assert_eq!(i[[0, 0]], 1);
    assert!(backend.memory_usage_bytes() > 0);
    backend.rebuild(&train.view()).unwrap();
    assert_eq!(backend.len(), 1);
}

#[test]
fn ball_tree_empty_and_zero_k() {
    let empty = Array2::<f64>::zeros((0, 2));
    let mut backend = BallTreeBackend::new(2, &empty.view()).unwrap();
    assert_eq!(backend.len(), 0);
    let (d, i) = backend.search(&array![[0.0, 0.0]].view(), 0, 3).unwrap();
    assert_eq!(d.ncols(), 3);
    assert_eq!(i.ncols(), 3);
    assert!(d.iter().all(|v| v.is_infinite()));
    backend
        .add(&Array2::<f64>::zeros((0, 2)).view(), 0)
        .unwrap();
    assert_eq!(backend.len(), 0);
}

#[test]
fn ball_tree_shape_errors() {
    let train = array![[0.0, 0.0]];
    let mut backend = BallTreeBackend::new(2, &train.view()).unwrap();
    let bad = array![[0.0, 0.0, 0.0]];
    assert!(matches!(
        backend.rebuild(&bad.view()),
        Err(IndexError::InvalidShape { expected: 2, got: 3 })
    ));
    assert!(matches!(
        backend.add(&bad.view(), 1),
        Err(IndexError::InvalidShape { expected: 2, got: 3 })
    ));
    assert!(matches!(
        backend.search(&bad.view(), 1, 1),
        Err(IndexError::InvalidShape { expected: 2, got: 3 })
    ));
}

#[test]
fn ball_tree_identical_points_and_pad() {
    let train = Array2::from_elem((20, 2), 1.0);
    let mut backend = BallTreeBackend::new(2, &train.view()).unwrap();
    assert_eq!(backend.len(), 20);
    let (d, i) = backend.search(&array![[1.0, 1.0]].view(), 3, 5).unwrap();
    assert_eq!(d.ncols(), 5);
    assert_eq!(i.ncols(), 5);
    assert!(d[[0, 0]] < 1e-12);
    assert!(d[[0, 4]].is_infinite() || d[[0, 4]] >= d[[0, 2]]);
}

#[test]
fn ball_tree_k_equals_n_and_multi_query() {
    let train = array![[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0]];
    let mut backend = BallTreeBackend::new(2, &train.view()).unwrap();
    let queries = array![[0.1, 0.1], [1.9, 1.9]];
    let (d, i) = backend.search(&queries.view(), 4, 4).unwrap();
    assert_eq!(i[[0, 0]], 0);
    assert_eq!(i[[1, 0]], 3);
    assert!(d[[0, 0]] <= d[[0, 3]]);
    assert!(d[[1, 0]] <= d[[1, 3]]);
}

#[test]
fn ball_tree_mid_n_uses_brute_not_tree() {
    let n = TREE_SEARCH_MIN_N - 1;
    let train = Array2::<f64>::zeros((n, 2));
    let backend = BallTreeBackend::new(2, &train.view()).unwrap();
    assert_eq!(backend.search_mode, SearchMode::Brute);
    assert!(backend.nodes.is_empty());
}

#[test]
fn ball_tree_ball_mode_search_matches_bruteforce() {
    let n = TREE_SEARCH_MIN_N;
    assert!(n < AABB_MODE_MIN_N);
    let mut train = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        train[[i, 0]] = (i as f64) * 0.01;
        train[[i, 1]] = ((i * 7) % n) as f64 * 0.01;
    }
    let mut backend = BallTreeBackend::new(2, &train.view()).unwrap();
    let q = [1.0, 2.0];
    let hits = {
        backend.search(&array![[1.0, 2.0]].view(), 3, 3).unwrap();
        assert_eq!(backend.search_mode, SearchMode::Ball);
        assert!(!backend.nodes.is_empty());
        backend.search_one_ball(&q, 3)
    };
    let mut best = (f64::INFINITY, 0usize);
    for i in 0..n {
        let dx = train[[i, 0]] - 1.0;
        let dy = train[[i, 1]] - 2.0;
        let dist = dx * dx + dy * dy;
        if dist < best.0 {
            best = (dist, i);
        }
    }
    assert_eq!(hits[0].1, best.1);
    let brute = backend.search_one_brute(&q, 3);
    assert_eq!(brute[0].1, best.1);
}

#[test]
fn ball_tree_aabb_mode_search_matches_bruteforce() {
    let n = AABB_MODE_MIN_N;
    let mut train = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        train[[i, 0]] = (i as f64) * 0.01;
        train[[i, 1]] = ((i * 7) % n) as f64 * 0.01;
    }
    let mut backend = BallTreeBackend::new(2, &train.view()).unwrap();
    let q = [1.0, 2.0];
    let (d, idx) = backend.search(&array![[1.0, 2.0]].view(), 3, 3).unwrap();
    assert_eq!(backend.search_mode, SearchMode::Aabb);
    assert!(!backend.nodes.is_empty());
    let hits = backend.search_one_aabb(&q, 5);
    assert!(hits.len() >= 3);
    let mut best = (f64::INFINITY, 0usize);
    for i in 0..n {
        let dx = train[[i, 0]] - 1.0;
        let dy = train[[i, 1]] - 2.0;
        let dist = dx * dx + dy * dy;
        if dist < best.0 {
            best = (dist, i);
        }
    }
    assert_eq!(idx[[0, 0]], best.1 as i64);
    assert!((d[[0, 0]] - best.0).abs() < 1e-9);
    assert_eq!(hits[0].1, best.1);
}

#[test]
fn ball_tree_add_crosses_into_tree_mode() {
    let n0 = TREE_SEARCH_MIN_N - 1;
    let train = Array2::<f64>::zeros((n0, 2));
    let mut backend = BallTreeBackend::new(2, &train.view()).unwrap();
    assert_eq!(backend.search_mode, SearchMode::Brute);
    backend.add(&array![[0.0, 0.0]].view(), n0 as u64).unwrap();
    assert!(backend.len() >= TREE_SEARCH_MIN_N);
    // Tree build is deferred to first search (Exact-cost adds / MuyGPyS pattern).
    assert!(backend.tree_pending || backend.nodes.is_empty());
    let (_d, i) = backend.search(&array![[0.0, 0.0]].view(), 1, 1).unwrap();
    assert_eq!(backend.search_mode, SearchMode::Ball);
    assert!(!backend.nodes.is_empty());
    assert_eq!(backend.tree_n, backend.len());
    assert!(i[[0, 0]] >= 0);
}

#[test]
fn ball_tree_incremental_insert_stays_exact() {
    let n0 = AABB_MODE_MIN_N.max(TREE_SEARCH_MIN_N);
    let mut train = Array2::<f64>::zeros((n0, 2));
    for i in 0..n0 {
        train[[i, 0]] = i as f64;
        train[[i, 1]] = 0.0;
    }
    let mut backend = BallTreeBackend::new(2, &train.view()).unwrap();
    let _ = backend.search(&array![[0.0, 0.0]].view(), 1, 1).unwrap();
    assert_eq!(backend.tree_n, n0);
    let add_n = (TAIL_REBUILD_MAX).min(n0).max(1);
    let mut extra = Array2::<f64>::zeros((add_n, 2));
    for i in 0..add_n {
        extra[[i, 0]] = (n0 + i) as f64;
        extra[[i, 1]] = 0.0;
    }
    backend.add(&extra.view(), n0 as u64).unwrap();
    assert!(backend.tree_pending);
    let q = array![[(n0 + add_n - 1) as f64, 0.0]];
    let (_d, idx) = backend.search(&q.view(), 1, 1).unwrap();
    assert_eq!(backend.tree_n, backend.len());
    assert_eq!(idx[[0, 0]], (n0 + add_n - 1) as i64);
}

#[test]
fn ball_tree_brute_fallback_and_empty_k() {
    let train = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
    let mut backend = BallTreeBackend::new(2, &train.view()).unwrap();
    let (d0, _) = backend.search(&array![[0.0, 0.0]].view(), 0, 1).unwrap();
    assert!(d0.iter().all(|v| v.is_infinite()) || d0.ncols() >= 1);
    let (_d, i) = backend.search(&array![[0.1, 0.0]].view(), 2, 2).unwrap();
    assert_eq!(i[[0, 0]], 0);
}





