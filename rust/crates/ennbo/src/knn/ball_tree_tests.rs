use super::{BallTreeBackend, SearchMode, AABB_MODE_MIN_N};
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
    let (d, i) = backend.search(&array![[0.1, 0.1]].view(), 2, 2).unwrap();
    assert_eq!(i[[0, 0]], 0);
    assert!(d[[0, 0]] < d[[0, 1]]);
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
fn ball_tree_aabb_mode_matches_bruteforce_at_large_n() {
    let n = AABB_MODE_MIN_N;
    let mut train = Array2::<f64>::zeros((n, 2));
    for i in 0..n {
        train[[i, 0]] = (i as f64) * 0.01;
        train[[i, 1]] = ((i * 7) % n) as f64 * 0.01;
    }
    let mut backend = BallTreeBackend::new(2, &train.view()).unwrap();
    assert_eq!(backend.search_mode, SearchMode::Aabb);
    let q = array![[1.0, 2.0]];
    let (d, idx) = backend.search(&q.view(), 3, 3).unwrap();
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
}
