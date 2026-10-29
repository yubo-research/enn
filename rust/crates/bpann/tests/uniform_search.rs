//! One search path must preserve f64 ranking across the former 8192-row boundary.
use ennbo_bpann::BpannBackend;
use ndarray::{array, Array1, Array2};
use tempfile::TempDir;

#[test]
fn continuous_differences_survive_tied_categorical_distances_at_every_size() {
    for n in [100, 8192, 8193, 10000] {
        let dir = TempDir::new().unwrap();
        // Every f32 coordinate rounds to the same value. Only f64 distinguishes rows.
        let x = Array2::from_shape_fn((n, 2), |(i, j)| {
            if j == 0 { 1.0 + (n - i) as f64 * 1e-12 } else { 1.0 }
        });
        let mut backend = BpannBackend::new(
            dir.path().to_path_buf(), x, Array2::zeros((n, 1)), None,
            false, Array1::ones(2),
        ).unwrap();
        let query = array![[0.0, 0.0]];
        for scaled in [false, true] {
            if scaled {
                backend.rescale_metric(&array![0.3, 1.0]).unwrap();
            }
            let (distances, indices) = backend.search(&query.view(), 3, false).unwrap();
            assert_eq!(indices.row(0).to_vec(), vec![(n-1) as i64, (n-2) as i64, (n-3) as i64]);
            assert!(distances[[0, 0]] < distances[[0, 1]]);
        }
    }
}

#[test]
fn self_exclusion_and_duplicate_ties_use_original_precision() {
    let dir = TempDir::new().unwrap();
    let x = array![[1.0], [1.0], [1.0 + 1e-10], [2.0]];
    let mut backend = BpannBackend::new(
        dir.path().to_path_buf(), x, Array2::zeros((4, 1)), None,
        false, Array1::ones(1),
    ).unwrap();
    let (_, ids) = backend.search(&array![[1.0]].view(), 3, true).unwrap();
    assert_eq!(ids.row(0).to_vec(), vec![1, 2, 3]);
    backend.mark_index_stale();
    let (_, pending) = backend.search(&array![[1.0]].view(), 3, true).unwrap();
    assert_eq!(pending, ids);
}

#[test]
fn bounded_leaf_scoring_matches_complete_f64_distances_on_novel_queries() {
    let dir = TempDir::new().unwrap();
    let x = Array2::from_shape_fn((10000, 16), |(i, j)| ((i * 7 + j * 13) as f64 * 0.3).sin());
    let query = x.slice(ndarray::s![..3, ..]).mapv(|v| v + 0.03);
    let mut backend = BpannBackend::new(
        dir.path().to_path_buf(), x.clone(), Array2::zeros((10000, 1)), None,
        false, Array1::ones(16),
    ).unwrap();
    for _ in 0..30 {
        backend.rescale_metric(&Array1::from_elem(16, 0.3)).unwrap();
        backend.rescale_metric(&Array1::ones(16)).unwrap();
    }
    for scale in [Array1::ones(16), Array1::from_shape_fn(16, |j| [0.1, 1.0, 3.0][j % 3])] {
        backend.rescale_metric(&scale).unwrap();
        let (distances, indices) = backend.search(&query.view(), 10, false).unwrap();
        for q in 0..query.nrows() {
            let mut oracle: Vec<(f64, i64)> = (0..x.nrows()).map(|row| (
                ennbo_bpann::distance::row_sq_l2(query.row(q), x.row(row), true, scale.view()), row as i64,
            )).collect();
            oracle.sort_by(|a,b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            assert_eq!(indices.row(q).to_vec(), oracle[..10].iter().map(|p| p.1).collect::<Vec<_>>());
            assert_eq!(distances.row(q).to_vec(), oracle[..10].iter().map(|p| p.0).collect::<Vec<_>>());
        }
    }
}

#[test]
fn underflowing_f32_distances_keep_the_f64_boundary() {
    let dir = TempDir::new().unwrap();
    let x = Array2::from_shape_fn((100, 1), |(i, _)| 3e-23 + (100-i) as f64 * 1e-26);
    let backend = BpannBackend::new(
        dir.path().to_path_buf(), x, Array2::zeros((100, 1)), None,
        false, Array1::ones(1),
    ).unwrap();
    let (_, ids) = backend.search(&array![[0.0]].view(), 3, false).unwrap();
    assert_eq!(ids.row(0).to_vec(), vec![99, 98, 97]);
}
