//! `train_rows_at` matches full in-memory views on random index sets.

use ennbo::{EpistemicNearestNeighbors, IndexDriver};
use ndarray::array;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[test]
fn train_rows_at_matches_train_x_y_views() {
    let seed = 42u64;
    let mut rng = StdRng::seed_from_u64(seed);
    println!("train_rows_at fuzz seed={seed}");

    let train_x = array![
        [0.0, 1.0],
        [1.0, 0.0],
        [0.5, 0.5],
        [2.0, 3.0],
        [1.0, 1.0]
    ];
    let train_y = array![[0.0], [1.0], [1.5], [2.0], [0.5]];
    let model =
        EpistemicNearestNeighbors::new(train_x, train_y, None, false, IndexDriver::Exact).unwrap();

    for _ in 0..20 {
        let n = model.len();
        let k = rng.gen_range(1..=n);
        let mut indices: Vec<usize> = (0..n).collect();
        for i in 0..k {
            let j = rng.gen_range(i..n);
            indices.swap(i, j);
        }
        indices.truncate(k);

        let (x_at, y_at, _) = model.rows().train_rows_at(&indices).unwrap();
        let all: Vec<usize> = (0..n).collect();
        let (x_all, y_all, _) = model.rows().train_rows_at(&all).unwrap();
        for (r, &idx) in indices.iter().enumerate() {
            assert_eq!(x_at.row(r).to_vec(), x_all.row(idx).to_vec());
            assert_eq!(y_at.row(r).to_vec(), y_all.row(idx).to_vec());
        }
    }
}

#[test]
fn single_index_train_rows_at_matches_full_gather() {
    let seed = 99u64;
    let mut rng = StdRng::seed_from_u64(seed);
    println!("single-index train_rows_at fuzz seed={seed}");

    let train_x = array![
        [0.0, 1.0],
        [1.0, 0.0],
        [0.5, 0.5],
        [2.0, 3.0],
        [1.0, 1.0]
    ];
    let train_y = array![[0.0], [1.0], [1.5], [2.0], [0.5]];
    let model =
        EpistemicNearestNeighbors::new(train_x, train_y, None, false, IndexDriver::Exact).unwrap();

    for _ in 0..20 {
        let i = rng.gen_range(0..model.len());
        let (x_one, y_one, _) = model.rows().train_rows_at(&[i]).unwrap();
        let all: Vec<usize> = (0..model.len()).collect();
        let (x_all, y_all, _) = model.rows().train_rows_at(&all).unwrap();
        assert_eq!(x_one.row(0).to_vec(), x_all.row(i).to_vec());
        assert_eq!(y_one.row(0).to_vec(), y_all.row(i).to_vec());
    }
}

#[test]
fn scale_x_true_append_to_nonempty_succeeds() {
    let train_x = array![[0.0, 0.0], [1.0, 0.0]];
    let train_y = array![[0.0], [1.0]];
    let mut model =
        EpistemicNearestNeighbors::new(train_x, train_y, None, true, IndexDriver::Exact).unwrap();
    model
        .add(
            &array![[0.5, 0.5]].view(),
            &array![[0.5]].view(),
            None,
        )
        .expect("scale_x=true append to nonempty model must succeed");
    assert_eq!(model.len(), 3);
}
