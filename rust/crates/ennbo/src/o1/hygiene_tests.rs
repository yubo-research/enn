use super::forest_tests::{assert_invariants, insert_all, recall, trajectory, D};
use super::*;
use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use tempfile::TempDir;

fn queries(seed: u64) -> Array2<f64> {
    let mut rng = StdRng::seed_from_u64(seed);
    Array2::from_shape_fn((20, D), |_| 0.5 + 0.02 * (rng.gen::<f64>() - 0.5))
}

#[test]
fn hygiene_keeps_every_row_through_deferred_rebuilds_and_answers_exactly() {
    std::env::set_var("ENN_HYGIENE", "lrows");
    std::env::set_var("ENN_CLUSTER_LEAVES", "lazy");
    let dir = TempDir::new().unwrap();
    let mut store = O1Store::open_forest(dir.path().to_path_buf(), D, 1).unwrap();
    store.set_defer(8, 1).unwrap();
    let before = hygiene::counts();
    insert_all(&mut store, &trajectory(6000, 3), 10);
    assert_invariants(&store);
    let m = store.index.forest().maint_stats();
    assert!(m.deferred > 0 && m.rebuilt_rows > 0, "{m:?}");
    let after = hygiene::counts();
    assert!(after[0] > before[0], "no leaf-row drops ran: {after:?}");
    store.set_probe_budget(1 << 20, 1 << 20).unwrap();
    assert_eq!(recall(&store, &queries(4), 10), 1.0);
}
