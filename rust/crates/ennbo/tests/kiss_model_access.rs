//! Kiss static coverage for model row/index accessor types.

use ennbo::{EnnIndexAccess, EnnRowAccess, EpistemicNearestNeighbors, IndexDriver};
use ndarray::array;

const MODEL_ACCESS_SRC: &str = include_str!("../src/model/access.rs");

#[test]
fn kiss_model_access_types_imported() {
    let model =
        EpistemicNearestNeighbors::new(array![[0.0]], array![[1.0]], None, false, IndexDriver::Exact)
            .unwrap();
    let _: EnnIndexAccess<'_> = model.index_access();
    let _: EnnRowAccess<'_> = model.rows();
    let _ = model.index_access().ensure_sync();
    let _ = model.index_access().memory_bytes();
    let _ = model.index_access().is_stale();
    let _ = model.index_access().len();
    let _ = model.rows().train_rows_at(&[0]).unwrap();
    let _ = model.rows().row_x(0).unwrap();
    let _ = model.rows().row_y(0).unwrap();
    let _ = model.rows().row_yvar(0).unwrap();
}

#[test]
fn kiss_model_access_helper_names_in_source() {
    for name in [
        "EnnIndexAccess",
        "EnnRowAccess",
        "ensure_sync",
        "memory_bytes",
        "is_stale",
        "neighbor_distances_and_indices",
        "index_neighbor_distances_and_indices",
        "index_access",
        "rows",
        "train_rows_at",
        "row_x",
        "row_y",
        "row_yvar",
    ] {
        assert!(
            MODEL_ACCESS_SRC.contains(name),
            "missing {name} in model/access.rs"
        );
    }
}
