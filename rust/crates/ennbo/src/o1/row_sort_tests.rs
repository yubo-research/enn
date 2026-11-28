use super::forest_tests::{assert_invariants, insert_all, trajectory, D};
use super::*;
use tempfile::TempDir;

fn ordered_store(dir: &TempDir, sort_bytes: usize, x: &ndarray::Array2<f64>) -> O1Store {
    let mut store = O1Store::open_forest(dir.path().to_path_buf(), D, 1).unwrap();
    store
        .set_row_order(OrderParams {
            rows_per_insert: 3,
            grow: 1.3,
            min_rows: 500,
            sort_bytes,
        })
        .unwrap();
    insert_all(&mut store, x, 200);
    store
}

fn order_files(dir: &TempDir) -> Vec<(String, Vec<u8>)> {
    let mut files: Vec<(String, Vec<u8>)> = std::fs::read_dir(dir.path())
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| {
            p.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("o1f_xo_")
        })
        .map(|p| {
            (
                p.file_name().unwrap().to_string_lossy().into_owned(),
                std::fs::read(&p).unwrap(),
            )
        })
        .map(|(name, mut bytes)| {
            bytes.truncate(5000 * D * 8);
            (name, bytes)
        })
        .collect();
    files.sort();
    files
}

#[test]
fn sorted_row_order_matches_gathered_copy() {
    let x = trajectory(5000, 21);
    let (gather_dir, sort_dir) = (TempDir::new().unwrap(), TempDir::new().unwrap());
    let mut gathered = ordered_store(&gather_dir, 0, &x);
    let mut sorted = ordered_store(&sort_dir, 1000, &x);
    let (g, s) = (
        gathered.maint_stats().unwrap(),
        sorted.maint_stats().unwrap(),
    );
    assert!(s.order_builds >= 2 && s.ordered_rows > 0, "{s:?}");
    assert_eq!(
        (g.order_builds, g.ordered_rows),
        (s.order_builds, s.ordered_rows)
    );
    assert_eq!(gathered.order_rows().unwrap(), sorted.order_rows().unwrap());
    let files = order_files(&sort_dir);
    assert!(files.iter().all(|(name, _)| !name.contains("sort")));
    assert_eq!(order_files(&gather_dir), files);
    let src = row_order::Rows {
        x: &sorted.x,
        order: sorted.index.forest().order.as_ref(),
        near: None,
    };
    (0..5000)
        .step_by(7)
        .for_each(|i| assert_eq!(src.mapped(i).unwrap(), x.row(i).as_slice().unwrap()));
    let q = x.slice(ndarray::s![..20, ..]).to_owned();
    assert_eq!(
        sorted.search(&q.view(), 10, false).unwrap(),
        gathered.search(&q.view(), 10, false).unwrap()
    );
    assert_invariants(&sorted);
}
