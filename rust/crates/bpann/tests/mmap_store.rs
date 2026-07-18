use bpann::mmap_store;
use ndarray::{array, Array2, ShapeBuilder};
use tempfile::TempDir;

#[test]
fn mmap_append_fortran_order_preserves_rows() {
    let dir = TempDir::new().unwrap();
    let mut store =
        mmap_store::MmapColumnStore::mmap_open_or_create(dir.path().join("c.bin"), 3, None).unwrap();
    // Column-major (Fortran) layout: each logical row is strided in memory.
    let mut f = Array2::<f64>::zeros((2, 3).f());
    f[[0, 0]] = 1.0;
    f[[0, 1]] = 2.0;
    f[[0, 2]] = 3.0;
    f[[1, 0]] = 4.0;
    f[[1, 1]] = 5.0;
    f[[1, 2]] = 6.0;
    assert!(
        !f.is_standard_layout(),
        "fixture must be non-C layout to exercise the strided path"
    );
    store.mmap_append(&f.view()).unwrap();
    assert_eq!(
        store.mmap_row_slice(0).unwrap(),
        &[1.0, 2.0, 3.0],
        "row0 corrupted under Fortran layout"
    );
    assert_eq!(
        store.mmap_row_slice(1).unwrap(),
        &[4.0, 5.0, 6.0],
        "row1 corrupted under Fortran layout"
    );
}

#[test]
fn mmap_append_strided_column_view_preserves_rows() {
    let dir = TempDir::new().unwrap();
    let mut store =
        mmap_store::MmapColumnStore::mmap_open_or_create(dir.path().join("c.bin"), 2, None).unwrap();
    // Take every other column of a wider C-order matrix: rows are no longer contiguous.
    let wide = array![[1.0, 9.0, 2.0, 8.0], [3.0, 7.0, 4.0, 6.0]];
    let view = wide.slice(ndarray::s![.., ..;2]);
    assert_eq!(view.nrows(), 2);
    assert_eq!(view.ncols(), 2);
    assert!(
        view.as_slice().is_none(),
        "fixture must not be C-contiguous"
    );
    store.mmap_append(&view).unwrap();
    assert_eq!(store.mmap_row_slice(0).unwrap(), &[1.0, 2.0]);
    assert_eq!(store.mmap_row_slice(1).unwrap(), &[3.0, 4.0]);
}

#[test]
#[allow(non_snake_case)]
fn MmapColumnStore() {
    let dir = TempDir::new().unwrap();
    let mut store =
        mmap_store::MmapColumnStore::mmap_open_or_create(dir.path().join("c.bin"), 2, None).unwrap();
    store.mmap_append(&array![[1.0, 2.0]].view()).unwrap();
    assert_eq!(store.mmap_row_slice(0).unwrap()[0], 1.0);
}

#[test]
fn mmap_open_or_create() {
    let dir = TempDir::new().unwrap();
    mmap_store::MmapColumnStore::mmap_open_or_create(dir.path().join("c.bin"), 2, None).unwrap();
}

#[test]
fn mmap_append() {
    let dir = TempDir::new().unwrap();
    let mut store =
        mmap_store::MmapColumnStore::mmap_open_or_create(dir.path().join("c.bin"), 2, None).unwrap();
    store.mmap_append(&array![[0.0, 0.0]].view()).unwrap();
}

#[test]
fn mmap_row_slice() {
    let dir = TempDir::new().unwrap();
    let mut store =
        mmap_store::MmapColumnStore::mmap_open_or_create(dir.path().join("c.bin"), 2, None).unwrap();
    store.mmap_append(&array![[0.0, 1.0]].view()).unwrap();
    assert_eq!(store.mmap_row_slice(0).unwrap()[1], 1.0);
}

#[test]
fn mmap_gather() {
    let dir = TempDir::new().unwrap();
    let mut store =
        mmap_store::MmapColumnStore::mmap_open_or_create(dir.path().join("c.bin"), 2, None).unwrap();
    store.mmap_append(&array![[0.0, 0.0], [1.0, 0.0]].view()).unwrap();
    assert_eq!(store.mmap_gather(&[1]).unwrap().nrows(), 1);
}
