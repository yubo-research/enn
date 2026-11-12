use super::*;
use crate::index::sync::IncrementalIndex;
use crate::index::tree_store::{META_FILE, META_WORDS};
use crate::mmap_store::MmapColumnStore;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use tempfile::TempDir;

const INTERNAL_TAG: u32 = 2 << 24;

fn strip_routes(index_dir: &Path) {
    let path = index_dir.join(META_FILE);
    let mut words: Vec<u32> = fs::read(&path)
        .unwrap()
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    for rec in words.chunks_exact_mut(META_WORDS) {
        if rec[0] & 0xFF00_0000 == INTERNAL_TAG {
            rec[META_WORDS - 32..].fill(0);
        }
    }
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    fs::write(&path, bytes).unwrap();
    let stamp = index_dir.join(STAMP_FILE);
    let text = fs::read_to_string(&stamp)
        .unwrap()
        .replace(",\"routes\":true", "");
    fs::write(&stamp, text).unwrap();
}

fn routes_match(t: &Tree) -> bool {
    let s = &t.store;
    (0..s.num_pages() as u32)
        .filter(|&p| !s.is_leaf(p))
        .all(|p| {
            let want: Vec<f32> = s.ids(p).iter().map(|&c| s.radius(c)).collect();
            s.child_radii(p) == &want[..]
        })
}

#[test]
fn an_index_without_routes_rebuilds_them_once_on_open() {
    let dir = TempDir::new().unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(5);
    let x = ndarray::Array2::from_shape_fn((3000, 2), |_| rng.gen::<f64>());
    let mut store =
        MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), 2, None).unwrap();
    store.mmap_append(&x.view()).unwrap();
    let mut idx = IncrementalIndex::new(dir.path().join("index"));
    idx.persist_to_disk_for_backend(&store, 2, false, &[1.0; 2], dir.path(), 1)
        .unwrap();
    let index_dir = dir.path().join("index");
    strip_routes(&index_dir);
    let t = Tree::open_stamped(&index_dir, false, &[1.0; 2])
        .unwrap()
        .unwrap();
    assert!(t.rows_cached() && routes_match(&t));
    let text = fs::read_to_string(index_dir.join(STAMP_FILE)).unwrap();
    assert!(text.contains("\"routes\":true"), "migration must restamp");
    drop(t);
    let again = Tree::open_stamped(&index_dir, false, &[1.0; 2])
        .unwrap()
        .unwrap();
    assert!(routes_match(&again));
}

#[test]
fn persist_after_add_skips_pages_bin_and_reopens_identically() {
    let dir = TempDir::new().unwrap();
    let mut rng = ChaCha8Rng::seed_from_u64(9);
    let x = ndarray::Array2::from_shape_fn((900, 2), |_| rng.gen::<f64>());
    let mut store =
        MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), 2, None).unwrap();
    store.mmap_append(&x.view()).unwrap();
    let index_dir = dir.path().join("index");
    let mut idx = IncrementalIndex::new(index_dir.clone());
    idx.persist_to_disk_for_backend(&store, 2, false, &[1.0; 2], dir.path(), 1)
        .unwrap();
    assert!(!index_dir.join("pages.bin").exists());

    // Add one row through the adopt-then-persist path, as a session does.
    let row = ndarray::Array2::from_shape_fn((1, 2), |_| rng.gen::<f64>());
    store.mmap_append(&row.view()).unwrap();
    let mut fast = IncrementalIndex::new(index_dir.clone());
    assert!(fast.adopt_persisted_scaled(false, &[1.0; 2]).unwrap());
    fast.persist_to_disk_for_backend(&store, 2, false, &[1.0; 2], dir.path(), 1)
        .unwrap();
    assert!(
        !index_dir.join("pages.bin").exists(),
        "persist after an add must not write pages.bin"
    );

    // Reopening through the stamp must reproduce the same tree, hence the same predictions.
    let live = fast.tree.as_ref().unwrap().page_bytes();
    let reopened = Tree::open_stamped(&index_dir, false, &[1.0; 2])
        .unwrap()
        .expect("stamp current after persist");
    assert_eq!(reopened.header.indexed_rows, 901);
    assert_eq!(reopened.page_bytes(), live);
}
