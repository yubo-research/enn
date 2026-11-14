use super::*;
use tempfile::TempDir;

fn store(dir: &TempDir, d: usize) -> TreeStore {
    TreeStore::create(dir.path(), d).unwrap()
}

/// Leaves hold int8 codes: decoded rows are within half a code step per coordinate.
fn assert_rows(s: &TreeStore, id: u32, want: &[f32]) {
    let got = s.leaf_rows(id);
    let amax = want
        .chunks(s.num_dim())
        .map(|r| r.iter().fold(0.0f32, |m, v| m.max(v.abs())));
    let tol = amax.fold(0.0f32, f32::max) / 127.0;
    assert_eq!(got.len(), want.len());
    assert!(
        got.iter().zip(want).all(|(g, w)| (g - w).abs() <= tol),
        "{got:?} vs {want:?}"
    );
}

#[test]
fn pages_keep_their_fields_entries_and_kind() {
    let dir = TempDir::new().unwrap();
    let mut s = store(&dir, 2);
    assert_eq!((s.num_dim(), s.num_pages()), (2, 0));
    let leaf = s.alloc(Kind::Leaf).unwrap();
    let node = s.alloc(Kind::Internal).unwrap();
    assert!(s.is_leaf(leaf) && !s.is_leaf(node));
    assert_eq!((s.len(leaf), s.count(leaf), s.parent(leaf)), (0, 0, None));
    assert_eq!(s.radius(leaf), f32::INFINITY);
    s.push_entry(leaf, 7, &[1.0, 2.0]);
    s.push_entry(leaf, 9, &[3.0, 4.0]);
    assert_eq!(s.ids(leaf), &[7, 9]);
    assert_rows(&s, leaf, &[1.0, 2.0, 3.0, 4.0]);
    s.set_entries(node, &[leaf], &[0.5, 0.5]);
    s.set_count(node, 2);
    s.set_radius(node, 1.5);
    s.set_parent(leaf, Some(node));
    assert_eq!(
        (s.ids(node), s.count(node), s.radius(node)),
        (&[leaf][..], 2, 1.5)
    );
    assert_eq!(s.parent(leaf), Some(node));
    s.set_kind(node, Kind::Leaf);
    assert!(s.is_leaf(node) && s.len(node) == 1);
}

#[test]
fn growing_past_the_first_mapping_keeps_every_page() {
    let dir = TempDir::new().unwrap();
    let mut s = store(&dir, 3);
    for i in 0..1000u32 {
        let id = s.alloc(Kind::Leaf).unwrap();
        s.push_entry(id, i, &[i as f32, 0.0, 1.0]);
        s.set_count(id, i as usize);
    }
    assert_eq!(s.num_pages(), 1000);
    assert!(s.file_bytes() >= 1000 * SLOT_ENTRIES * 4 * 4);
    assert!((0..1000u32).all(|i| s.ids(i) == [i]
        && (s.leaf_rows(i)[0] - i as f32).abs() <= 1e-2 * (i as f32 + 1.0)
        && s.count(i) == i as usize));
    let copy = s.try_clone().unwrap();
    s.set_count(3, 99);
    assert_eq!((copy.count(3), copy.ids(999)), (3, &[999u32][..]));
}

#[test]
fn a_small_budget_releases_pages_and_keeps_the_data() {
    let dir = TempDir::new().unwrap();
    let mut s = store(&dir, 8);
    s.set_resident_budget(16 << 10);
    for i in 0..250u32 {
        let id = s.alloc(Kind::Leaf).unwrap();
        s.push_entry(id, i, &[i as f32; 8]);
        assert!(s.residency.estimate() <= 16 << 10);
    }
    assert!(s.residency.tracking());
    s.release_resident();
    assert_eq!(s.residency.estimate(), 0);
    assert!((0..250u32).all(|i| s.ids(i) == [i]));
    (0..250u32).for_each(|i| assert_rows(&s, i, &[i as f32; 8]));
    s.set_resident_budget(usize::MAX);
    assert!(!s.residency.tracking());
    let copy = s.try_clone().unwrap();
    assert!((0..250u32).all(|i| copy.ids(i) == [i]));
    (0..250u32).for_each(|i| assert_rows(&copy, i, &[i as f32; 8]));
}

#[test]
fn lowering_the_budget_turns_on_residency_tracking() {
    let dir = TempDir::new().unwrap();
    let mut s = store(&dir, 2);
    let id = s.alloc(Kind::Leaf).unwrap();
    s.push_entry(id, 5, &[1.0, 2.0]);
    s.release_resident();
    assert_eq!(s.ids(id), &[5], "releasing pages keeps the data");
    s.set_resident_budget(1024);
    assert!(s.residency.tracking());
    assert_eq!(s.ids(id), &[5u32][..]);
    assert_rows(&s, id, &[1.0, 2.0]);
}

#[test]
#[should_panic(expected = "out of range")]
fn a_page_past_the_end_panics() {
    let dir = TempDir::new().unwrap();
    store(&dir, 1).count(0);
}

#[test]
fn codes_follow_every_row_write() {
    let dir = TempDir::new().unwrap();
    let mut s = store(&dir, 3);
    let leaf = s.alloc(Kind::Leaf).unwrap();
    s.push_entry(leaf, 1, &[1.0, -2.0, 3.0]);
    s.push_entry(leaf, 2, &[0.5, 0.25, -8.0]);
    assert!(s.codes_valid(leaf));
    assert_rows(&s, leaf, &[1.0, -2.0, 3.0, 0.5, 0.25, -8.0]);
    assert!(s.leaf_err(leaf) > 0.0);
    s.set_entries(leaf, &[4], &[-1.0, 0.0, 1.0]);
    assert!(s.codes_valid(leaf));
    assert_rows(&s, leaf, &[-1.0, 0.0, 1.0]);
    for i in 0..200 {
        let p = s.alloc(Kind::Leaf).unwrap();
        s.push_entry(p, i, &[i as f32, 1.0, -1.0]);
    }
    let copy = s.try_clone().unwrap();
    assert!((0..copy.num_pages() as u32).all(|p| copy.codes_valid(p)));
    assert!((0..copy.num_pages() as u32).all(|p| copy.leaf_rows(p) == s.leaf_rows(p)));
}

#[test]
fn parents_keep_their_children_radii_and_rebuild_restores_them() {
    let dir = TempDir::new().unwrap();
    let mut s = store(&dir, 2);
    let node = s.alloc(Kind::Internal).unwrap();
    let (a, b) = (s.alloc(Kind::Leaf).unwrap(), s.alloc(Kind::Leaf).unwrap());
    s.set_radius(a, 0.5);
    s.set_entries(node, &[a], &[0.0, 0.0]);
    s.set_parent(a, Some(node));
    assert_eq!(s.child_radii(node), &[0.5]);
    s.set_radius(b, 1.5);
    s.push_entry(node, b, &[1.0, 1.0]);
    s.set_parent(b, Some(node));
    s.set_radius(a, 2.5);
    assert_eq!(s.child_radii(node), &[2.5, 1.5]);
    s.set_route(node, 0, 0.0);
    s.set_route(node, 1, 0.0);
    s.rebuild_routes();
    assert_eq!(s.child_radii(node), &[2.5, 1.5]);
    assert_eq!(s.ids(node), &[a, b]);
}
