use super::*;
use tempfile::TempDir;

fn store(dir: &TempDir, d: usize) -> TreeStore {
    TreeStore::create(dir.path(), d).unwrap()
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
    assert_eq!(s.block(leaf), &[1.0, 2.0, 3.0, 4.0]);
    s.entry_mut(leaf, 1)[0] = 5.0;
    assert_eq!(s.block(leaf), &[1.0, 2.0, 5.0, 4.0]);
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
    s.block_mut(leaf).iter_mut().for_each(|v| *v = 0.0);
    assert_eq!(s.block(leaf), &[0.0; 4]);
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
    assert!((0..1000u32)
        .all(|i| s.ids(i) == [i] && s.block(i)[0] == i as f32 && s.count(i) == i as usize));
    let copy = s.try_clone().unwrap();
    s.set_count(3, 99);
    assert_eq!((copy.count(3), copy.ids(999)), (3, &[999u32][..]));
}

#[test]
fn a_small_budget_releases_pages_and_keeps_the_data() {
    let dir = TempDir::new().unwrap();
    let mut s = store(&dir, 8);
    s.set_resident_budget(256 << 10);
    for i in 0..4000u32 {
        let id = s.alloc(Kind::Leaf).unwrap();
        s.push_entry(id, i, &[i as f32; 8]);
        assert!(s.residency.estimate() <= 256 << 10);
    }
    assert!(s.residency.tracking());
    s.release_resident();
    assert_eq!(s.residency.estimate(), 0);
    assert!((0..4000u32).all(|i| s.ids(i) == [i] && s.block(i) == [i as f32; 8]));
    s.set_resident_budget(usize::MAX);
    assert!(!s.residency.tracking());
    let copy = s.try_clone().unwrap();
    assert!((0..4000u32).all(|i| copy.ids(i) == [i] && copy.block(i) == [i as f32; 8]));
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
    assert_eq!((s.ids(id), s.block(id)), (&[5u32][..], &[1.0f32, 2.0][..]));
}

#[test]
#[should_panic(expected = "out of range")]
fn a_page_past_the_end_panics() {
    let dir = TempDir::new().unwrap();
    store(&dir, 1).count(0);
}

fn codes_match_rows(s: &TreeStore, id: u32) -> bool {
    let d = s.num_dim();
    let rows = s.block(id).to_vec();
    rows.chunks_exact(d).enumerate().all(|(slot, v)| {
        let (scale, _, c) = crate::index::tree_codes::decode_header(s.code_row(id, slot));
        v.iter()
            .zip(c)
            .all(|(&x, &q)| (x - scale * f32::from(q)).abs() <= 0.5 * scale + 1e-6)
    })
}

#[test]
fn codes_follow_every_row_write() {
    let dir = TempDir::new().unwrap();
    let mut s = store(&dir, 3);
    let leaf = s.alloc(Kind::Leaf).unwrap();
    s.push_entry(leaf, 1, &[1.0, -2.0, 3.0]);
    s.push_entry(leaf, 2, &[0.5, 0.25, -8.0]);
    assert!(s.codes_valid(leaf) && codes_match_rows(&s, leaf));
    s.entry_mut(leaf, 0).copy_from_slice(&[9.0, 9.0, 9.0]);
    assert!(!s.codes_valid(leaf));
    s.set_ids_only(leaf, &[1, 2]);
    assert!(s.codes_valid(leaf) && codes_match_rows(&s, leaf));
    s.set_entries(leaf, &[4], &[-1.0, 0.0, 1.0]);
    assert!(s.codes_valid(leaf) && codes_match_rows(&s, leaf));
    for i in 0..200 {
        let p = s.alloc(Kind::Leaf).unwrap();
        s.push_entry(p, i, &[i as f32, 1.0, -1.0]);
    }
    let copy = s.try_clone().unwrap();
    assert!((0..copy.num_pages() as u32).all(|p| copy.codes_valid(p) && codes_match_rows(&copy, p)));
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
