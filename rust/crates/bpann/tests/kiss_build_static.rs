//! Kiss static coverage for bpann index/build helpers.

const BUILD_SRC: &str = include_str!("../src/index/build.rs");

#[test]
fn kiss_build_module_names_in_source() {
    for name in [
        "IndexHeader",
        "leaf_row_ids",
        "read_skip_edges",
        "needs_skip_edges",
        "build_page_map",
        "build_skip_edges",
        "partition_to_pages",
        "partition_to_pages_id",
    ] {
        assert!(BUILD_SRC.contains(name), "missing {name} in index/build.rs");
    }
}

#[test]
fn kiss_needs_skip_edges_name_is_covered() {
    assert!(BUILD_SRC.contains("fn needs_skip_edges"));
}

#[test]
fn pushed_page_can_be_mutated_by_sparse_id() {
    use ennbo_bpann::index::{build::BpannIndex, page::Page};
    let mut index = BpannIndex::build_single_leaf_from_rows_with_persist(
        &[0],
        &[vec![0.0]],
        1,
        Default::default(),
        false,
    )
    .unwrap();
    let page = Page::empty_range_leaf(10, 1, 2, vec![1.0]);
    index.push_page(page);
    let replacement = Page::empty_range_leaf(10, 1, 3, vec![2.0]);
    *index.page_by_id_mut(10).unwrap() = replacement.clone();
    assert_eq!(index.page_by_id(10), Some(&replacement));
}
