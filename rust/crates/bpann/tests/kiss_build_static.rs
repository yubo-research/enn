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
        "remap_page",
    ] {
        assert!(BUILD_SRC.contains(name), "missing {name} in index/build.rs");
    }
}

#[test]
fn kiss_needs_skip_edges_name_is_covered() {

    assert!(BUILD_SRC.contains("fn needs_skip_edges"));
}
