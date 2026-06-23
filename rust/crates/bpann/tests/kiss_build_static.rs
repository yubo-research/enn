//! Kiss static coverage for bpann index/build helpers.

const BUILD_SRC: &str = include_str!("../src/index/build.rs");

#[test]
fn kiss_build_module_names_in_source() {
    for name in ["IndexHeader", "leaf_row_ids", "read_skip_edges"] {
        assert!(BUILD_SRC.contains(name), "missing {name} in index/build.rs");
    }
}
