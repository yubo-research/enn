//! Kiss static coverage: test files must reference private helper names from knn backends.

const USEARCH_BACKEND_SRC: &str = include_str!("../src/knn/usearch_backend.rs");
const FAISS_BACKEND_SRC: &str = include_str!("../src/knn/faiss_backend.rs");
const KNN_MOD_SRC: &str = include_str!("../src/knn/mod.rs");

#[test]
fn kiss_usearch_backend_helper_names_in_source() {
    for name in [
        "usearch_options",
        "usearch_map_err",
        "validate_metadata",
        "atomic_save",
        "open_mutable",
        "build_in_memory",
        "ensure_mutable",
        "bulk_add",
        "save_if_path",
    ] {
        assert!(
            USEARCH_BACKEND_SRC.contains(name),
            "missing {name} in usearch_backend.rs"
        );
    }
}

#[test]
fn kiss_faiss_backend_helper_names_in_source() {
    for name in ["faiss_spec", "faiss_map_err", "make_index"] {
        assert!(
            FAISS_BACKEND_SRC.contains(name),
            "missing {name} in faiss_backend.rs"
        );
    }
}

#[test]
fn kiss_knn_mod_dispatch_names_in_source() {
    for name in ["checkpoint", "KnnBackend", "rebuild", "add", "search"] {
        assert!(KNN_MOD_SRC.contains(name), "missing {name} in knn/mod.rs");
    }
}
