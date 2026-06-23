//! Kiss static coverage: disk HNSW backend and graph module symbols.

const DISK_HNSW_SRC: &str = include_str!("../src/disk_hnsw/enn_backend.rs");
const DISK_OBSERVATION_SRC: &str = include_str!("../src/backend/disk_observation.rs");
const HNSW_GRAPH_SRC: &str = include_str!("../src/disk_hnsw/hnsw.rs");
const HNSW_STORE_SRC: &str = include_str!("../src/disk_hnsw/store.rs");

#[test]
fn kiss_hnsw_search_units_are_linked() {
    use ennbo::disk_hnsw::store::RamGraph;
    use ndarray::array;
    use rand::SeedableRng;
    use tempfile::TempDir;
    let vecs = vec![vec![0.0f32, 0.0], vec![1.0, 0.0]];
    let _ = ennbo::disk_hnsw::hnsw::brute_force_topk(&vecs, &[0.0, 0.0], 1);
    let dir = TempDir::new().unwrap();
    let mut store =
        ennbo::knn::MmapColumnStore::mmap_open_or_create(dir.path().join("x.bin"), 2, None).unwrap();
    store
        .mmap_append(&array![[0.0, 0.0], [1.0, 0.0]].view())
        .unwrap();
    let _ = ennbo::disk_hnsw::hnsw::brute_force_topk_mmap(&store, 0, 2, &[0.0, 0.0], 1, false, &[1.0, 1.0]).unwrap();
    let _ = ennbo::disk_hnsw::hnsw::merge_topk_candidates(
        &store,
        &[0.0, 0.0],
        &[(0, 0.0)],
        &[(1, 1.0)],
        1,
        2,
        false,
        false,
        &[1.0, 1.0],
    )
    .unwrap();
    let mut graph = RamGraph::new(2);
    let mut header = ennbo::disk_hnsw::hnsw::HnswHeader {
        entry_point: 0,
        max_level: 0,
        num_dim: 2,
    };
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    ennbo::disk_hnsw::hnsw::insert(&mut graph, &mut header, 0, &[0.0, 0.0], &mut rng);
    let _ = ennbo::disk_hnsw::hnsw::search::<RamGraph>(&graph, &header, &[0.0, 0.0], 1, 8, 2);
}

#[test]
fn kiss_disk_hnsw_backend_names_in_source() {
    for name in [
        "DiskHnswEnnBackend",
        "open_or_create_graph",
        "disk_obs::load_num_obs",
        "INDEX_BACKEND",
    ] {
        assert!(DISK_HNSW_SRC.contains(name), "missing {name}");
    }
}

#[test]
fn kiss_disk_observation_and_graph_names_in_source() {
    for name in [
        "load_indexed_rows",
        "write_metadata",
        "search_layer",
        "check_append_row_limit",
        "truncate_nodes",
        "NodeLayout",
    ] {
        assert!(
            DISK_OBSERVATION_SRC.contains(name)
                || HNSW_GRAPH_SRC.contains(name)
                || HNSW_STORE_SRC.contains(name),
            "missing {name}"
        );
    }
}
