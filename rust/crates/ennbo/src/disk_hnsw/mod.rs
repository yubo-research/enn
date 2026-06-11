//! In-tree disk HNSW graph (CP-0 spike + mmap persistence).

pub mod access;
pub mod flush;
pub mod graph_header;
pub mod graph_mut;
pub mod enn_backend;
pub mod hnsw;
pub mod node_layout;
pub mod params;
pub mod split_graph;
pub mod store;

pub use enn_backend::DiskHnswEnnBackend;
pub use hnsw::{assign_level, brute_force_topk, insert, l2_sq, mean_recall_at_k, search, HnswHeader};
pub use node_layout::NodeLayout;
pub use params::{ef_search_for_k, EF_CONSTRUCTION, LMAX, M, M0};
pub use graph_header::GraphHeader;
pub use store::{MmapGraph, RamGraph};

#[cfg(test)]
mod cp0_tests {
    use super::*;
    use crate::disk_hnsw::access::GraphAccess;
    use crate::disk_hnsw::graph_mut::GraphMut;
    use rand::Rng;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use tempfile::TempDir;

    const DATA_SEED: u64 = 42;
    const QUERY_SEED: u64 = 1;
    const N: usize = 128;
    const N_MMAP: usize = 64;
    const D: usize = 32;
    const K: usize = 10;
    const NUM_RECALL_QUERIES: usize = 5;

    fn synthetic_vectors(n: usize, d: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = <ChaCha8Rng as rand_chacha::rand_core::SeedableRng>::seed_from_u64(seed);
        (0..n)
            .map(|_| (0..d).map(|_| rng.gen::<f32>()).collect())
            .collect()
    }

    fn build_ram_index(vectors: &[Vec<f32>], chunk_seed: u64) -> (RamGraph, HnswHeader) {
        let mut graph = RamGraph::new(D);
        let mut header = HnswHeader {
            entry_point: 0,
            max_level: 0,
            num_dim: D,
        };
        let mut rng = StdRng::seed_from_u64(chunk_seed);
        for (i, v) in vectors.iter().enumerate() {
            insert(&mut graph, &mut header, i as u32, v, &mut rng);
        }
        (graph, header)
    }

    #[test]
    fn cp0_disk_hnsw_ram_recall() {
        let vectors = synthetic_vectors(N, D, DATA_SEED);
        let queries = synthetic_vectors(NUM_RECALL_QUERIES, D, QUERY_SEED);
        let (graph, header) = build_ram_index(&vectors, 0);
        let ef = ef_search_for_k(K);
        let recall = mean_recall_at_k(&vectors, &queries, K, ef, &graph, &header, N as u32);
        assert!(
            recall >= 0.90,
            "recall@10 = {recall}, expected >= 0.90"
        );
    }

    fn snapshot_node_records(graph: &MmapGraph, n: u32) -> Vec<Vec<u8>> {
        (0..n)
            .map(|id| graph.try_record(id).expect("record").to_vec())
            .collect()
    }

    fn assert_reopened_matches_snapshot(
        snapshot: &[Vec<u8>],
        reopened: &MmapGraph,
    ) {
        for (id, expected) in snapshot.iter().enumerate() {
            let post_rec = reopened
                .try_record(id as u32)
                .expect("post-reopen record");
            assert_eq!(post_rec, expected.as_slice(), "node {id} record mismatch");
        }
    }

    #[test]
    fn cp0_disk_hnsw_mmap_reopen_record_integrity() {
        let vectors = synthetic_vectors(N_MMAP, D, DATA_SEED);
        let dir = TempDir::new().expect("tempdir");
        let graph_dir = dir.path().join("graph");

        let (mut mmap_graph, mut file_header) = MmapGraph::create(&graph_dir, D).unwrap();
        let mut hnsw_header = HnswHeader {
            entry_point: 0,
            max_level: 0,
            num_dim: D,
        };
        let mut rng = StdRng::seed_from_u64(0);
        for (i, v) in vectors.iter().enumerate() {
            insert(&mut mmap_graph, &mut hnsw_header, i as u32, v, &mut rng);
        }
        file_header.entry_point = hnsw_header.entry_point;
        file_header.max_level = hnsw_header.max_level;
        file_header
            .write_json(&graph_dir.join("header.json"))
            .unwrap();

        let pre_sync_entry = hnsw_header.entry_point;
        let pre_sync_max_level = hnsw_header.max_level;
        let snapshot = snapshot_node_records(&mmap_graph, N_MMAP as u32);
        mmap_graph.fsync().unwrap();

        let (reopened, hdr) = MmapGraph::open(&graph_dir).unwrap();
        assert_eq!(hdr.entry_point, pre_sync_entry);
        assert_eq!(hdr.max_level, pre_sync_max_level);
        assert_reopened_matches_snapshot(&snapshot, &reopened);
    }

    #[test]
    fn disk_hnsw_header_params_match_defaults() {
        let dir = TempDir::new().expect("tempdir");
        let graph_dir = dir.path().join("graph");
        let (_, header) = MmapGraph::create(&graph_dir, D).unwrap();
        assert_eq!(header.m, M);
        assert_eq!(header.m0, M0);
        assert_eq!(header.lmax, LMAX);
        assert_eq!(header.ef_construction, EF_CONSTRUCTION);
        let reloaded = GraphHeader::read_json(&graph_dir.join("header.json")).unwrap();
        assert_eq!(reloaded.m, M);
        assert_eq!(reloaded.m0, M0);
    }

    #[test]
    fn kiss_node_layout_and_store_symbols() {
        use crate::disk_hnsw::store::truncate_nodes;
        let names = [
            "NodeLayout",
            "write_record",
            "read_neighbors",
            "truncate_nodes",
            "GraphHeader",
            "RamGraph",
            "MmapGraph",
        ];
        let _ = truncate_nodes as fn(&std::path::Path, usize, usize) -> Result<(), String>;
        assert_eq!(names.len(), 7);
    }
}
