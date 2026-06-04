//! Default HNSW parameters for disk graph (Faiss/HNSW32 convention).

pub const M: usize = 16;
pub const M0: usize = 32;
pub const LMAX: usize = 16;
pub const EF_CONSTRUCTION: usize = 200;
pub const GRAPH_FORMAT_VERSION: u32 = 1;
pub const EMPTY_NEIGHBOR: u32 = u32::MAX;

pub fn max_neighbors(layer: u8) -> usize {
    if layer == 0 { M0 } else { M }
}

pub fn ef_search_for_k(k: usize) -> usize {
    64.max(2 * k)
}

#[cfg(test)]
mod params_tests {
    use super::*;

    #[test]
    fn max_neighbors_layer0_vs_upper() {
        assert_eq!(max_neighbors(0), M0);
        assert_eq!(max_neighbors(3), M);
        assert_eq!(ef_search_for_k(10), 64);
        assert_eq!(ef_search_for_k(100), 200);
    }
}
