//! Read-only graph store access for in-tree HNSW search.

use crate::disk_hnsw::hnsw::l2_sq;
use crate::disk_hnsw::node_layout::NodeLayout;

pub trait GraphAccess {
    fn layout(&self) -> &NodeLayout;
    fn num_nodes(&self) -> u32;
    fn node_level(&self, id: u32) -> u8;
    fn vector(&self, id: u32) -> Vec<f32>;
    fn neighbors(&self, id: u32, layer: u8) -> Vec<u32>;

    fn vector_l2_sq(&self, id: u32, query: &[f32]) -> f32 {
        let layout = self.layout();
        if let Some(record) = self.try_record(id) {
            return layout.l2_sq_from_record(record, query);
        }
        l2_sq(query, &self.vector(id))
    }

    /// When implemented, enables allocation-free distance in `vector_l2_sq`.
    fn try_record(&self, id: u32) -> Option<&[u8]> {
        let _ = id;
        None
    }
}
