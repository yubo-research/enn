//! Read-only graph store access for in-tree HNSW search.

use crate::disk_hnsw::node_layout::NodeLayout;

pub trait GraphAccess {
    fn layout(&self) -> &NodeLayout;
    fn num_nodes(&self) -> u32;
    fn node_level(&self, id: u32) -> u8;
    fn vector(&self, id: u32) -> Vec<f32>;
    fn neighbors(&self, id: u32, layer: u8) -> Vec<u32>;
}
