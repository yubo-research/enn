//! Mutable graph store access (insert + neighbor patching).

use crate::disk_hnsw::access::GraphAccess;

pub trait GraphMut: GraphAccess {
    fn write_node(&mut self, id: u32, level: u8, vector: &[f32]);
    fn set_neighbors(&mut self, id: u32, layer: u8, neighbors: &[u32]);
    fn read_record_mut(&mut self, id: u32) -> &mut [u8];
    fn read_record(&self, id: u32) -> &[u8];
    fn fsync(&mut self) -> Result<(), String>;
}
