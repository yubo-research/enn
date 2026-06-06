//! Composite graph: mmap prefix + RAM tail for batch flush inserts.
#![allow(dead_code)]

use std::collections::HashMap;

use crate::disk_hnsw::access::GraphAccess;
use crate::disk_hnsw::graph_mut::GraphMut;
use crate::disk_hnsw::node_layout::NodeLayout;
use crate::disk_hnsw::store::{MmapGraph, RamGraph};

/// Deferred neighbor patches for mmap prefix nodes during inline RAM inserts.
#[derive(Default)]
pub(crate) struct PrefixNeighborOverlay {
    patches: HashMap<(u32, u8), Vec<u32>>,
}

impl PrefixNeighborOverlay {
    pub(crate) fn clear(&mut self) {
        self.patches.clear();
    }

    pub(crate) fn patches(&self) -> &HashMap<(u32, u8), Vec<u32>> {
        &self.patches
    }
}

pub(crate) struct SplitGraphMut<'a> {
    mmap: &'a MmapGraph,
    ram: &'a mut RamGraph,
    split: u32,
    overlay: &'a mut PrefixNeighborOverlay,
}

impl<'a> SplitGraphMut<'a> {
    pub(crate) fn new(
        mmap: &'a MmapGraph,
        ram: &'a mut RamGraph,
        split: u32,
        overlay: &'a mut PrefixNeighborOverlay,
    ) -> Self {
        Self {
            mmap,
            ram,
            split,
            overlay,
        }
    }

    fn in_ram(&self, id: u32) -> bool {
        id >= self.split
    }
}

impl GraphAccess for SplitGraphMut<'_> {
    fn layout(&self) -> &NodeLayout {
        self.mmap.layout()
    }

    fn num_nodes(&self) -> u32 {
        self.mmap.num_nodes().max(self.ram.num_nodes())
    }

    fn node_level(&self, id: u32) -> u8 {
        if self.in_ram(id) {
            self.ram.node_level(id)
        } else {
            self.mmap.node_level(id)
        }
    }

    fn vector(&self, id: u32) -> Vec<f32> {
        if self.in_ram(id) {
            self.ram.vector(id)
        } else {
            self.mmap.vector(id)
        }
    }

    fn neighbors(&self, id: u32, layer: u8) -> Vec<u32> {
        if self.in_ram(id) {
            self.ram.neighbors(id, layer)
        } else if let Some(nbrs) = self.overlay.patches.get(&(id, layer)) {
            nbrs.clone()
        } else {
            self.mmap.neighbors(id, layer)
        }
    }

    fn try_record(&self, id: u32) -> Option<&[u8]> {
        if self.in_ram(id) {
            self.ram.try_record(id)
        } else {
            self.mmap.try_record(id)
        }
    }
}

impl GraphMut for SplitGraphMut<'_> {
    fn write_node(&mut self, id: u32, level: u8, vector: &[f32]) {
        if self.in_ram(id) {
            self.ram.write_node(id, level, vector);
        } else {
            panic!("SplitGraphMut::write_node on mmap prefix id {id}");
        }
    }

    fn set_neighbors(&mut self, id: u32, layer: u8, neighbors: &[u32]) {
        if self.in_ram(id) {
            self.ram.set_neighbors(id, layer, neighbors);
        } else {
            self.overlay
                .patches
                .insert((id, layer), neighbors.to_vec());
        }
    }

    fn read_record_mut(&mut self, id: u32) -> &mut [u8] {
        if self.in_ram(id) {
            self.ram.read_record_mut(id)
        } else {
            panic!("SplitGraphMut::read_record_mut on mmap prefix id {id}");
        }
    }

    fn read_record(&self, id: u32) -> &[u8] {
        if self.in_ram(id) {
            self.ram.read_record(id)
        } else {
            self.mmap.read_record(id)
        }
    }

    fn fsync(&mut self) -> Result<(), String> {
        Ok(())
    }
}

#[cfg(test)]
mod split_graph_tests {
    use super::*;
    use crate::disk_hnsw::hnsw::{self, HnswHeader};
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use tempfile::TempDir;

    #[test]
    fn split_graph_mut_inline_insert_and_merge() {
        let dir = TempDir::new().expect("tempdir");
        let graph_dir = dir.path().join("graph");
        let (mut mmap, _) = MmapGraph::create(&graph_dir, 2).unwrap();
        let mut ram = RamGraph::new(2);
        let mut overlay = PrefixNeighborOverlay::default();
        let mut header = HnswHeader {
            entry_point: 0,
            max_level: 0,
            num_dim: 2,
        };
        let mut rng = StdRng::seed_from_u64(0);
        hnsw::insert(&mut mmap, &mut header, 0, &[0.0, 0.0], &mut rng);
        let mut split = SplitGraphMut::new(&mmap, &mut ram, 1, &mut overlay);
        hnsw::insert(&mut split, &mut header, 1, &[1.0, 0.0], &mut rng);
        assert_eq!(split.vector(1), vec![1.0, 0.0]);
        for ((id, layer), neighbors) in overlay.patches().clone() {
            GraphMut::set_neighbors(&mut mmap, id, layer, &neighbors);
        }
        mmap.merge_ram_tail(&ram, 1, 2).unwrap();
        assert_eq!(mmap.vector(1), vec![1.0, 0.0]);
    }

    #[test]
    fn prefix_overlay_defers_mmap_neighbor_write() {
        let dir = TempDir::new().expect("tempdir");
        let graph_dir = dir.path().join("graph");
        let (mmap, _) = MmapGraph::create(&graph_dir, 2).unwrap();
        let mut ram = RamGraph::new(2);
        let mut overlay = PrefixNeighborOverlay::default();
        let mut split = SplitGraphMut::new(&mmap, &mut ram, 1, &mut overlay);
        split.set_neighbors(0, 0, &[1]);
        assert_eq!(split.neighbors(0, 0), vec![1]);
        assert_eq!(overlay.patches.get(&(0, 0)).map(Vec::as_slice), Some(&[1][..]));
    }
}
