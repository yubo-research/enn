//! RAM and mmap-backed graph stores.

use std::fs::{File, OpenOptions};
use std::path::Path;

use memmap2::MmapMut;

use crate::disk_hnsw::access::GraphAccess;
use crate::disk_hnsw::graph_header::GraphHeader;
use crate::disk_hnsw::graph_mut::GraphMut;
use crate::disk_hnsw::node_layout::NodeLayout;
use crate::disk_hnsw::params::LMAX;

pub struct RamGraph {
    layout: NodeLayout,
    records: Vec<Vec<u8>>,
}

impl RamGraph {
    pub fn new(num_dim: usize) -> Self {
        Self {
            layout: NodeLayout::new(num_dim),
            records: Vec::new(),
        }
    }

    fn ensure_id(&mut self, id: u32) {
        let need = id as usize + 1;
        while self.records.len() < need {
            self.records.push(vec![0u8; self.layout.record_stride]);
        }
    }

}

impl GraphAccess for RamGraph {
    fn layout(&self) -> &NodeLayout {
        &self.layout
    }

    fn num_nodes(&self) -> u32 {
        self.records.len() as u32
    }

    fn node_level(&self, id: u32) -> u8 {
        self.layout.read_level(&self.records[id as usize])
    }

    fn vector(&self, id: u32) -> Vec<f32> {
        self.layout.read_vector(&self.records[id as usize])
    }

    fn neighbors(&self, id: u32, layer: u8) -> Vec<u32> {
        self.layout
            .read_neighbors(&self.records[id as usize], layer)
    }

    fn try_record(&self, id: u32) -> Option<&[u8]> {
        Some(&self.records[id as usize])
    }
}

impl GraphMut for RamGraph {
    fn write_node(&mut self, id: u32, level: u8, vector: &[f32]) {
        self.ensure_id(id);
        let empty_neighbors: Vec<Vec<u32>> = (0..LMAX).map(|_| Vec::new()).collect();
        self.layout.write_record(
            &mut self.records[id as usize],
            level,
            vector,
            &empty_neighbors,
        );
    }

    fn set_neighbors(&mut self, id: u32, layer: u8, neighbors: &[u32]) {
        self.ensure_id(id);
        self.layout
            .write_neighbors_layer(&mut self.records[id as usize], layer, neighbors);
    }

    fn read_record_mut(&mut self, id: u32) -> &mut [u8] {
        self.ensure_id(id);
        &mut self.records[id as usize]
    }

    fn read_record(&self, id: u32) -> &[u8] {
        &self.records[id as usize]
    }

    fn fsync(&mut self) -> Result<(), String> {
        Ok(())
    }
}

pub struct MmapGraph {
    layout: NodeLayout,
    file: File,
    mmap: MmapMut,
    num_nodes: u32,
}

impl MmapGraph {
    pub fn create(graph_dir: &Path, num_dim: usize) -> Result<(Self, GraphHeader), String> {
        std::fs::create_dir_all(graph_dir).map_err(|e| e.to_string())?;
        let path = graph_dir.join("nodes.bin");
        let header = GraphHeader::defaults(num_dim);
        header
            .write_json(&graph_dir.join("header.json"))
            .map_err(|e| e.to_string())?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)
            .map_err(|e| e.to_string())?;
        file.set_len(4096).map_err(|e| e.to_string())?;
        let mmap = unsafe { MmapMut::map_mut(&file).map_err(|e| e.to_string())? };
        Ok((
            Self {
                layout: NodeLayout::new(num_dim),
                file,
                mmap,
                num_nodes: 0,
            },
            header,
        ))
    }

    pub fn open(graph_dir: &Path) -> Result<(Self, GraphHeader), String> {
        let header = GraphHeader::read_json(&graph_dir.join("header.json"))?;
        let path = graph_dir.join("nodes.bin");
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&path)
            .map_err(|e| e.to_string())?;
        let len = file.metadata().map_err(|e| e.to_string())?.len();
        let layout = NodeLayout::new(header.num_dim);
        let num_nodes = if layout.record_stride > 0 {
            (len / layout.record_stride as u64) as u32
        } else {
            0
        };
        let mmap = unsafe { MmapMut::map_mut(&file).map_err(|e| e.to_string())? };
        Ok((
            Self {
                layout,
                file,
                mmap,
                num_nodes,
            },
            header,
        ))
    }

    pub fn set_num_nodes(&mut self, n: u32) {
        self.num_nodes = n;
    }

    /// Copy RAM-built tail records `[start, end)` into this mmap graph.
    pub fn merge_ram_tail(
        &mut self,
        ram: &RamGraph,
        start: usize,
        end: usize,
    ) -> Result<(), String> {
        if start >= end {
            return Ok(());
        }
        let need = end * self.layout.record_stride;
        self.grow_for_id((end - 1) as u32)?;
        if need > self.mmap.len() {
            return Err(format!("nodes.bin capacity {need} exceeds mmap {}", self.mmap.len()));
        }
        for i in start..end {
            let id = i as u32;
            let dst_start = i * self.layout.record_stride;
            let record = ram.read_record(id);
            self.mmap[dst_start..dst_start + self.layout.record_stride].copy_from_slice(record);
        }
        self.num_nodes = end as u32;
        Ok(())
    }

    fn grow_for_id(&mut self, id: u32) -> Result<(), String> {
        let need = (id as usize + 1) * self.layout.record_stride;
        if need <= self.mmap.len() {
            return Ok(());
        }
        let grow_chunk = self.layout.record_stride.saturating_mul(64);
        let new_len = need.max(self.mmap.len().saturating_add(grow_chunk));
        self.file
            .set_len(new_len as u64)
            .map_err(|e| e.to_string())?;
        self.mmap = unsafe { MmapMut::map_mut(&self.file).map_err(|e| e.to_string())? };
        Ok(())
    }

    fn record_range(&self, id: u32) -> std::ops::Range<usize> {
        let start = id as usize * self.layout.record_stride;
        start..start + self.layout.record_stride
    }
}

impl GraphAccess for MmapGraph {
    fn layout(&self) -> &NodeLayout {
        &self.layout
    }

    fn num_nodes(&self) -> u32 {
        self.num_nodes
    }

    fn node_level(&self, id: u32) -> u8 {
        let range = self.record_range(id);
        self.layout.read_level(&self.mmap[range])
    }

    fn vector(&self, id: u32) -> Vec<f32> {
        let range = self.record_range(id);
        self.layout.read_vector(&self.mmap[range])
    }

    fn neighbors(&self, id: u32, layer: u8) -> Vec<u32> {
        let range = self.record_range(id);
        self.layout.read_neighbors(&self.mmap[range], layer)
    }

    fn try_record(&self, id: u32) -> Option<&[u8]> {
        let range = self.record_range(id);
        Some(&self.mmap[range])
    }
}

impl GraphMut for MmapGraph {
    fn write_node(&mut self, id: u32, level: u8, vector: &[f32]) {
        self.grow_for_id(id).expect("grow nodes.bin");
        let stride = self.layout.record_stride;
        let start = id as usize * stride;
        let empty: Vec<Vec<u32>> = (0..LMAX).map(|_| Vec::new()).collect();
        self.layout
            .write_record(&mut self.mmap[start..start + stride], level, vector, &empty);
        if id + 1 > self.num_nodes {
            self.num_nodes = id + 1;
        }
    }

    fn set_neighbors(&mut self, id: u32, layer: u8, neighbors: &[u32]) {
        self.grow_for_id(id).expect("grow nodes.bin");
        let stride = self.layout.record_stride;
        let start = id as usize * stride;
        self.layout
            .write_neighbors_layer(&mut self.mmap[start..start + stride], layer, neighbors);
    }

    fn read_record_mut(&mut self, id: u32) -> &mut [u8] {
        self.grow_for_id(id).expect("grow nodes.bin");
        let stride = self.layout.record_stride;
        let start = id as usize * stride;
        &mut self.mmap[start..start + stride]
    }

    fn read_record(&self, id: u32) -> &[u8] {
        let range = self.record_range(id);
        &self.mmap[range]
    }

    fn fsync(&mut self) -> Result<(), String> {
        self.mmap.flush().map_err(|e| e.to_string())?;
        self.file.sync_all().map_err(|e| e.to_string())
    }
}

pub fn truncate_nodes(path: &Path, indexed_rows: usize, record_stride: usize) -> Result<(), String> {
    let file = OpenOptions::new()
        .write(true)
        .open(path)
        .map_err(|e| e.to_string())?;
    file.set_len((indexed_rows * record_stride) as u64)
        .map_err(|e| e.to_string())
}

#[cfg(test)]
mod store_tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn truncate_nodes_shrinks_file() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("nodes.bin");
        std::fs::write(&path, vec![0u8; 100]).unwrap();
        truncate_nodes(&path, 2, 10).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 20);
    }

    #[test]
    fn ram_graph_record_api() {
        let mut graph = RamGraph::new(4);
        graph.ensure_id(0);
        assert_eq!(graph.num_nodes(), 1);
        assert!(graph.try_record(0).is_some());
        let _ = graph.read_record_mut(0);
    }

    #[test]
    fn mmap_graph_set_num_nodes() {
        let dir = TempDir::new().expect("tempdir");
        let (mut mmap, _) = MmapGraph::create(&dir.path().join("g"), 2).unwrap();
        mmap.set_num_nodes(1);
        assert_eq!(mmap.num_nodes(), 1);
    }
}
