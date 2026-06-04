//! RAM and mmap-backed graph stores.

use std::fs::{File, OpenOptions};
use std::path::Path;

use memmap2::MmapMut;

use crate::disk_hnsw::access::GraphAccess;
use crate::disk_hnsw::graph_mut::GraphMut;
use crate::disk_hnsw::node_layout::NodeLayout;
use crate::disk_hnsw::params::{GRAPH_FORMAT_VERSION, LMAX, M, M0};

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
        let mut nbrs: Vec<Vec<u32>> = (0..LMAX)
            .map(|l| self.layout.read_neighbors(&self.records[id as usize], l as u8))
            .collect();
        nbrs[layer as usize] = neighbors.to_vec();
        let level = self.layout.read_level(&self.records[id as usize]);
        let vector = self.layout.read_vector(&self.records[id as usize]);
        self.layout.write_record(
            &mut self.records[id as usize],
            level,
            &vector,
            &nbrs,
        );
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

pub struct GraphHeader {
    pub format_version: u32,
    pub num_dim: usize,
    pub m: usize,
    pub m0: usize,
    pub lmax: usize,
    pub ef_construction: usize,
    pub entry_point: u32,
    pub max_level: u8,
}

impl GraphHeader {
    pub fn defaults(num_dim: usize) -> Self {
        Self {
            format_version: GRAPH_FORMAT_VERSION,
            num_dim,
            m: M,
            m0: M0,
            lmax: LMAX,
            ef_construction: crate::disk_hnsw::params::EF_CONSTRUCTION,
            entry_point: 0,
            max_level: 0,
        }
    }

    pub fn write_json(&self, path: &Path) -> Result<(), String> {
        let json = format!(
            "{{\"format_version\":{},\"num_dim\":{},\"M\":{},\"M0\":{},\"LMAX\":{},\"ef_construction\":{},\"entry_point\":{},\"max_level\":{}}}",
            self.format_version,
            self.num_dim,
            self.m,
            self.m0,
            self.lmax,
            self.ef_construction,
            self.entry_point,
            self.max_level
        );
        std::fs::write(path, json).map_err(|e| e.to_string())
    }

    pub fn read_json(path: &Path) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
        Ok(Self {
            format_version: parse_u32(&text, "format_version")?,
            num_dim: parse_usize(&text, "num_dim")?,
            m: parse_usize(&text, "M")?,
            m0: parse_usize(&text, "M0")?,
            lmax: parse_usize(&text, "LMAX")?,
            ef_construction: parse_usize(&text, "ef_construction")?,
            entry_point: parse_u32(&text, "entry_point")?,
            max_level: parse_u8(&text, "max_level")?,
        })
    }
}

fn parse_usize(text: &str, field: &str) -> Result<usize, String> {
    parse_json_number(text, field)
}

fn parse_u32(text: &str, field: &str) -> Result<u32, String> {
    parse_json_number(text, field)
}

fn parse_u8(text: &str, field: &str) -> Result<u8, String> {
    let v: usize = parse_json_number(text, field)?;
    u8::try_from(v).map_err(|_| format!("{field} out of range"))
}

pub(crate) fn parse_json_number<T: std::str::FromStr>(text: &str, field: &str) -> Result<T, String>
where
    T::Err: std::fmt::Display,
{
    let key = format!("\"{field}\":");
    let pos = text.find(&key).ok_or_else(|| format!("missing {field}"))? + key.len();
    let tail = text[pos..].trim_start();
    let end = tail
        .find(|c: char| !c.is_ascii_digit())
        .unwrap_or(tail.len());
    tail[..end]
        .parse::<T>()
        .map_err(|e| format!("parse {field}: {e}"))
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

    fn grow_for_id(&mut self, id: u32) -> Result<(), String> {
        let need = (id as usize + 1) * self.layout.record_stride;
        if need <= self.mmap.len() {
            return Ok(());
        }
        let new_len = need.max(self.mmap.len().saturating_add(self.layout.record_stride * 64));
        if !self.mmap.is_empty() {
            self.mmap.flush().map_err(|e| e.to_string())?;
        }
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
        let mut nbrs: Vec<Vec<u32>> = (0..LMAX)
            .map(|l| {
                self.layout
                    .read_neighbors(&self.mmap[start..start + stride], l as u8)
            })
            .collect();
        nbrs[layer as usize] = neighbors.to_vec();
        let level = self.layout.read_level(&self.mmap[start..start + stride]);
        let vector = self.layout.read_vector(&self.mmap[start..start + stride]);
        self.layout.write_record(
            &mut self.mmap[start..start + stride],
            level,
            &vector,
            &nbrs,
        );
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
    fn graph_header_json_roundtrip() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("header.json");
        let hdr = GraphHeader::defaults(8);
        hdr.write_json(&path).unwrap();
        let loaded = GraphHeader::read_json(&path).unwrap();
        assert_eq!(loaded.num_dim, 8);
        assert_eq!(loaded.m, M);
    }

    #[test]
    fn truncate_nodes_shrinks_file() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("nodes.bin");
        std::fs::write(&path, vec![0u8; 100]).unwrap();
        truncate_nodes(&path, 2, 10).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 20);
    }

    #[test]
    fn graph_header_parse_all_fields() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("header.json");
        let hdr = GraphHeader {
            format_version: 1,
            num_dim: 4,
            m: M,
            m0: M0,
            lmax: LMAX,
            ef_construction: 200,
            entry_point: 3,
            max_level: 2,
        };
        hdr.write_json(&path).unwrap();
        let loaded = GraphHeader::read_json(&path).unwrap();
        assert_eq!(loaded.entry_point, 3);
        assert_eq!(loaded.max_level, 2);
    }

    #[test]
    fn parse_json_number_branches() {
        let text = "{\"format_version\":1,\"num_dim\":4,\"M\":16,\"M0\":32,\"LMAX\":16,\"ef_construction\":200,\"entry_point\":3,\"max_level\":2}";
        assert_eq!(parse_json_number::<usize>(text, "num_dim").unwrap(), 4);
        assert!(parse_json_number::<usize>(text, "missing").is_err());
        assert!(parse_json_number::<usize>("{\"num_dim\":}", "num_dim").is_err());
    }

    #[test]
    fn graph_header_rejects_bad_max_level() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("header.json");
        std::fs::write(&path, "{\"format_version\":1,\"num_dim\":2,\"M\":16,\"M0\":32,\"LMAX\":16,\"ef_construction\":200,\"entry_point\":0,\"max_level\":999}").unwrap();
        assert!(GraphHeader::read_json(&path).is_err());
    }

    #[test]
    fn graph_header_read_rejects_missing_field() {
        let dir = TempDir::new().expect("tempdir");
        let path = dir.path().join("bad.json");
        std::fs::write(&path, "{}").unwrap();
        assert!(GraphHeader::read_json(&path).is_err());
    }
}
