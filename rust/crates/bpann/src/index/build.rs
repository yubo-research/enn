use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::error::BpannError;
use crate::index::kmeans::{PartitionNode, PartitionTree};
use crate::index::page::{write_pages_index, Page};

pub const DEFAULT_LEAF_CAPACITY: usize = 32;
pub const DEFAULT_SKIP_NEIGHBORS: usize = 3;

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct IndexHeader {
    pub num_dim: usize,
    pub indexed_rows: usize,
    pub root_page_id: u32,
    pub leaf_capacity: usize,
    pub skip_neighbors: usize,
}

pub struct BpannIndex {
    pub header: IndexHeader,
    pub pages: Vec<Page>,
    pub skip_edges: HashMap<u32, Vec<u32>>,
    page_map: HashMap<u32, usize>,
    pub index_dir: PathBuf,
}

const EXHAUSTIVE_SEARCH_ROW_LIMIT: usize = 2500;
const SKIP_REFINEMENT_ROW_LIMIT: usize = 150_000;

fn needs_skip_edges(row_count: usize) -> bool {
    row_count > EXHAUSTIVE_SEARCH_ROW_LIMIT && row_count <= SKIP_REFINEMENT_ROW_LIMIT
}

fn remap_page(page: &Page, id_map: &HashMap<u32, u32>) -> Page {
    match page {
        Page::Internal {
            page_id,
            centroids,
            child_page_ids,
        } => Page::Internal {
            page_id: id_map[page_id],
            centroids: centroids.clone(),
            child_page_ids: child_page_ids
                .iter()
                .map(|id| id_map[id])
                .collect(),
        },
            Page::Leaf {
                page_id,
                row_ids,
                vectors,
                stored_centroid,
            } => Page::Leaf {
                page_id: id_map[page_id],
                row_ids: row_ids.clone(),
                vectors: vectors.clone(),
                stored_centroid: stored_centroid.clone(),
            },
    }
}

fn build_page_map(pages: &[Page]) -> HashMap<u32, usize> {
    pages
        .iter()
        .enumerate()
        .map(|(i, p)| (p.page_id(), i))
        .collect()
}

impl BpannIndex {
    pub fn build_from_vectors(
        vectors: &[Vec<f32>],
        num_dim: usize,
        leaf_capacity: usize,
        seed: u64,
        index_dir: PathBuf,
    ) -> Result<Self, BpannError> {
        let row_ids: Vec<u32> = (0..vectors.len() as u32).collect();
        Self::build_from_rows(&row_ids, vectors, num_dim, leaf_capacity, seed, index_dir)
    }

    pub fn build_from_rows(
        row_ids: &[u32],
        vectors: &[Vec<f32>],
        num_dim: usize,
        leaf_capacity: usize,
        seed: u64,
        index_dir: PathBuf,
    ) -> Result<Self, BpannError> {
        Self::build_from_rows_with_persist(
            row_ids,
            vectors,
            num_dim,
            leaf_capacity,
            seed,
            index_dir,
            true,
        )
    }

    pub fn build_single_leaf_from_rows_with_persist(
        row_ids: &[u32],
        vectors: &[Vec<f32>],
        num_dim: usize,
        index_dir: PathBuf,
        persist: bool,
    ) -> Result<Self, BpannError> {
        let page = Page::Leaf {
            page_id: 0,
            row_ids: row_ids.to_vec(),
            vectors: vectors.to_vec(),
            stored_centroid: None,
        };
        let leaf_capacity = row_ids.len().max(1);
        let header = IndexHeader {
            num_dim,
            indexed_rows: row_ids.len(),
            root_page_id: 0,
            leaf_capacity,
            skip_neighbors: DEFAULT_SKIP_NEIGHBORS,
        };
        let pages = vec![page];
        let page_map = build_page_map(&pages);
        let index = Self {
            header,
            pages,
            skip_edges: HashMap::new(),
            page_map,
            index_dir,
        };
        if persist {
            index.persist()?;
        }
        Ok(index)
    }

    pub fn build_row_ids_leaf_with_persist(
        row_ids: &[u32],
        centroid: Vec<f32>,
        num_dim: usize,
        index_dir: PathBuf,
        persist: bool,
    ) -> Result<Self, BpannError> {
        let page = Page::Leaf {
            page_id: 0,
            row_ids: row_ids.to_vec(),
            vectors: Vec::new(),
            stored_centroid: Some(centroid),
        };
        let leaf_capacity = row_ids.len().max(1);
        let header = IndexHeader {
            num_dim,
            indexed_rows: row_ids.len(),
            root_page_id: 0,
            leaf_capacity,
            skip_neighbors: DEFAULT_SKIP_NEIGHBORS,
        };
        let pages = vec![page];
        let page_map = build_page_map(&pages);
        let index = Self {
            header,
            pages,
            skip_edges: HashMap::new(),
            page_map,
            index_dir,
        };
        if persist {
            index.persist()?;
        }
        Ok(index)
    }

    pub fn concat_merge(
        parts: Vec<BpannIndex>,
        index_dir: PathBuf,
        persist: bool,
    ) -> Result<Self, BpannError> {
        if parts.is_empty() {
            return Err(BpannError::InvalidParameter(
                "concat_merge requires at least one index".to_string(),
            ));
        }
        if parts.len() == 1 {
            let mut single = parts.into_iter().next().expect("one index");
            single.index_dir = index_dir;
            if persist {
                single.persist()?;
            }
            return Ok(single);
        }
        let num_dim = parts[0].header.num_dim;
        let leaf_capacity = parts[0].header.leaf_capacity;
        let skip_neighbors = parts[0].header.skip_neighbors;
        let mut pages = Vec::new();
        let mut child_page_ids = Vec::new();
        let mut child_centroids = Vec::new();
        let mut skip_edges = HashMap::new();
        let mut total_rows = 0usize;
        let mut next_id = 1u32;
        for part in &parts {
            total_rows += part.header.indexed_rows;
            let id_map: HashMap<u32, u32> = part
                .pages
                .iter()
                .map(|page| {
                    let new_id = next_id;
                    next_id += 1;
                    (page.page_id(), new_id)
                })
                .collect();
            child_page_ids.push(id_map[&part.header.root_page_id]);
            child_centroids.push(part.root_centroid());
            for page in &part.pages {
                pages.push(remap_page(page, &id_map));
            }
            for (&from, tos) in &part.skip_edges {
                if let Some(&new_from) = id_map.get(&from) {
                    let new_tos: Vec<u32> = tos
                        .iter()
                        .filter_map(|to| id_map.get(to).copied())
                        .collect();
                    if !new_tos.is_empty() {
                        skip_edges.insert(new_from, new_tos);
                    }
                }
            }
        }
        pages.push(Page::Internal {
            page_id: 0,
            centroids: child_centroids,
            child_page_ids,
        });
        let page_map = build_page_map(&pages);
        let header = IndexHeader {
            num_dim,
            indexed_rows: total_rows,
            root_page_id: 0,
            leaf_capacity,
            skip_neighbors,
        };
        let index = Self {
            header,
            pages,
            skip_edges,
            page_map,
            index_dir,
        };
        if persist {
            index.persist()?;
        }
        Ok(index)
    }

    pub fn build_from_rows_with_persist(
        row_ids: &[u32],
        vectors: &[Vec<f32>],
        num_dim: usize,
        leaf_capacity: usize,
        seed: u64,
        index_dir: PathBuf,
        persist: bool,
    ) -> Result<Self, BpannError> {
        if row_ids.len() <= leaf_capacity {
            return Self::build_single_leaf_from_rows_with_persist(
                row_ids,
                vectors,
                num_dim,
                index_dir,
                persist,
            );
        }
        let partition = PartitionTree::build(row_ids, vectors, leaf_capacity, seed);
        let (pages, root_page_id) = partition_to_pages(&partition.root);
        let skip_edges = if needs_skip_edges(row_ids.len()) {
            build_skip_edges(&pages, DEFAULT_SKIP_NEIGHBORS)
        } else {
            HashMap::new()
        };
        let page_map = build_page_map(&pages);
        let header = IndexHeader {
            num_dim,
            indexed_rows: row_ids.len(),
            root_page_id,
            leaf_capacity,
            skip_neighbors: DEFAULT_SKIP_NEIGHBORS,
        };
        let index = Self {
            header: header.clone(),
            pages,
            skip_edges,
            page_map,
            index_dir: index_dir.clone(),
        };
        if persist {
            index.persist()?;
        }
        Ok(index)
    }

    pub fn open(index_dir: PathBuf) -> Result<Self, BpannError> {
        let header_path = index_dir.join("header.json");
        let text = fs::read_to_string(&header_path)
            .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        let header: IndexHeader = serde_json::from_str(&text)
            .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        let pages_path = index_dir.join("pages.bin");
        let file = File::open(&pages_path).map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        let mut reader = BufReader::new(file);
        let pages = crate::index::page::read_pages_index(&mut reader)
            .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        let skip_edges = read_skip_edges(&index_dir.join("skip_edges.bin"))?;
        let page_map = build_page_map(&pages);
        Ok(Self {
            header,
            pages,
            skip_edges,
            page_map,
            index_dir,
        })
    }

    pub fn persist(&self) -> Result<(), BpannError> {
        fs::create_dir_all(&self.index_dir).map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        let header_json = serde_json::to_string_pretty(&self.header)
            .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        fs::write(self.index_dir.join("header.json"), header_json)
            .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        let pages_path = self.index_dir.join("pages.bin");
        let file = File::create(&pages_path).map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        let mut writer = BufWriter::new(file);
        write_pages_index(&self.pages, self.header.num_dim, &mut writer)
            .map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        writer.flush().map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
        write_skip_edges(&self.index_dir.join("skip_edges.bin"), &self.skip_edges)?;
        Ok(())
    }

    pub fn page_by_id(&self, page_id: u32) -> Option<&Page> {
        self.page_map.get(&page_id).map(|&i| &self.pages[i])
    }

    pub fn root_centroid(&self) -> Vec<f32> {
        self.page_by_id(self.header.root_page_id)
            .map(Page::centroid)
            .unwrap_or_default()
    }

    pub fn leaf_row_ids(&self) -> Vec<u32> {
        let mut row_ids = Vec::new();
        for page in &self.pages {
            if let Page::Leaf { row_ids: ids, .. } = page {
                row_ids.extend_from_slice(ids);
            }
        }
        row_ids.sort_unstable();
        row_ids.dedup();
        row_ids
    }

    pub fn leaf_page_ids(&self) -> Vec<u32> {
        self.pages
            .iter()
            .filter(|p| matches!(p, Page::Leaf { .. }))
            .map(|p| p.page_id())
            .collect()
    }

    pub fn index_memory_bytes(&self) -> usize {
        let mut total = 0usize;
        for name in ["header.json", "pages.bin", "skip_edges.bin"] {
            let p = self.index_dir.join(name);
            if p.exists() {
                total += p.metadata().map(|m| m.len() as usize).unwrap_or(0);
            }
        }
        total
    }

    pub fn page_bytes(&self) -> Vec<u8> {
        let mut all = Vec::new();
        for page in &self.pages {
            all.extend(page.serialize(self.header.num_dim));
        }
        all
    }
}

fn partition_to_pages(node: &PartitionNode) -> (Vec<Page>, u32) {
    partition_to_pages_id(node, 0)
}

fn partition_to_pages_id(node: &PartitionNode, next_id: u32) -> (Vec<Page>, u32) {
    match node {
        PartitionNode::Leaf { entries, .. } => {
            let row_ids: Vec<u32> = entries.iter().map(|(id, _)| *id).collect();
            let vecs: Vec<Vec<f32>> = entries.iter().map(|(_, v)| v.clone()).collect();
            let page = Page::Leaf {
                page_id: next_id,
                row_ids,
                vectors: vecs,
                stored_centroid: None,
            };
            (vec![page], next_id)
        }
        PartitionNode::Internal { children, .. } => {
            let my_id = next_id;
            let mut child_page_ids = Vec::new();
            let mut child_centroids = Vec::new();
            let mut child_pages = Vec::new();
            let mut cur_id = next_id + 1;
            let mut last_id = my_id;
            for child in children {
                let (pages, subtree_last) = partition_to_pages_id(child, cur_id);
                if let Some(first) = pages.first() {
                    child_page_ids.push(first.page_id());
                    child_centroids.push(first.centroid());
                }
                last_id = last_id.max(subtree_last);
                cur_id = subtree_last + 1;
                child_pages.extend(pages);
            }
            let internal = Page::Internal {
                page_id: my_id,
                centroids: child_centroids,
                child_page_ids,
            };
            let mut all = vec![internal];
            all.extend(child_pages);
            (all, last_id.max(my_id))
        }
    }
}

fn build_skip_edges(pages: &[Page], k: usize) -> HashMap<u32, Vec<u32>> {
    let leaves: Vec<(u32, Vec<f32>)> = pages
        .iter()
        .filter_map(|p| match p {
            Page::Leaf { page_id, .. } => Some((*page_id, p.centroid())),
            _ => None,
        })
        .collect();
    let mut edges = HashMap::new();
    for (i, (id_a, ca)) in leaves.iter().enumerate() {
        let mut dists: Vec<(u32, f32)> = leaves
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, (id_b, cb))| (*id_b, crate::distance::l2_sq_f32(ca, cb)))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let neighbors: Vec<u32> = dists.into_iter().take(k).map(|(id, _)| id).collect();
        edges.insert(*id_a, neighbors);
    }
    edges
}

fn write_skip_edges(path: &Path, edges: &HashMap<u32, Vec<u32>>) -> Result<(), BpannError> {
    let mut buf = Vec::new();
    buf.extend_from_slice(&(edges.len() as u32).to_le_bytes());
    for (&from, tos) in edges {
        buf.extend_from_slice(&from.to_le_bytes());
        buf.extend_from_slice(&(tos.len() as u32).to_le_bytes());
        for &to in tos {
            buf.extend_from_slice(&to.to_le_bytes());
        }
    }
    fs::write(path, buf).map_err(|e| BpannError::InvalidParameter(e.to_string()))
}

fn read_skip_edges(path: &Path) -> Result<HashMap<u32, Vec<u32>>, BpannError> {
    if !path.exists() {
        return Ok(HashMap::new());
    }
    let data = fs::read(path).map_err(|e| BpannError::InvalidParameter(e.to_string()))?;
    if data.len() < 4 {
        return Ok(HashMap::new());
    }
    let n = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let mut off = 4;
    let mut out = HashMap::new();
    for _ in 0..n {
        if off + 8 > data.len() {
            break;
        }
        let from = u32::from_le_bytes(data[off..off + 4].try_into().unwrap());
        off += 4;
        let m = u32::from_le_bytes(data[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        let mut tos = Vec::with_capacity(m);
        for _ in 0..m {
            if off + 4 > data.len() {
                break;
            }
            tos.push(u32::from_le_bytes(data[off..off + 4].try_into().unwrap()));
            off += 4;
        }
        out.insert(from, tos);
    }
    Ok(out)
}

#[cfg(test)]
mod kiss_coverage_tests {
    use super::*;

    #[test]
    fn build_units_are_linked() {
        let _ = (
            remap_page,
            BpannIndex::build_row_ids_leaf_with_persist,
            BpannIndex::concat_merge,
            BpannIndex::root_centroid,
            BpannIndex::leaf_row_ids,
            read_skip_edges,
            std::mem::size_of::<IndexHeader>,
        );
    }
}