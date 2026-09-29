//! `pages.bin` for the incremental tree, written, compared and read one page at a
//! time so that persisting or reopening a tree never holds it in memory.
//!
//! The format is the one [`crate::index::page`] defines: internal pages carry their
//! child centroids, and each leaf its row ids and centroid (its entry in the parent,
//! or the mean of its rows for a lone root leaf). Leaf coordinates are not stored;
//! a reopened tree reads them back from the observation file ([`Tree::fill_blocks`]).

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::path::Path;

use crate::error::BpannError;
use crate::index::build::IndexHeader;
use crate::index::page::Page;
use crate::index::persist_atomic::{persist_index_files_with, skip_edges_bytes};
use crate::index::tree::Tree;
use crate::index::tree_store::{Kind, SLOT_ENTRIES};

fn invalid(e: impl ToString) -> BpannError {
    BpannError::InvalidParameter(e.to_string())
}

/// A writer that checks the bytes written against a reader instead of storing them.
struct Compare<R: Read> {
    reader: R,
    equal: bool,
    buf: Vec<u8>,
}

impl<R: Read> Write for Compare<R> {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        self.buf.resize(data.len(), 0);
        if self.equal && (self.reader.read_exact(&mut self.buf).is_err() || self.buf != data) {
            self.equal = false;
        }
        Ok(data.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn read_u32(r: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

impl Tree {
    /// Page `id` as a persisted [`Page`].
    pub fn page(&self, id: u32) -> Page {
        let s = &self.store;
        let d = s.num_dim();
        if !s.is_leaf(id) {
            return Page::Internal {
                page_id: id,
                centroids: s.block(id).chunks_exact(d).map(<[f32]>::to_vec).collect(),
                child_page_ids: s.ids(id).to_vec(),
            };
        }
        let centroid = match s.parent(id) {
            Some(p) => {
                let slot = s.ids(p).iter().position(|&k| k == id).expect("child listed in its parent");
                s.block(p)[slot * d..(slot + 1) * d].to_vec()
            }
            None => self.page_centroid(id),
        };
        Page::Leaf {
            page_id: id,
            row_ids: s.ids(id).to_vec(),
            row_range: None,
            vectors: Vec::new(),
            stored_centroid: Some(centroid),
        }
    }

    /// Write `pages.bin` to `w`.
    pub fn write_pages(&self, w: &mut impl Write) -> std::io::Result<()> {
        w.write_all(&(self.num_pages() as u32).to_le_bytes())?;
        for id in 0..self.num_pages() as u32 {
            let bytes = self.page(id).serialize(self.header.num_dim);
            w.write_all(&(bytes.len() as u32).to_le_bytes())?;
            w.write_all(&bytes)?;
        }
        Ok(())
    }

    /// Every page serialized, concatenated.
    pub fn page_bytes(&self) -> Vec<u8> {
        (0..self.num_pages() as u32).flat_map(|id| self.page(id).serialize(self.header.num_dim)).collect()
    }

    /// Write `header.json`, `pages.bin` and an empty `skip_edges.bin` atomically.
    pub fn persist(&self) -> Result<(), BpannError> {
        persist_index_files_with(&self.index_dir, &self.header, |w| self.write_pages(w), &HashMap::new())
    }

    /// Whether the files on disk are exactly what [`Self::persist`] would write.
    pub fn on_disk_index_matches(&self) -> Result<bool, BpannError> {
        let file = File::open(self.index_dir.join("pages.bin")).map_err(invalid)?;
        let mut cmp = Compare {
            reader: BufReader::new(file),
            equal: true,
            buf: Vec::new(),
        };
        self.write_pages(&mut cmp).map_err(invalid)?;
        let at_end = cmp.reader.read(&mut [0u8; 1]).map_err(invalid)? == 0;
        let skip = fs::read(self.index_dir.join("skip_edges.bin")).map_err(invalid)?;
        Ok(cmp.equal && at_end && skip == skip_edges_bytes(&HashMap::new()))
    }

    /// Bytes of the persisted index files in `index_dir`.
    pub fn persisted_bytes(index_dir: &Path) -> usize {
        ["header.json", "pages.bin", "skip_edges.bin"]
            .iter()
            .filter_map(|name| index_dir.join(name).metadata().ok())
            .map(|m| m.len() as usize)
            .sum()
    }

    /// Read the tree persisted in `index_dir`. `None` if it is not an incremental
    /// row-id tree (e.g. one written by an older fragment layout); leaf blocks are
    /// empty until [`Self::fill_blocks`].
    pub fn open_persisted(index_dir: &Path) -> Result<Option<Self>, BpannError> {
        let text = fs::read_to_string(index_dir.join("header.json")).map_err(invalid)?;
        let header: IndexHeader = serde_json::from_str(&text).map_err(invalid)?;
        let file = File::open(index_dir.join("pages.bin")).map_err(invalid)?;
        let mut r = BufReader::new(file);
        let num_pages = read_u32(&mut r).map_err(invalid)? as usize;
        let mut tree = Self::empty(header.num_dim, index_dir.to_path_buf())?;
        for _ in 0..num_pages {
            tree.store.alloc(Kind::Leaf)?;
        }
        let mut data = Vec::new();
        for _ in 0..num_pages {
            data.resize(read_u32(&mut r).map_err(invalid)? as usize, 0);
            r.read_exact(&mut data).map_err(invalid)?;
            if !tree.load_page(Page::deserialize(&data).map_err(invalid)?) {
                return Ok(None);
            }
        }
        tree.header = header;
        tree.rows_cached = false;
        Ok(tree.recount().then_some(tree))
    }

    /// Put a persisted page in its slot, marked unvisited for [`Self::recount`].
    fn load_page(&mut self, page: Page) -> bool {
        let (id, kind, ids, block) = match page {
            Page::Leaf {
                page_id,
                row_ids,
                row_range: None,
                vectors,
                stored_centroid: Some(_),
            } if vectors.is_empty() => (page_id, Kind::Leaf, row_ids, Vec::new()),
            Page::Internal {
                page_id,
                centroids,
                child_page_ids,
            } if !child_page_ids.is_empty() => (page_id, Kind::Internal, child_page_ids, centroids.concat()),
            _ => return false,
        };
        let d = self.store.num_dim();
        let fits = ids.len() <= SLOT_ENTRIES && (kind == Kind::Leaf || block.len() == ids.len() * d);
        if (id as usize) >= self.num_pages() || !fits || self.store.count(id) == u32::MAX as usize {
            return false;
        }
        self.store.set_kind(id, kind);
        let block = if kind == Kind::Leaf { vec![0.0; ids.len() * d] } else { block };
        self.store.set_entries(id, &ids, &block);
        self.store.set_count(id, u32::MAX as usize);
        true
    }
}
