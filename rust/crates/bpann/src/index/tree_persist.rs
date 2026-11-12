use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, Read, Write};
use std::path::Path;

use crate::error::BpannError;
use crate::index::build::IndexHeader;
use crate::index::page::Page;
use crate::index::persist_atomic::{persist_index_files_with, skip_edges_bytes};
use crate::index::tree::Tree;
use crate::index::tree_store::{Kind, TreeStore, SLOT_ENTRIES, STAMP_FILE};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, PartialEq)]
struct Stamp {
    num_pages: usize,
    root_page_id: u32,
    indexed_rows: usize,
    meta_bytes: usize,
    slots_bytes: usize,
    relative_error: f64,
    absolute_error: f64,
    scale_x: bool,
    x_scale: Vec<f64>,
    #[serde(default)]
    routes: bool,
}

fn write_stamp(path: &Path, stamp: &Stamp) -> Result<(), BpannError> {
    let tmp = path.with_extension("tmp");
    fs::write(&tmp, serde_json::to_vec(stamp).map_err(invalid)?).map_err(invalid)?;
    File::open(&tmp)
        .and_then(|f| f.sync_all())
        .map_err(invalid)?;
    fs::rename(&tmp, path).map_err(invalid)
}

fn invalid(e: impl ToString) -> BpannError {
    BpannError::InvalidParameter(e.to_string())
}

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
                let slot = s
                    .ids(p)
                    .iter()
                    .position(|&k| k == id)
                    .expect("child listed in its parent");
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

    pub fn write_pages(&self, w: &mut impl Write) -> std::io::Result<()> {
        w.write_all(&(self.num_pages() as u32).to_le_bytes())?;
        for id in 0..self.num_pages() as u32 {
            let bytes = self.page(id).serialize(self.header.num_dim);
            w.write_all(&(bytes.len() as u32).to_le_bytes())?;
            w.write_all(&bytes)?;
        }
        Ok(())
    }

    pub fn page_bytes(&self) -> Vec<u8> {
        (0..self.num_pages() as u32)
            .flat_map(|id| self.page(id).serialize(self.header.num_dim))
            .collect()
    }

    pub fn persist(&self) -> Result<(), BpannError> {
        persist_index_files_with(
            &self.index_dir,
            &self.header,
            |w| self.write_pages(w),
            &HashMap::new(),
        )
    }

    pub fn persist_stamped(&mut self, scale_x: bool, x_scale: &[f64]) -> Result<(), BpannError> {
        let path = self.index_dir.join(STAMP_FILE);
        let _ = fs::remove_file(&path);
        self.persist()?;
        if !self.rows_cached || !self.store.is_named_in(&self.index_dir) {
            return Ok(());
        }
        self.store.flush()?;
        let (meta_bytes, slots_bytes) = self.store.file_lens();
        let stamp = Stamp {
            num_pages: self.num_pages(),
            root_page_id: self.header.root_page_id,
            indexed_rows: self.header.indexed_rows,
            meta_bytes,
            slots_bytes,
            relative_error: self.coordinate_relative_error,
            absolute_error: self.coordinate_absolute_error,
            scale_x,
            x_scale: x_scale.to_vec(),
            routes: true,
        };
        write_stamp(&path, &stamp)?;
        self.store.mark_stamped();
        Ok(())
    }

    pub fn open_stamped(
        index_dir: &Path,
        scale_x: bool,
        x_scale: &[f64],
    ) -> Result<Option<Self>, BpannError> {
        let Ok(text) = fs::read_to_string(index_dir.join(STAMP_FILE)) else {
            return Ok(None);
        };
        let Ok(mut stamp) = serde_json::from_str::<Stamp>(&text) else {
            return Ok(None);
        };
        let header_text = fs::read_to_string(index_dir.join("header.json")).map_err(invalid)?;
        let header: IndexHeader = serde_json::from_str(&header_text).map_err(invalid)?;
        let scale_ok = stamp.scale_x == scale_x && (!scale_x || stamp.x_scale == x_scale);
        if !scale_ok
            || stamp.root_page_id != header.root_page_id
            || stamp.indexed_rows != header.indexed_rows
        {
            return Ok(None);
        }
        let Some(mut store) = TreeStore::open_named(index_dir, header.num_dim, stamp.num_pages)
        else {
            return Ok(None);
        };
        if store.file_lens() != (stamp.meta_bytes, stamp.slots_bytes) {
            return Ok(None);
        }
        if !stamp.routes {
            store.rebuild_routes();
            store.flush()?;
            stamp.routes = true;
            write_stamp(&index_dir.join(STAMP_FILE), &stamp)?;
            store.mark_stamped();
            store.release_resident();
        }
        Ok(Some(Self {
            header,
            store,
            index_dir: index_dir.to_path_buf(),
            rows_cached: true,
            coordinate_relative_error: stamp.relative_error,
            coordinate_absolute_error: stamp.absolute_error,
        }))
    }

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

    pub fn persisted_bytes(index_dir: &Path) -> usize {
        ["header.json", "pages.bin", "skip_edges.bin"]
            .iter()
            .filter_map(|name| index_dir.join(name).metadata().ok())
            .map(|m| m.len() as usize)
            .sum()
    }

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
            } if !child_page_ids.is_empty() => {
                (page_id, Kind::Internal, child_page_ids, centroids.concat())
            }
            _ => return false,
        };
        let d = self.store.num_dim();
        let fits =
            ids.len() <= SLOT_ENTRIES && (kind == Kind::Leaf || block.len() == ids.len() * d);
        if (id as usize) >= self.num_pages() || !fits || self.store.count(id) == u32::MAX as usize {
            return false;
        }
        self.store.set_kind(id, kind);
        if kind == Kind::Leaf {
            self.store.set_ids_only(id, &ids);
        } else {
            self.store.set_entries(id, &ids, &block);
        }
        self.store.set_count(id, u32::MAX as usize);
        true
    }
}

#[cfg(test)]
#[path = "tree_persist_tests.rs"]
mod tests;
