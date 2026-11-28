use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::time::Instant;

use super::graph_box::{bbox, empty_box, BoxFiles};
use super::graph_io::{GFiles, Geo, Items};
use super::graph_up::Updates;
use super::graph_split::{nearest, split_to_cap};
use super::lsh::io_error;
use crate::backend::MmapColumnStore;
use crate::error::ENNError;

pub(crate) const GRAPH_META: &str = "o1g_meta.json";

pub(crate) fn selected(work_dir: &Path) -> bool {
    work_dir.join(GRAPH_META).exists()
        || std::env::var("ENN_O1_KIND").is_ok_and(|v| v.eq_ignore_ascii_case("graph"))
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse::<f64>().ok()).map_or(default, |v| v as usize)
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct GraphStats {
    pub batches: u64,
    pub leaf_splits: u64,
    pub up_splits: u64,
    pub link_writes: u64,
    pub route_ns: u64,
    pub apply_ns: u64,
    pub link_ns: u64,
    pub up_ns: u64,
    pub max_batch_ns: u64,
    pub leaf_reads: u64,
    pub leaf_writes: u64,
    pub side_reads: u64,
    pub side_writes: u64,
    pub rehashes: u64,
    #[serde(default)]
    pub buf_flushes: u64,
    #[serde(default)]
    pub buf_flushed_rows: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GraphMeta {
    pub d: usize,
    pub cap: usize,
    pub fan: usize,
    pub m: usize,
    pub n: u64,
    pub height: usize,
    pub root: u32,
    pub counts: Vec<u32>,
    pub scale: Vec<f64>,
    pub stats: GraphStats,
}

pub(crate) struct GraphIndex {
    pub(crate) dir: PathBuf,
    pub(crate) meta: GraphMeta,
    pub(crate) files: GFiles,
    pub(crate) max_rows: usize,
    pub(crate) buf: BTreeMap<u32, Items>,
    pub(crate) buf_rows: usize,
    pub(crate) buf_cap: usize,
    pub(crate) width: usize,
    pub(crate) boxes: BoxFiles,
    pub(crate) bound: bool,
}

pub(crate) struct Pieces {
    pub ids: Vec<u32>,
    pub cents: Vec<Vec<f64>>,
    pub boxes: Vec<Vec<f64>>,
}

impl GraphIndex {
    pub(crate) fn open(work_dir: &Path, fallback_scale: &[f64], rows: &MmapColumnStore) -> Result<Self, ENNError> {
        let d = rows.ncols;
        let stored: Option<GraphMeta> = ennbo_bpann::bpann_read_json(&work_dir.join(GRAPH_META)).map_err(io_error)?;
        let fresh = stored.is_none();
        let meta = stored.unwrap_or_else(|| GraphMeta {
            d,
            cap: env_usize("ENN_G_ROWS", 128).max(4),
            fan: env_usize("ENN_G_FAN", 64).max(4),
            m: env_usize("ENN_G_M", 16).max(1),
            n: 0,
            height: 2,
            root: 0,
            counts: vec![1, 1],
            scale: fallback_scale.to_vec(),
            stats: GraphStats::default(),
        });
        let geo = Geo { d, cap: meta.cap, fan: meta.fan, m: meta.m };
        let files = GFiles::open(work_dir, geo, meta.height)?;
        let boxes = BoxFiles::open(work_dir, d, meta.height)?;
        let buf_cap = env_usize("ENN_G_BUF", 4 << 20) / (4 + 8 * d);
        let mut index = Self { dir: work_dir.to_path_buf(), meta, files, max_rows: env_usize("ENN_G_BUDGET", 3200), buf: BTreeMap::new(), buf_rows: 0, buf_cap, width: env_usize("ENN_G_WIDTH", 1).max(1), boxes, bound: env_usize("ENN_G_BOUND", 1) > 0 };
        if fresh {
            let mut root = Items::default();
            root.push(0, &vec![0.0; d]);
            index.files.write_up(1, 0, &root)?;
            index.files.write_leaf(0, &Items::default())?;
            index.files.write_par(0, 0, 0)?;
            index.boxes.write(0, 0, &empty_box(d))?;
            index.boxes.write(1, 0, &empty_box(d))?;
        }
        index.save()?;
        index.reload_buffer(rows)?;
        index.insert_rows(rows)?;
        Ok(index)
    }

    pub(crate) fn save(&mut self) -> Result<(), ENNError> {
        let (c, st) = (&self.files.ctr, &mut self.meta.stats);
        let get = |a: &std::sync::atomic::AtomicU64| a.load(std::sync::atomic::Ordering::Relaxed);
        st.leaf_reads = get(&c.leaf_reads);
        st.leaf_writes = get(&c.leaf_writes);
        st.side_reads = get(&c.side_reads);
        st.side_writes = get(&c.side_writes);
        ennbo_bpann::bpann_write_json(&self.dir.join(GRAPH_META), &self.meta).map_err(io_error)
    }

    pub(crate) fn hash_scale(&self) -> &[f64] {
        &self.meta.scale
    }

    pub(crate) fn weights(&self) -> Vec<f64> {
        self.meta.scale.iter().map(|s| 1.0 / (s * s)).collect()
    }

    pub(crate) fn rehash(&mut self, scale: &[f64]) -> Result<(), ENNError> {
        if scale != self.meta.scale.as_slice() {
            self.meta.scale = scale.to_vec();
            self.meta.stats.rehashes += 1;
            self.save()?;
        }
        Ok(())
    }

    pub(crate) fn insert_rows(&mut self, rows: &MmapColumnStore) -> Result<(), ENNError> {
        while (self.meta.n as usize) < rows.nrows {
            let lo = self.meta.n as usize;
            let hi = rows.nrows.min(lo + 4096);
            let x = rows.mmap_row_range(lo, hi)?;
            let x: Vec<f64> = x.iter().copied().collect();
            self.insert_batch(lo as u32, &x)?;
            self.meta.n = hi as u64;
        }
        self.save()
    }

    fn insert_batch(&mut self, lo: u32, x: &[f64]) -> Result<(), ENNError> {
        let (t0, d, w) = (Instant::now(), self.meta.d, self.weights());
        let mut memo: HashMap<(usize, u32), Items> = HashMap::new();
        let mut groups: BTreeMap<u32, Items> = BTreeMap::new();
        for (i, r) in x.chunks_exact(d).enumerate() {
            groups.entry(self.route(r, &w, &mut memo)?).or_default().push(lo + i as u32, r);
        }
        self.meta.stats.route_ns += t0.elapsed().as_nanos() as u64;
        let due = self.buffer(groups)?;
        self.apply(due, &w)?;
        let st = &mut self.meta.stats;
        st.batches += 1;
        st.max_batch_ns = st.max_batch_ns.max(t0.elapsed().as_nanos() as u64);
        Ok(())
    }

    pub(crate) fn apply(&mut self, groups: BTreeMap<u32, Items>, w: &[f64]) -> Result<(), ENNError> {
        let t1 = Instant::now();
        let mut touched = Updates::new();
        for (leaf, rows) in groups {
            let mut it = self.files.read_leaf(leaf)?;
            it.ids.extend_from_slice(&rows.ids);
            it.x.extend_from_slice(&rows.x);
            let parent = self.files.read_par(0, leaf)?;
            let p = self.store_leaf(leaf, it, parent, w)?;
            let ups = touched.entry(parent).or_default();
            for ((&id, c), b) in p.ids.iter().zip(p.cents).zip(p.boxes) {
                ups.insert(id, (c, b));
            }
        }
        let t2 = Instant::now();
        self.propagate(touched, w)?;
        self.meta.stats.apply_ns += (t2 - t1).as_nanos() as u64;
        self.meta.stats.up_ns += t2.elapsed().as_nanos() as u64;
        Ok(())
    }

    pub(crate) fn route(&self, r: &[f64], w: &[f64], memo: &mut HashMap<(usize, u32), Items>) -> Result<u32, ENNError> {
        let mut node = self.meta.root;
        for level in (1..self.meta.height).rev() {
            if let std::collections::hash_map::Entry::Vacant(e) = memo.entry((level, node)) {
                e.insert(self.files.read_up(level, node)?);
            }
            let rec = &memo[&(level, node)];
            node = rec.ids[nearest(r, rec, self.meta.d, w)];
        }
        Ok(node)
    }

    fn store_leaf(&mut self, leaf: u32, it: Items, parent: u32, w: &[f64]) -> Result<Pieces, ENNError> {
        let d = self.meta.d;
        let parts = split_to_cap(it, d, w, self.meta.cap);
        let mut p = Pieces { ids: Vec::new(), cents: Vec::new(), boxes: Vec::new() };
        for (k, part) in parts.iter().enumerate() {
            let id = match k {
                0 => leaf,
                _ => self.new_id(0),
            };
            self.files.write_leaf(id, part)?;
            let c = part.mean(d);
            self.files.write_cent(id, &c)?;
            self.files.write_par(0, id, parent)?;
            let b = bbox(part, d);
            self.boxes.write(0, id, &b)?;
            p.ids.push(id);
            p.cents.push(c);
            p.boxes.push(b);
        }
        if p.ids.len() > 1 {
            self.meta.stats.leaf_splits += p.ids.len() as u64 - 1;
            let t = Instant::now();
            self.rewire(&p, parent, w)?;
            self.meta.stats.link_ns += t.elapsed().as_nanos() as u64;
        }
        Ok(p)
    }

    pub(crate) fn new_id(&mut self, level: usize) -> u32 {
        self.meta.counts[level] += 1;
        self.meta.counts[level] - 1
    }

    pub(crate) fn leaves(&self) -> usize {
        self.meta.counts[0] as usize
    }

    pub(crate) fn memory_bytes(&self) -> usize {
        std::mem::size_of::<Self>() + 8 * self.meta.scale.len() + 4 * self.meta.counts.len()
    }

    pub(crate) fn set_budget(&mut self, _max_leaves: usize, max_candidates: usize) {
        self.max_rows = max_candidates.max(1);
    }
}
