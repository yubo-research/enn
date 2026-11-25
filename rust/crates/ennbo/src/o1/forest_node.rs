use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::cache::{CacheKey, Cached, SharedCache};
use super::forest_rules::SplitRule;
use super::lsh::io_error;
use crate::error::ENNError;

pub(crate) const NIL: u32 = 0;
const HEADER_BYTES: usize = 24;
const TAG_HEADER: u32 = 0x4f31_4631;
const TAG_LEAF: u32 = 1;
const TAG_SPLIT: u32 = 2;
const TAG_FREE: u32 = 3;
const TAG_PIVOT: u32 = 4;
const TAG_DIR8: u32 = 5;

static NEXT_FILE_UID: AtomicU64 = AtomicU64::new(1);

fn next_uid() -> u64 {
    NEXT_FILE_UID.fetch_add(1, Ordering::Relaxed)
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Dir {
    pub scale: f32,
    pub code: Box<[i8]>,
}

impl Dir {
    pub(crate) fn quantize(dir: &[f64]) -> Self {
        let top = dir.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let scale = (top / 127.0) as f32;
        let inv = if scale > 0.0 { 1.0 / f64::from(scale) } else { 0.0 };
        Self { scale, code: dir.iter().map(|&v| (v * inv).round().clamp(-127.0, 127.0) as i8).collect() }
    }

    pub(crate) fn dot(&self, x: &[f64]) -> f64 {
        f64::from(self.scale) * self.code.iter().zip(x).map(|(&a, b)| f64::from(a) * b).sum::<f64>()
    }

    pub(crate) fn len(&self) -> usize {
        self.code.len()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Cut {
    Seed(u32),
    Pivot(u32, u32),
    Dir(Arc<Dir>),
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Node {
    Leaf { ids: Vec<u32>, next: u32 },
    Split { cut: Cut, thr: f64, left: u32, right: u32 },
    Free { next: u32 },
}

#[derive(Debug, Default)]
pub(crate) struct NodeCounters {
    pub reads: AtomicU64,
    pub writes: AtomicU64,
}

pub(crate) struct NodeFile {
    file: File,
    cap: usize,
    dim: usize,
    payload: usize,
    root: u32,
    free_head: u32,
    num_nodes: u32,
    uid: u64,
    cache: SharedCache,
    pub(crate) counters: NodeCounters,
}

fn put_u32(buf: &mut [u8], at: usize, v: u32) {
    buf[at..at + 4].copy_from_slice(&v.to_le_bytes());
}

fn get_u32(buf: &[u8], at: usize) -> u32 {
    u32::from_le_bytes(buf[at..at + 4].try_into().expect("u32 field"))
}

fn get_thr(buf: &[u8]) -> f64 {
    f64::from_le_bytes(buf[16..24].try_into().expect("thr field"))
}

pub(crate) fn record_bytes(cap: usize, payload: usize) -> usize {
    HEADER_BYTES + (4 * cap).max(payload)
}

fn encode(node: &Node, cap: usize, payload: usize) -> Vec<u8> {
    let mut buf = vec![0u8; record_bytes(cap, payload)];
    match node {
        Node::Leaf { ids, next } => {
            put_u32(&mut buf, 0, TAG_LEAF);
            put_u32(&mut buf, 4, ids.len() as u32);
            put_u32(&mut buf, 8, *next);
            for (i, id) in ids.iter().enumerate() {
                put_u32(&mut buf, HEADER_BYTES + 4 * i, *id);
            }
        }
        Node::Split { cut, thr, left, right } => {
            put_u32(&mut buf, 4, *left);
            put_u32(&mut buf, 8, *right);
            buf[16..24].copy_from_slice(&thr.to_le_bytes());
            match *cut {
                Cut::Seed(seed) => {
                    put_u32(&mut buf, 0, TAG_SPLIT);
                    put_u32(&mut buf, 12, seed);
                }
                Cut::Pivot(a, b) => {
                    put_u32(&mut buf, 0, TAG_PIVOT);
                    put_u32(&mut buf, 12, a);
                    put_u32(&mut buf, HEADER_BYTES, b);
                }
                Cut::Dir(ref dir) => encode_dir(&mut buf, dir),
            }
        }
        Node::Free { next } => {
            put_u32(&mut buf, 0, TAG_FREE);
            put_u32(&mut buf, 8, *next);
        }
    }
    buf
}

fn encode_dir(buf: &mut [u8], dir: &Dir) {
    buf[12..16].copy_from_slice(&dir.scale.to_le_bytes());
    put_u32(buf, 0, TAG_DIR8);
    let body = &mut buf[HEADER_BYTES..];
    dir.code.iter().zip(body.iter_mut()).for_each(|(&v, b)| *b = v as u8);
}

fn decode_dir(buf: &[u8], dim: usize) -> Option<Dir> {
    let scale = f32::from_le_bytes(buf[12..16].try_into().ok()?);
    let body = buf.get(HEADER_BYTES..HEADER_BYTES + dim)?;
    Some(Dir { scale, code: body.iter().map(|&b| b as i8).collect() })
}

fn decode(buf: &[u8], cap: usize, dim: usize) -> Result<Node, ENNError> {
    let (a, b) = (get_u32(buf, 4), get_u32(buf, 8));
    match get_u32(buf, 0) {
        TAG_LEAF if (a as usize) <= cap => Ok(Node::Leaf {
            ids: (0..a as usize).map(|i| get_u32(buf, HEADER_BYTES + 4 * i)).collect(),
            next: b,
        }),
        TAG_SPLIT => Ok(Node::Split { cut: Cut::Seed(get_u32(buf, 12)), thr: get_thr(buf), left: a, right: b }),
        TAG_PIVOT if cap > 0 => Ok(Node::Split {
            cut: Cut::Pivot(get_u32(buf, 12), get_u32(buf, HEADER_BYTES)),
            thr: get_thr(buf),
            left: a,
            right: b,
        }),
        TAG_DIR8 => match decode_dir(buf, dim) {
            Some(dir) if dim > 0 => Ok(Node::Split { cut: Cut::Dir(Arc::new(dir)), thr: get_thr(buf), left: a, right: b }),
            _ => Err(io_error("forest direction record does not fit its file")),
        },
        TAG_FREE => Ok(Node::Free { next: b }),
        tag => Err(io_error(format!("forest node is corrupt (tag {tag}, count {a})"))),
    }
}

impl NodeFile {
    pub(crate) fn create(path: &Path, cap: usize, dim: usize, payload: usize) -> Result<Self, ENNError> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .map_err(io_error)?;
        let mut nf = Self { file, cap, dim, payload, root: 1, free_head: NIL, num_nodes: 1, uid: next_uid(), cache: None, counters: NodeCounters::default() };
        let root = nf.alloc()?;
        nf.root = root;
        nf.write(root, &Node::Leaf { ids: Vec::new(), next: NIL })?;
        nf.write_header()?;
        Ok(nf)
    }

    pub(crate) fn open(path: &Path, cap: usize) -> Result<Self, ENNError> {
        let file = OpenOptions::new().read(true).write(true).open(path).map_err(io_error)?;
        let mut buf = vec![0u8; record_bytes(cap, 0)];
        file.read_exact_at(&mut buf, 0).map_err(io_error)?;
        if get_u32(&buf, 0) != TAG_HEADER || get_u32(&buf, 16) as usize != cap {
            return Err(io_error("forest node file header does not match"));
        }
        let (num_nodes, dim, payload) = (get_u32(&buf, 12), get_u32(&buf, 20) as usize, get_u32(&buf, HEADER_BYTES) as usize);
        let len = file.metadata().map_err(io_error)?.len();
        if len < (num_nodes as u64) * record_bytes(cap, payload) as u64 {
            return Err(io_error("forest node file is truncated"));
        }
        Ok(Self {
            file,
            cap,
            dim,
            payload,
            root: get_u32(&buf, 4),
            free_head: get_u32(&buf, 8),
            num_nodes,
            uid: next_uid(),
            cache: None,
            counters: NodeCounters::default(),
        })
    }

    fn write_header(&self) -> Result<(), ENNError> {
        let mut buf = vec![0u8; self.record_bytes()];
        put_u32(&mut buf, 0, TAG_HEADER);
        put_u32(&mut buf, 4, self.root);
        put_u32(&mut buf, 8, self.free_head);
        put_u32(&mut buf, 12, self.num_nodes);
        put_u32(&mut buf, 16, self.cap as u32);
        put_u32(&mut buf, 20, self.dim as u32);
        put_u32(&mut buf, HEADER_BYTES, self.payload as u32);
        self.file.write_all_at(&buf, 0).map_err(io_error)
    }

    pub(crate) fn root(&self) -> u32 {
        self.root
    }

    pub(crate) fn cap(&self) -> usize {
        self.cap
    }

    pub(crate) fn num_nodes(&self) -> u32 {
        self.num_nodes
    }

    pub(crate) fn record_bytes(&self) -> usize {
        record_bytes(self.cap, self.payload)
    }

    pub(crate) fn holds(&self, rule: SplitRule, dim: usize) -> bool {
        rule != SplitRule::PivotI8 || (dim == self.dim && rule.payload(dim) <= (4 * self.cap).max(self.payload))
    }

    pub(crate) fn fits(&self, dir: &Dir) -> bool {
        dir.len() == self.dim && dir.len() <= (4 * self.cap).max(self.payload)
    }

    fn offset(&self, id: u32) -> u64 {
        u64::from(id) * self.record_bytes() as u64
    }

    pub(crate) fn set_cache(&mut self, cache: SharedCache) {
        self.cache = cache;
    }

    fn key(&self, id: u32) -> CacheKey {
        CacheKey::Node { file: self.uid, id }
    }

    pub(crate) fn read(&self, id: u32) -> Result<Node, ENNError> {
        if id == NIL || id >= self.num_nodes {
            return Err(io_error(format!("forest node {id} is out of range")));
        }
        let Some(cache) = &self.cache else {
            return self.read_disk(id);
        };
        match cache.get_or_load(self.key(id), || self.read_disk(id).map(|n| Cached::Node(Arc::new(n))))? {
            Cached::Node(n) => Ok((*n).clone()),
            Cached::Row(_) => Err(io_error("forest node cache entry holds a row")),
        }
    }

    pub(crate) fn shared_cache(&self) -> SharedCache {
        self.cache.clone()
    }

    pub(crate) fn absorb_counters(&self, old: &NodeFile) {
        self.counters.reads.fetch_add(old.counters.reads.load(Ordering::Relaxed), Ordering::Relaxed);
        self.counters.writes.fetch_add(old.counters.writes.load(Ordering::Relaxed), Ordering::Relaxed);
    }

    pub(crate) fn dim_payload(&self) -> (usize, usize) {
        (self.dim, self.payload)
    }

    pub(crate) fn set_layout(&mut self, root: u32, num_nodes: u32) -> Result<(), ENNError> {
        (self.root, self.free_head, self.num_nodes) = (root, NIL, num_nodes);
        self.write_header()
    }

    pub(crate) fn read_disk(&self, id: u32) -> Result<Node, ENNError> {
        let mut buf = vec![0u8; self.record_bytes()];
        self.file.read_exact_at(&mut buf, self.offset(id)).map_err(io_error)?;
        self.counters.reads.fetch_add(1, Ordering::Relaxed);
        decode(&buf, self.cap, self.dim)
    }

    pub(crate) fn write(&self, id: u32, node: &Node) -> Result<(), ENNError> {
        match node {
            Node::Leaf { ids, .. } if ids.len() > self.cap => return Err(io_error("forest leaf exceeds its capacity")),
            Node::Split { cut: Cut::Dir(dir), .. } if !self.fits(dir) => return Err(io_error("forest direction does not fit its record")),
            _ => {}
        }
        let written = self.file.write_all_at(&encode(node, self.cap, self.payload), self.offset(id)).map_err(io_error);
        if let Some(cache) = &self.cache {
            match &written {
                Ok(()) => cache.put(self.key(id), Cached::Node(Arc::new(node.clone()))),
                Err(_) => cache.remove(&self.key(id)),
            }
        }
        written?;
        self.counters.writes.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    pub(crate) fn alloc(&mut self) -> Result<u32, ENNError> {
        let id = if self.free_head != NIL {
            let id = self.free_head;
            match self.read(id)? {
                Node::Free { next } => self.free_head = next,
                _ => return Err(io_error("forest free list is corrupt")),
            }
            id
        } else {
            let id = self.num_nodes;
            self.num_nodes = id.checked_add(1).ok_or_else(|| io_error("forest node file is full"))?;
            id
        };
        self.write_header()?;
        Ok(id)
    }

    pub(crate) fn free(&mut self, ids: &[u32]) -> Result<(), ENNError> {
        for &id in ids {
            self.write(id, &Node::Free { next: self.free_head })?;
            self.free_head = id;
        }
        self.write_header()
    }

    pub(crate) fn free_count(&self) -> Result<usize, ENNError> {
        let mut count = 0;
        let mut at = self.free_head;
        while at != NIL {
            match self.read(at)? {
                Node::Free { next } => at = next,
                _ => return Err(io_error("forest free list is corrupt")),
            }
            count += 1;
        }
        Ok(count)
    }
}
