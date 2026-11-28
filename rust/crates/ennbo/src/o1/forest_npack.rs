use std::sync::Mutex;

pub(crate) struct Pack {
    rec: usize,
    keys: Vec<u32>,
    buf: Vec<u8>,
}

impl Pack {
    pub(crate) fn new(rec: usize, budget: usize) -> Option<Self> {
        if rec == 0 || budget < rec {
            return None;
        }
        let n = budget / rec;
        Some(Self {
            rec,
            keys: vec![0; n],
            buf: vec![0u8; n * rec],
        })
    }

    fn slot(&self, id: u32) -> usize {
        (super::lsh::mix64(u64::from(id)) as usize) % self.keys.len()
    }

    pub(crate) fn copied(&self, id: u32) -> Option<Vec<u8>> {
        let i = self.slot(id);
        if self.keys[i] != id {
            return None;
        }
        let rec = self.rec;
        Some(self.buf[i * rec..(i + 1) * rec].to_vec())
    }

    pub(crate) fn put(&mut self, id: u32, raw: &[u8]) {
        let i = self.slot(id);
        let rec = self.rec;
        self.keys[i] = id;
        self.buf[i * rec..(i + 1) * rec].copy_from_slice(&raw[..rec]);
    }

    pub(crate) fn drop_id(&mut self, id: u32) {
        let i = self.slot(id);
        if self.keys[i] == id {
            self.keys[i] = 0;
        }
    }
}

pub(crate) fn from_env(rec: usize) -> Option<Mutex<Pack>> {
    let budget = std::env::var("ENN_PACK_INTERIOR")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    Pack::new(rec, budget).map(Mutex::new)
}

pub(crate) fn copied(pack: &Mutex<Pack>, id: u32) -> Option<Vec<u8>> {
    pack.lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .copied(id)
}

pub(crate) fn store(
    pack: &Mutex<Pack>,
    id: u32,
    split: bool,
    raw: &[u8],
    count_miss: bool,
    misses: &std::sync::atomic::AtomicU64,
) {
    let mut g = pack
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if split {
        g.put(id, raw);
        if count_miss {
            misses.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    } else {
        g.drop_id(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::o1::forest_node::{Cut, Node, NodeFile, NIL};
    use std::sync::atomic::Ordering;

    #[test]
    fn packed_interior_record_serves_the_second_read() {
        let path = std::env::temp_dir().join(format!("enn_pack_{}", std::process::id()));
        let mut nf = NodeFile::create(&path, 4, 0, 0).unwrap();
        nf.pack = Pack::new(nf.record_bytes(), nf.record_bytes()).map(Mutex::new);
        let (a, b) = (nf.alloc().unwrap(), nf.alloc().unwrap());
        nf.write(
            a,
            &Node::Split {
                cut: Cut::Seed(3),
                thr: 1.5,
                left: 1,
                right: 1,
            },
        )
        .unwrap();
        nf.write(
            b,
            &Node::Split {
                cut: Cut::Seed(4),
                thr: 0.5,
                left: 1,
                right: 1,
            },
        )
        .unwrap();
        let before = nf.counters.reads.load(Ordering::Relaxed);
        assert!(matches!(
            nf.read(a).unwrap(),
            Node::Split {
                cut: Cut::Seed(3),
                ..
            }
        ));
        assert!(matches!(
            nf.read(a).unwrap(),
            Node::Split {
                cut: Cut::Seed(3),
                ..
            }
        ));
        assert_eq!(nf.counters.reads.load(Ordering::Relaxed), before + 1);
        assert_eq!(nf.counters.pack_hits.load(Ordering::Relaxed), 1);
        assert_eq!(nf.counters.pack_miss.load(Ordering::Relaxed), 1);
        nf.write(
            a,
            &Node::Split {
                cut: Cut::Seed(9),
                thr: 2.0,
                left: 1,
                right: 1,
            },
        )
        .unwrap();
        assert!(
            matches!(nf.read(a).unwrap(), Node::Split { cut: Cut::Seed(9), thr, .. } if thr == 2.0)
        );
        assert_eq!(nf.counters.reads.load(Ordering::Relaxed), before + 1);
        let leaf = nf.alloc().unwrap();
        nf.write(
            leaf,
            &Node::Leaf {
                ids: vec![1],
                next: NIL,
            },
        )
        .unwrap();
        let leaf_reads = nf.counters.reads.load(Ordering::Relaxed);
        nf.read(leaf).unwrap();
        nf.read(leaf).unwrap();
        assert_eq!(nf.counters.reads.load(Ordering::Relaxed), leaf_reads + 2);
        assert_eq!(nf.counters.pack_hits.load(Ordering::Relaxed), 2);
        let _ = std::fs::remove_file(&path);
    }
}
