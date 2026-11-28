use std::collections::{BTreeMap, BTreeSet};

use super::graph::{GraphIndex, Pieces};
use super::graph_box::{empty_box, union_into};
use super::graph_io::Items;
use super::graph_split::{pick_links, split_to_cap};
use crate::error::ENNError;

pub(crate) type Updates = BTreeMap<u32, BTreeMap<u32, (Vec<f64>, Vec<f64>)>>;

impl GraphIndex {
    pub(crate) fn propagate(&mut self, touched: Updates, w: &[f64]) -> Result<(), ENNError> {
        let (mut pending, mut level) = (touched, 1);
        while !pending.is_empty() {
            let mut next = Updates::new();
            for (blk, ups) in pending {
                self.update_record(level, blk, ups, w, &mut next)?;
            }
            pending = next;
            level += 1;
        }
        Ok(())
    }

    fn update_record(&mut self, level: usize, blk: u32, ups: BTreeMap<u32, (Vec<f64>, Vec<f64>)>, w: &[f64], next: &mut Updates) -> Result<(), ENNError> {
        let d = self.meta.d;
        let mut rec = self.files.read_up(level, blk)?;
        ups.iter().for_each(|(c, (v, _))| rec.upsert(*c, v, d));
        let parts = match rec.len() > self.meta.fan {
            true => split_to_cap(rec, d, w, self.meta.fan),
            false => vec![rec],
        };
        if parts.len() > 1 && level + 1 == self.meta.height {
            self.grow_root(blk)?;
        }
        let parent = match level + 1 < self.meta.height {
            true => Some(self.files.read_par(level, blk)?),
            false => None,
        };
        self.meta.stats.up_splits += parts.len() as u64 - 1;
        for (k, part) in parts.iter().enumerate() {
            let id = match k {
                0 => blk,
                _ => self.new_id(level),
            };
            self.files.write_up(level, id, part)?;
            let bx = match parts.len() {
                1 => {
                    let mut b = self.boxes.read(level, blk)?;
                    ups.values().for_each(|(_, cb)| union_into(&mut b, cb));
                    b
                }
                _ => self.boxes.union_of(level - 1, &part.ids)?,
            };
            self.boxes.write(level, id, &bx)?;
            if k > 0 {
                for &c in &part.ids {
                    self.files.write_par(level - 1, c, id)?;
                }
            }
            if let Some(pp) = parent {
                self.files.write_par(level, id, pp)?;
                next.entry(pp).or_default().insert(id, (part.mean(d), bx));
            }
        }
        Ok(())
    }

    fn grow_root(&mut self, old: u32) -> Result<(), ENNError> {
        let level = self.meta.height;
        self.meta.height += 1;
        self.meta.counts.push(1);
        self.files.ensure_levels(self.meta.height)?;
        self.boxes.ensure(self.meta.height)?;
        self.boxes.write(level, 0, &empty_box(self.meta.d))?;
        let mut root = Items::default();
        root.push(old, &vec![0.0; self.meta.d]);
        self.files.write_up(level, 0, &root)?;
        self.files.write_par(level - 1, old, 0)?;
        self.meta.root = 0;
        Ok(())
    }

    pub(crate) fn rewire(&mut self, p: &Pieces, parent: u32, w: &[f64]) -> Result<(), ENNError> {
        let (d, m) = (self.meta.d, self.meta.m);
        let mut cands = Items::default();
        p.ids.iter().zip(&p.cents).for_each(|(&id, c)| cands.push(id, c));
        let sib = self.files.read_up(1, parent)?;
        for i in 0..sib.len() {
            if !p.ids.contains(&sib.ids[i]) {
                cands.push(sib.ids[i], sib.row(i, d));
            }
        }
        let old = self.files.read_links(p.ids[0])?;
        for &o in &old {
            if !cands.ids.contains(&o) {
                cands.push(o, &self.files.read_cent(o)?);
            }
        }
        let mut back: BTreeSet<u32> = old.iter().copied().filter(|o| !p.ids.contains(o)).collect();
        for (k, &id) in p.ids.iter().enumerate() {
            let links = pick_links(&p.cents[k], &cands, id, d, w, m);
            self.files.write_links(id, &links)?;
            back.extend(links.into_iter().filter(|l| !p.ids.contains(l)));
        }
        for q in back {
            self.relink(q, p, w)?;
        }
        Ok(())
    }

    fn relink(&mut self, q: u32, p: &Pieces, w: &[f64]) -> Result<(), ENNError> {
        let mut cc = Items::default();
        for l in self.files.read_links(q)? {
            if !p.ids.contains(&l) {
                cc.push(l, &self.files.read_cent(l)?);
            }
        }
        p.ids.iter().zip(&p.cents).for_each(|(&id, c)| cc.push(id, c));
        let cq = self.files.read_cent(q)?;
        let links = pick_links(&cq, &cc, q, self.meta.d, w, self.meta.m);
        self.files.write_links(q, &links)?;
        self.meta.stats.link_writes += 1;
        Ok(())
    }
}
