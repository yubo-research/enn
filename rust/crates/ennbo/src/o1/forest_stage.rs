use super::forest_node::Node;
use super::forest_pack::children;
use super::forest_tree::{row, Tree};
use crate::backend::MmapColumnStore;
use crate::error::ENNError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Placed {
    Present,
    Appended,
    Split(usize),
}

pub(crate) struct Parted {
    pub node: Node,
    pub left: (u32, Vec<u32>),
    pub right: (u32, Vec<u32>),
}

#[derive(Debug, Default)]
pub(crate) struct Staged {
    pub slot: u32,
    pub above: Vec<u32>,
    root: u32,
    work: Vec<(u32, Vec<u32>)>,
    late: Vec<u32>,
}

impl Tree {
    pub(super) fn note_late(&mut self, path: &[u32], id: u32) {
        if let Some(s) = self.staged.as_mut().filter(|s| path.contains(&s.slot)) {
            s.late.push(id);
        }
    }

    pub(super) fn rebuild_or_stage(&mut self, path: &[u32], size: usize, rows: &MmapColumnStore, inv_scale: &[f64]) -> Result<(), ENNError> {
        let (w, above) = path.split_last().ok_or_else(|| super::lsh::io_error("forest path is empty"))?;
        if self.staged.as_ref().is_some_and(|s| s.slot == *w || s.above.contains(w)) {
            return Ok(());
        }
        let limit = self.split.stage_above as usize;
        if limit == 0 || self.split.stage_work == 0 || size <= limit {
            return self.rebuild(*w, rows, inv_scale);
        }
        if self.staged.is_some() {
            return Ok(());
        }
        let (mut ids, mut nodes) = (Vec::new(), Vec::new());
        self.gather(*w, &mut ids, &mut nodes)?;
        let root = self.nodes.alloc()?;
        let n = ids.len() as u64;
        self.stats.rebuilds += 1;
        self.stats.rebuilt_rows += n;
        self.stats.max_rebuild_rows = self.stats.max_rebuild_rows.max(n);
        self.staged = Some(Staged { slot: *w, above: above.to_vec(), root, work: vec![(root, ids)], ..Staged::default() });
        Ok(())
    }

    pub(crate) fn advance(&mut self, rows: &MmapColumnStore, inv_scale: &[f64], budget: usize) -> Result<(), ENNError> {
        let mut left = budget;
        while left > 0 {
            if let Some(mut s) = self.staged.take() {
                if let Some((slot, ids)) = s.work.pop() {
                    left = left.saturating_sub(ids.len());
                    if let Some(p) = self.part(slot, ids, rows, inv_scale)? {
                        self.nodes.write(slot, &p.node)?;
                        self.stats.splits += 1;
                        s.work.extend([p.left, p.right]);
                    }
                    self.staged = Some(s);
                } else if s.late.is_empty() {
                    left -= 1;
                    self.swap_in(s)?;
                } else {
                    let batch = s.late.split_off(s.late.len() - s.late.len().min(left));
                    left -= batch.len();
                    for id in batch {
                        let (_, leaf) = self.descend_from(s.root, row(rows, id)?, inv_scale, rows)?;
                        self.place(id, leaf, rows, inv_scale, false)?;
                    }
                    self.staged = Some(s);
                }
            } else if self.release.is_empty() {
                return Ok(());
            } else {
                let batch = self.release.split_off(self.release.len() - self.release.len().min(left));
                left -= batch.len();
                for &at in &batch {
                    let next = children(&self.nodes.read(at)?);
                    self.release.extend(next);
                }
                self.nodes.free(&batch)?;
            }
        }
        Ok(())
    }

    fn swap_in(&mut self, s: Staged) -> Result<(), ENNError> {
        let old = self.nodes.read(s.slot)?;
        self.nodes.write(s.slot, &self.nodes.read(s.root)?)?;
        self.nodes.free(&[s.root])?;
        self.release.extend(children(&old));
        Ok(())
    }
}
