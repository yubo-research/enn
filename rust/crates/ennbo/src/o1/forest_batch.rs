
use std::sync::OnceLock;

use super::forest::ForestIndex;
use super::forest_node::Node;
use super::forest_tree::{row, RowSource, SplitRef, Tree};
use super::row_order::Rows;
use crate::backend::MmapColumnStore;
use crate::error::ENNError;

pub(crate) fn buffer_rows() -> usize {
    static ROWS: OnceLock<usize> = OnceLock::new();
    *ROWS.get_or_init(|| {
        std::env::var("ENN_O1_BUFFER")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
    })
}

pub(crate) fn group_rows() -> usize {
    static ROWS: OnceLock<usize> = OnceLock::new();
    *ROWS.get_or_init(|| {
        std::env::var("ENN_O1_GROUP")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(256)
    })
}

impl Tree {
    pub(crate) fn batch_order(
        &self,
        ids: &[u32],
        rows: Rows<'_>,
        inv_scale: &[f64],
    ) -> Result<Vec<u32>, ENNError> {
        Ok(self.batch_groups(ids, rows, inv_scale, 1)?.concat())
    }

    pub(crate) fn batch_groups(
        &self,
        ids: &[u32],
        rows: Rows<'_>,
        inv_scale: &[f64],
        stop: usize,
    ) -> Result<Vec<Vec<u32>>, ENNError> {
        let src = RowSource {
            store: rows,
            cache: None,
            pread: false,
        };
        let mut out = Vec::new();
        let mut stack = vec![(self.nodes.root(), ids.to_vec())];
        while let Some((node, group)) = stack.pop() {
            let split = match self.nodes.read(node)? {
                Node::Split {
                    cut,
                    thr,
                    left,
                    right,
                } if group.len() > stop => SplitRef {
                    cut,
                    thr,
                    left,
                    right,
                },
                _ => {
                    out.push(group);
                    continue;
                }
            };
            let (mut lo, mut hi) = (Vec::new(), Vec::new());
            for id in group {
                let x = row(rows, id)?;
                let (_, _, near, _) = self.branch(&x, inv_scale, &split, src, None)?;
                if near == split.left {
                    lo.push(id);
                } else {
                    hi.push(id);
                }
            }
            stack.push((split.right, hi));
            stack.push((split.left, lo));
        }
        Ok(out)
    }

    pub(crate) fn insert_batch(
        &mut self,
        ids: &[u32],
        rows: Rows<'_>,
        inv_scale: &[f64],
        alpha: f64,
        group: bool,
    ) -> Result<(), ENNError> {
        let width = super::forest_prefetch::prefetch_width();
        if width > 1 && !ids.is_empty() && self.warm.due() {
            let slow = self.prefetch_counted(ids, rows, inv_scale, width)?;
            self.warm.observe(slow, ids.len());
        }
        if !group || ids.len() < 2 {
            return ids
                .iter()
                .try_for_each(|&id| self.insert(id, rows, inv_scale, alpha));
        }
        self.mixed = true;
        let stop = group_rows();
        let chunks = if stop > 1 {
            self.batch_groups(ids, rows, inv_scale, stop)?
        } else {
            vec![ids.to_vec()]
        };
        for chunk in chunks {
            self.batch_order(&chunk, rows, inv_scale)?
                .into_iter()
                .try_for_each(|id| self.insert(id, rows, inv_scale, alpha))?;
        }
        Ok(())
    }
}

impl ForestIndex {
    pub(crate) fn buffered(&self, nrows: usize) -> std::ops::Range<u32> {
        self.num_rows.min(nrows) as u32..nrows as u32
    }

    pub(super) fn holds_back(&self, rows: &MmapColumnStore) -> bool {
        let cap = self.params.buffer;
        cap > 0 && rows.nrows.saturating_sub(self.num_rows) < cap
    }

    pub(super) fn flush_buffer(&mut self, rows: &MmapColumnStore) -> Result<(), ENNError> {
        if self.num_rows >= rows.nrows {
            return Ok(());
        }
        self.insert_from(rows, self.num_rows, 0)
    }

    pub(super) fn maintain_held(
        &mut self,
        rows: &MmapColumnStore,
        fresh: usize,
    ) -> Result<(), ENNError> {
        let t0 = std::time::Instant::now();
        let done = self.maintain(rows, fresh.saturating_mul(self.params.repair_rows))?;
        self.maint.repair_ns += t0.elapsed().as_nanos() as u64;
        if done > 0 {
            self.commit()?;
        }
        Ok(())
    }
}
