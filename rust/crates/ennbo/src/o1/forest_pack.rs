use std::collections::VecDeque;
use std::path::Path;

use super::forest::ForestIndex;
use super::forest_node::{Node, NodeFile, NIL};
use super::forest_tree::Tree;
use super::lsh::io_error;
use crate::error::ENNError;

pub(crate) const PAGE_BYTES: u64 = 4096;

pub(crate) fn children(node: &Node) -> Vec<u32> {
    match *node {
        Node::Split { left, right, .. } => vec![left, right],
        Node::Leaf { next, .. } if next != NIL => vec![next],
        _ => Vec::new(),
    }
}

pub(crate) fn page_ids(page: u64, rec: u64) -> Vec<u32> {
    if rec * 2 > PAGE_BYTES {
        return vec![page as u32 + 1];
    }
    let (lo, hi) = (page * PAGE_BYTES, (page + 1) * PAGE_BYTES);
    (lo.div_ceil(rec)..hi / rec).filter(|&i| i >= 1 && (i + 1) * rec <= hi).map(|i| i as u32).collect()
}

pub(crate) struct PackJob {
    map: Vec<u32>,
    order: Vec<u32>,
    blocks: VecDeque<u32>,
    page: u64,
    end: u32,
    written: usize,
    out: Option<NodeFile>,
    dst: std::path::PathBuf,
}

impl PackJob {
    pub(crate) fn new(nodes: &NodeFile, dst: &Path) -> Self {
        Self { map: vec![NIL; nodes.num_nodes() as usize], order: Vec::new(), blocks: VecDeque::from([nodes.root()]), page: 0, end: 1, written: 0, out: None, dst: dst.to_path_buf() }
    }

    fn walk_page(&mut self, nodes: &NodeFile) -> Result<usize, ENNError> {
        let slots = page_ids(self.page, nodes.record_bytes() as u64);
        self.page += 1;
        let mut queue = VecDeque::new();
        for &slot in &slots {
            let Some(at) = queue.pop_front().or_else(|| self.blocks.pop_front()) else { break };
            self.map[at as usize] = slot;
            self.order.push(at);
            self.end = self.end.max(slot + 1);
            queue.extend(children(&nodes.read_disk(at)?));
        }
        self.blocks.extend(queue);
        Ok(slots.len().max(1))
    }

    fn open_out(&mut self, nodes: &NodeFile) -> Result<(), ENNError> {
        let (dim, payload) = nodes.dim_payload();
        let mut out = NodeFile::create(&self.dst, nodes.cap(), dim, payload)?;
        out.set_layout(self.map[nodes.root() as usize], self.end)?;
        let mut used = vec![false; self.end as usize];
        self.order.iter().for_each(|&at| used[self.map[at as usize] as usize] = true);
        for slot in (1..self.end).filter(|&s| !used[s as usize]) {
            out.write(slot, &Node::Free { next: NIL })?;
        }
        self.out = Some(out);
        Ok(())
    }

    pub(crate) fn step(&mut self, nodes: &NodeFile, left: &mut usize) -> Result<Option<NodeFile>, ENNError> {
        while *left > 0 {
            if !self.blocks.is_empty() {
                *left = left.saturating_sub(self.walk_page(nodes)?);
                continue;
            }
            if self.out.is_none() {
                self.open_out(nodes)?;
            }
            let Some(&at) = self.order.get(self.written) else { return Ok(self.out.take()) };
            let out = self.out.as_ref().ok_or_else(|| io_error("pack output is missing"))?;
            out.write(self.map[at as usize], &remap(nodes.read_disk(at)?, &self.map))?;
            self.written += 1;
            *left -= 1;
        }
        Ok(None)
    }
}

fn remap(node: Node, map: &[u32]) -> Node {
    let m = |id: u32| if id == NIL { NIL } else { map[id as usize] };
    match node {
        Node::Split { cut, thr, left, right } => Node::Split { cut, thr, left: m(left), right: m(right) },
        Node::Leaf { ids, next } => Node::Leaf { ids, next: m(next) },
        other => other,
    }
}

pub(crate) fn pack(nodes: &NodeFile, dst: &Path) -> Result<NodeFile, ENNError> {
    let mut left = usize::MAX;
    PackJob::new(nodes, dst).step(nodes, &mut left)?.ok_or_else(|| io_error("pack did not finish"))
}

impl Tree {
    pub(crate) fn compact(&mut self, path: &Path) -> Result<usize, ENNError> {
        self.staged = None;
        self.release.clear();
        let tmp = path.with_extension("pack");
        let mut packed = pack(&self.nodes, &tmp)?;
        std::fs::rename(&tmp, path).map_err(io_error)?;
        packed.set_cache(self.nodes.shared_cache());
        packed.absorb_counters(&self.nodes);
        self.nodes = packed;
        self.packed_at = self.nodes.num_nodes();
        Ok(self.nodes.num_nodes() as usize)
    }
}

pub(crate) const REPACK_MIN_NODES: u32 = 64;

impl Tree {
    pub(crate) fn needs_repack(&self) -> bool {
        u64::from(self.nodes.num_nodes()) * 10 >= u64::from(self.packed_at.max(REPACK_MIN_NODES)) * 13
    }
}

impl ForestIndex {
    pub(crate) fn repack_one(&mut self) -> Result<bool, ENNError> {
        let Some(t) = self.active_trees().find(|&t| self.trees[t].needs_repack()) else { return Ok(false) };
        let path = self.tree_file(t);
        let nodes = self.trees[t].compact(&path)?;
        self.maint.repacks += 1;
        self.maint.repacked_nodes += nodes as u64;
        Ok(true)
    }

    pub(crate) fn compact(&mut self) -> Result<usize, ENNError> {
        let mut nodes = 0;
        for t in 0..self.trees.len() {
            let path = self.tree_file(t);
            nodes += self.trees[t].compact(&path)?;
        }
        Ok(nodes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_slots_hold_whole_records_and_skip_the_header() {
        assert_eq!(page_ids(0, 152), (1..26).collect::<Vec<u32>>());
        assert_eq!(page_ids(1, 152), (27..53).collect::<Vec<u32>>());
        assert!(page_ids(1, 152).iter().all(|&i| u64::from(i) * 152 >= 4096 && u64::from(i + 1) * 152 <= 8192));
        assert_eq!(page_ids(3, 3000), vec![4]);
    }
}
