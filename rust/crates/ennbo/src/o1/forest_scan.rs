use super::forest_node::{Node, NIL};
use super::forest_tree::Tree;
use crate::error::ENNError;

pub(crate) struct Scan {
    pub i: usize,
    pub below: Option<usize>,
    pub ids: Vec<u32>,
    pub stack: Vec<u32>,
    pub start: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScanStep {
    Busy,
    Found(usize),
    Balanced,
    Broken,
}

impl Scan {
    pub(crate) fn new(anc: &[u32], start: u32) -> Self {
        let i = anc.len() - 1;
        Self {
            i,
            below: None,
            ids: Vec::new(),
            stack: vec![anc[i]],
            start,
        }
    }
}

impl Tree {
    pub(crate) fn scan(
        &mut self,
        s: &mut Scan,
        anc: &[u32],
        alpha: f64,
        budget: usize,
        limit: usize,
    ) -> Result<(usize, ScanStep), ENNError> {
        let mut cost = 0;
        let step = self.scan_inner(s, anc, alpha, (budget, limit), &mut cost);
        self.stats.scan_nodes += cost as u64;
        Ok((cost, step?))
    }

    fn scan_inner(
        &self,
        s: &mut Scan,
        anc: &[u32],
        alpha: f64,
        (budget, limit): (usize, usize),
        cost: &mut usize,
    ) -> Result<ScanStep, ENNError> {
        loop {
            while let Some(at) = s.stack.pop() {
                *cost += 1;
                match self.nodes.read(at)? {
                    Node::Leaf { ids, next } => {
                        s.ids.extend(ids.into_iter().filter(|&id| id < s.start));
                        s.stack.extend((next != NIL).then_some(next));
                    }
                    Node::Split { left, right, .. } => s.stack.extend([left, right]),
                    Node::Free { .. } => return Ok(ScanStep::Broken),
                }
                if *cost >= budget || s.ids.len() > limit {
                    return Ok(ScanStep::Busy);
                }
            }
            if let Some(below) = s.below {
                let size = s.ids.len();
                if below.max(size - below) as f64 > alpha * size as f64 {
                    return Ok(ScanStep::Found(s.i - 1));
                }
                s.i -= 1;
            }
            if s.i == 0 {
                return Ok(ScanStep::Balanced);
            }
            let Node::Split { left, right, .. } = self.nodes.read(anc[s.i - 1])? else {
                return Ok(ScanStep::Broken);
            };
            let sibling = match (left == anc[s.i], right == anc[s.i]) {
                (true, _) => right,
                (_, true) => left,
                _ => return Ok(ScanStep::Broken),
            };
            s.below = Some(s.ids.len());
            s.stack.push(sibling);
        }
    }

    pub(crate) fn free_some(
        &mut self,
        stack: &mut Vec<u32>,
        budget: usize,
    ) -> Result<usize, ENNError> {
        let mut done = Vec::new();
        while done.len() < budget {
            let Some(at) = stack.pop() else { break };
            match self.nodes.read(at)? {
                Node::Leaf { next, .. } => stack.extend((next != NIL).then_some(next)),
                Node::Split { left, right, .. } => stack.extend([left, right]),
                Node::Free { .. } => continue,
            }
            done.push(at);
        }
        self.nodes.free(&done)?;
        Ok(done.len().max(1))
    }
}
