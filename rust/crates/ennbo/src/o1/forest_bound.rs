use std::collections::{BinaryHeap, HashSet};

use super::forest_node::{Code, Dir};
use super::forest_rules::{ForestRules, ProbeRule};
use super::forest_tree::RowSource;
use super::search::{fetch_row, weighted_dist2, X_COLUMN};
use crate::error::ENNError;

#[derive(PartialEq)]
struct Dist(f64);

impl Eq for Dist {}

impl PartialOrd for Dist {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Dist {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&other.0)
    }
}

pub(crate) fn dual_norm(dir: &Dir, weights: &[f64]) -> f64 {
    let sum = |v: &mut dyn Iterator<Item = f64>| v.zip(weights).map(|(a, w)| a * a / w).sum::<f64>().sqrt();
    match &dir.code {
        Code::I8(q) => f64::from(dir.scale) * sum(&mut q.iter().map(|&a| f64::from(a))),
        Code::F32(v) => sum(&mut v.iter().map(|&a| f64::from(a))),
    }
}

pub(crate) struct Scorer<'w> {
    pub weights: &'w [f64],
    need: usize,
    bound: f64,
    patience: u32,
    stale: u32,
    squared: bool,
    top: BinaryHeap<Dist>,
    seen: Option<HashSet<u32>>,
    codes: bool,
    narrow: Option<(Vec<f32>, Vec<f32>)>,
    pub scored: Vec<(f64, u32)>,
    pub coded: Vec<(f64, u32)>,
}

impl<'w> Scorer<'w> {
    pub(crate) fn new(weights: &'w [f64], need: usize, invs: &[Vec<f64>], rules: &ForestRules) -> Self {
        let codes = rules.rerank > 0;
        Self { weights, need: need.max(1), bound: rules.bound, patience: rules.patience, stale: 0, squared: rules.probe == ProbeRule::SumSquares, top: BinaryHeap::new(), seen: (invs.len() > 1).then(HashSet::new), codes, narrow: None, scored: Vec::new(), coded: Vec::new() }
    }

    pub(crate) fn kth(&self) -> f64 {
        if self.top.len() < self.need { f64::INFINITY } else { self.top.peek().map_or(f64::INFINITY, |d| d.0) }
    }

    pub(crate) fn prunes(&self, cost: f64) -> bool {
        let lb2 = if self.squared { cost } else { cost * cost };
        self.bound > 0.0 && lb2 > self.bound * self.kth()
    }

    pub(crate) fn settled(&self) -> bool {
        self.patience > 0 && self.stale >= self.patience
    }

    pub(crate) fn add(&mut self, query: &[f64], ids: &[u32], src: RowSource<'_>) -> Result<(), ENNError> {
        let before = self.kth();
        for &id in ids {
            if self.seen.as_mut().is_some_and(|s| !s.insert(id)) {
                continue;
            }
            let code = if self.codes {
                let (q, w) = self.narrow.get_or_insert_with(|| (query.iter().map(|&v| v as f32).collect(), self.weights.iter().map(|&v| v as f32).collect()));
                src.store.code_dist2(id, q, w)
            } else {
                None
            };
            let d = match code {
                Some(d) => {
                    self.coded.push((d, id));
                    d
                }
                None => {
                    let d = weighted_dist2(query, &fetch_row(src.store, src.cache, X_COLUMN, id as usize)?, self.weights);
                    self.scored.push((d, id));
                    d
                }
            };
            if self.top.len() < self.need {
                self.top.push(Dist(d));
            } else if d < self.kth() {
                self.top.pop();
                self.top.push(Dist(d));
            }
        }
        self.stale = if self.kth() < before { 0 } else { self.stale + 1 };
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn margins_scale_to_the_current_metric_and_pruning_needs_a_full_top() {
        let dir = Dir::quantize(&[0.6, 0.8], true);
        assert!((dual_norm(&dir, &[1.0, 1.0]) - 1.0).abs() < 1e-6);
        assert!((dual_norm(&dir, &[4.0, 1.0]) - 0.73f64.sqrt()).abs() < 1e-6);
        assert!((dual_norm(&Dir::quantize(&[0.6, 0.8], false), &[1.0, 1.0]) - 1.0).abs() < 0.01);
        let w = [1.0, 1.0];
        let mut s = Scorer::new(&w, 2, &[vec![1.0, 1.0]], &ForestRules::default());
        assert!(!s.prunes(100.0));
        s.top.push(Dist(4.0));
        s.top.push(Dist(1.0));
        assert_eq!(s.kth(), 4.0);
        assert!(!s.prunes(1.9));
        assert!(s.prunes(2.1));
        assert!(!s.settled());
    }
}
