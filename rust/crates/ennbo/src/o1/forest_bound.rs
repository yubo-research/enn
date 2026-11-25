use std::collections::{BinaryHeap, HashSet};

use super::forest_node::Dir;
use super::forest_rules::ForestRules;
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
    f64::from(dir.scale) * dir.code.iter().zip(weights).map(|(&a, w)| f64::from(a) * f64::from(a) / w).sum::<f64>().sqrt()
}

pub(crate) struct Scorer<'w> {
    pub weights: &'w [f64],
    need: usize,
    bound: f64,
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
        Self { weights, need: need.max(1), bound: rules.bound, top: BinaryHeap::new(), seen: (invs.len() > 1).then(HashSet::new), codes, narrow: None, scored: Vec::new(), coded: Vec::new() }
    }

    pub(crate) fn kth(&self) -> f64 {
        if self.top.len() < self.need { f64::INFINITY } else { self.top.peek().map_or(f64::INFINITY, |d| d.0) }
    }

    pub(crate) fn prunes(&self, cost: f64) -> bool {
        self.bound > 0.0 && cost * cost > self.bound * self.kth()
    }

    pub(crate) fn add(&mut self, query: &[f64], ids: &[u32], src: RowSource<'_>) -> Result<(), ENNError> {
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
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn margins_scale_to_the_current_metric_and_pruning_needs_a_full_top() {
        let dir = Dir::quantize(&[0.6, 0.8]);
        assert!((dual_norm(&dir, &[1.0, 1.0]) - 1.0).abs() < 0.01);
        assert!((dual_norm(&dir, &[4.0, 1.0]) - 0.73f64.sqrt()).abs() < 0.01);
        let w = [1.0, 1.0];
        let mut s = Scorer::new(&w, 2, &[vec![1.0, 1.0]], &ForestRules::default());
        assert!(!s.prunes(100.0));
        s.top.push(Dist(4.0));
        s.top.push(Dist(1.0));
        assert_eq!(s.kth(), 4.0);
        assert!(!s.prunes(1.9));
        assert!(s.prunes(2.1));
    }
}
