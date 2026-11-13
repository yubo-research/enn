use std::collections::BinaryHeap;

use crate::index::tree_search_knobs::stop_epsilon;
use crate::small_n_search::OrderedFloat;

pub(super) struct TopK {
    pub(super) k: usize,
    pub(super) heap: BinaryHeap<(OrderedFloat<f64>, u32)>,
}

impl TopK {
    pub(super) fn offer(&mut self, row: u32, dist: f64) {
        if self.heap.len() < self.k {
            self.heap.push((OrderedFloat(dist), row));
        } else if let Some(&(worst, worst_row)) = self.heap.peek() {
            if (OrderedFloat(dist), row) < (worst, worst_row) {
                self.heap.pop();
                self.heap.push((OrderedFloat(dist), row));
            }
        }
    }

    pub(super) fn cannot_improve(&self, bound: f32) -> bool {
        let scale = (1.0 + stop_epsilon()).powi(2);
        let lower =
            (f64::from(bound) * (1.0 - 1e-4) - 64.0 * f32::MIN_POSITIVE as f64).max(0.0) * scale;
        self.heap.len() >= self.k && self.heap.peek().is_some_and(|&(worst, _)| lower > worst.0)
    }

    pub(super) fn limit(&self) -> f64 {
        if self.heap.len() < self.k {
            f64::INFINITY
        } else {
            self.heap.peek().map_or(f64::INFINITY, |p| p.0 .0)
        }
    }

    pub(super) fn into_sorted(self) -> Vec<(u32, f64)> {
        let mut out: Vec<(u32, f64)> = self.heap.into_iter().map(|(d, r)| (r, d.0)).collect();
        out.sort_by(|a, b| {
            OrderedFloat(a.1)
                .cmp(&OrderedFloat(b.1))
                .then(a.0.cmp(&b.0))
        });
        out
    }
}
