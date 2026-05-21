//! Incremental incumbent candidate tracker for TuRBO optimizers.

use ndarray::{Array1, ArrayView2};

const ALL_CANDIDATES_M_THRESHOLD: usize = 1_000_000_000;

pub struct IncrementalIncumbentTracker {
    m: usize,
    observation_count: usize,
    all_indices: Vec<usize>,
    max_y: f64,
    max_indices: Vec<usize>,
    max_initialized: bool,
    top_entries: Vec<(usize, f64)>,
    per_metric_entries: Vec<Vec<(usize, f64)>>,
    use_all_candidates: bool,
    use_noiseless_max: bool,
    use_scalar_topm: bool,
}

impl IncrementalIncumbentTracker {
    pub fn new(m: usize, noise_aware: bool, num_metrics: usize) -> Self {
        assert!(m >= 1);
        assert!(num_metrics >= 1);
        let use_all_candidates = m >= ALL_CANDIDATES_M_THRESHOLD;
        let use_noiseless_max = !use_all_candidates && num_metrics == 1 && !noise_aware;
        let use_scalar_topm = !use_all_candidates && num_metrics == 1 && noise_aware;
        let use_multi_topm = !use_all_candidates && num_metrics > 1;
        let per_metric_entries = if use_multi_topm {
            (0..num_metrics).map(|_| Vec::new()).collect()
        } else {
            Vec::new()
        };
        Self {
            m,
            observation_count: 0,
            all_indices: Vec::new(),
            max_y: f64::NEG_INFINITY,
            max_indices: Vec::new(),
            max_initialized: false,
            top_entries: Vec::new(),
            per_metric_entries,
            use_all_candidates,
            use_noiseless_max,
            use_scalar_topm,
        }
    }

    pub fn tell(&mut self, index: usize, y: &Array1<f64>) {
        self.observation_count += 1;
        if self.use_all_candidates {
            self.all_indices.push(index);
            return;
        }
        if self.use_noiseless_max {
            let value = y[0];
            if !self.max_initialized || value > self.max_y {
                self.max_y = value;
                self.max_indices = vec![index];
                self.max_initialized = true;
            } else if value == self.max_y {
                self.max_indices.push(index);
            }
            return;
        }
        if self.use_scalar_topm {
            Self::push_top_m(&mut self.top_entries, index, y[0], self.m);
            return;
        }
        for (col, entries) in self.per_metric_entries.iter_mut().enumerate() {
            Self::push_top_m(entries, index, y[col], self.m);
        }
    }

    pub fn ask(&self) -> Vec<usize> {
        if self.observation_count == 0 {
            return Vec::new();
        }
        if self.use_all_candidates {
            let mut out = self.all_indices.clone();
            out.sort_unstable();
            return out;
        }
        if self.use_noiseless_max {
            let mut out = self.max_indices.clone();
            out.sort_unstable();
            return out;
        }
        if self.use_scalar_topm {
            return Self::sorted_indices(&self.top_entries);
        }
        let mut union: std::collections::BTreeSet<usize> = std::collections::BTreeSet::new();
        for entries in &self.per_metric_entries {
            for idx in Self::sorted_indices(entries) {
                union.insert(idx);
            }
        }
        union.into_iter().collect()
    }

    pub fn observation_count(&self) -> usize {
        self.observation_count
    }

    pub fn reset(&mut self) {
        self.observation_count = 0;
        self.all_indices.clear();
        self.max_y = f64::NEG_INFINITY;
        self.max_indices.clear();
        self.max_initialized = false;
        self.top_entries.clear();
        for entries in &mut self.per_metric_entries {
            entries.clear();
        }
    }

    pub fn rebuild(&mut self, y_obs: &ArrayView2<f64>) {
        self.reset();
        if y_obs.nrows() == 0 {
            return;
        }
        for i in 0..y_obs.nrows() {
            let row = y_obs.row(i).to_owned();
            self.tell(i, &row);
        }
    }

    fn push_top_m(entries: &mut Vec<(usize, f64)>, index: usize, value: f64, m: usize) {
        entries.push((index, value));
        entries.sort_by(|&(ia, va), &(ib, vb)| {
            vb.total_cmp(&va).then_with(|| ia.cmp(&ib))
        });
        entries.truncate(m);
    }

    fn sorted_indices(entries: &[(usize, f64)]) -> Vec<usize> {
        let mut out: Vec<usize> = entries.iter().map(|(i, _)| *i).collect();
        out.sort_unstable();
        out
    }
}

pub fn tracker_m_from_enn_k(k: i32) -> usize {
    k.max(1) as usize
}

pub fn tracker_m_no_surrogate() -> usize {
    ALL_CANDIDATES_M_THRESHOLD + 1
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_noiseless_max_ties() {
        let mut t = IncrementalIncumbentTracker::new(3, false, 1);
        t.tell(0, &array![1.0]);
        assert_eq!(t.ask(), vec![0]);
        t.tell(1, &array![3.0]);
        assert_eq!(t.ask(), vec![1]);
        t.tell(3, &array![3.0]);
        assert_eq!(t.ask(), vec![1, 3]);
    }

    #[test]
    fn test_noise_aware_top_m() {
        let mut t = IncrementalIncumbentTracker::new(3, true, 1);
        for (i, v) in [(0, 0.1), (1, 0.9), (2, 0.3), (3, 0.8), (4, 0.2)] {
            t.tell(i, &array![v]);
        }
        assert_eq!(t.ask(), vec![1, 2, 3]);
    }

    #[test]
    fn test_multi_metric_union() {
        let mut t = IncrementalIncumbentTracker::new(2, false, 2);
        let rows = [
            array![10.0, 0.0],
            array![9.0, 1.0],
            array![0.0, 10.0],
            array![1.0, 9.0],
            array![5.0, 5.0],
        ];
        for (i, row) in rows.into_iter().enumerate() {
            t.tell(i, &row);
        }
        assert_eq!(t.ask(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_all_candidates_mode() {
        let mut t = IncrementalIncumbentTracker::new(tracker_m_no_surrogate(), false, 1);
        for (i, v) in [(0, 0.1), (1, 0.3), (2, -0.2)] {
            t.tell(i, &array![v]);
        }
        assert_eq!(t.ask(), vec![0, 1, 2]);
    }

    #[test]
    fn test_tracker_m_from_enn_k_and_rebuild() {
        assert_eq!(tracker_m_from_enn_k(4), 4);
        let mut t = IncrementalIncumbentTracker::new(2, true, 1);
        t.rebuild(&array![[1.0], [3.0], [2.0]].view());
        assert_eq!(t.ask(), vec![1, 2]);
        t.reset();
        assert!(t.ask().is_empty());
    }

    #[test]
    fn test_top_m_tie_break_lower_index() {
        let mut t = IncrementalIncumbentTracker::new(2, true, 1);
        for (i, v) in [(0, 5.0), (1, 5.0), (2, 5.0)] {
            t.tell(i, &array![v]);
        }
        assert_eq!(t.ask(), vec![0, 1]);
    }
}
