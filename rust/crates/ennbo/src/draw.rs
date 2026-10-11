
use ndarray::{Array2, Array3};

#[derive(Debug, Clone, PartialEq)]
pub struct DrawInternals {
    pub idx: Vec<Vec<usize>>,
    pub w_normalized: Array3<f64>,
    pub l2: Array2<f64>,
    pub mu: Array2<f64>,
    pub se: Array2<f64>,
    pub se_epi: Array2<f64>,
    pub se_ale: Array2<f64>,
}

impl DrawInternals {
    pub fn new(
        idx: Vec<Vec<usize>>,
        w_normalized: Array3<f64>,
        l2: Array2<f64>,
        mu: Array2<f64>,
        se: Array2<f64>,
        se_epi: Array2<f64>,
        se_ale: Array2<f64>,
    ) -> Self {
        Self {
            idx,
            w_normalized,
            l2,
            mu,
            se,
            se_epi,
            se_ale,
        }
    }

    pub fn n_queries(&self) -> usize {
        self.mu.nrows()
    }

    pub fn n_neighbors(&self) -> usize {
        self.w_normalized.shape()[1]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NeighborData {
    pub dist2s: Array2<f64>,
    pub idx: Vec<Vec<usize>>,
    pub y_neighbors: Array2<f64>,
    pub k: usize,
}

impl NeighborData {
    pub fn new(
        dist2s: Array2<f64>,
        idx: Vec<Vec<usize>>,
        y_neighbors: Array2<f64>,
        k: usize,
    ) -> Self {
        Self {
            dist2s,
            idx,
            y_neighbors,
            k,
        }
    }

    pub fn n_queries(&self) -> usize {
        self.dist2s.nrows()
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn is_empty(&self) -> bool {
        self.k == 0
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Candidates {
    pub x: Array2<f64>,
    pub y: Array2<f64>,
    pub yvar: Option<Array2<f64>>,
}

impl Candidates {
    pub fn new(x: Array2<f64>, y: Array2<f64>) -> Self {
        Self { x, y, yvar: None }
    }

    pub fn with_noise(x: Array2<f64>, y: Array2<f64>, yvar: Array2<f64>) -> Self {
        Self {
            x,
            y,
            yvar: Some(yvar),
        }
    }

    pub fn len(&self) -> usize {
        self.x.nrows()
    }

    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConditionalPosteriorDrawInternals {
    pub base_internals: DrawInternals,
    pub whatif_candidates: Candidates,
}

impl ConditionalPosteriorDrawInternals {
    pub fn new(base_internals: DrawInternals, whatif_candidates: Candidates) -> Self {
        Self {
            base_internals,
            whatif_candidates,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_draw_internals() {
        let idx = vec![vec![0, 1], vec![1, 2]];
        let w = array![[[0.5, 0.5], [0.5, 0.5]], [[0.3, 0.7], [0.3, 0.7]]];
        let l2 = array![[0.7, 0.7], [0.5, 0.5]];
        let mu = array![[1.0, 1.0], [2.0, 2.0]];
        let se = array![[0.1, 0.1], [0.2, 0.2]];

        let internals = DrawInternals::new(idx.clone(), w, l2, mu, se.clone(), se, array![[0.0, 0.0], [0.0, 0.0]]);

        assert_eq!(internals.n_queries(), 2);
        assert_eq!(internals.n_neighbors(), 2);
        assert_eq!(internals.idx, idx);
    }

    #[test]
    fn test_neighbor_data() {
        let dist2s = array![[1.0, 2.0, 3.0]];
        let idx = vec![vec![0, 1, 2]];
        let y = array![[1.0, 2.0, 3.0]];

        let data = NeighborData::new(dist2s, idx, y, 3);

        assert_eq!(data.n_queries(), 1);
        assert_eq!(data.k(), 3);
        assert!(!data.is_empty());
    }

    #[test]
    fn test_candidates() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![[1.0], [2.0]];

        let candidates = Candidates::new(x, y);

        assert_eq!(candidates.len(), 2);
        assert!(!candidates.is_empty());
        assert!(candidates.yvar.is_none());
    }

    #[test]
    fn test_candidates_with_noise() {
        let x = array![[1.0, 2.0]];
        let y = array![[1.0]];
        let yvar = array![[0.1]];

        let candidates = Candidates::with_noise(x, y, yvar);

        assert!(candidates.yvar.is_some());
    }

    #[test]
    fn test_conditional_posterior_internals() {
        let base = DrawInternals::new(
            vec![vec![0]],
            array![[[1.0]]],
            array![[1.0]],
            array![[1.0]],
            array![[0.1]],
            array![[0.1]],
            array![[0.0]],
        );
        let whatif = Candidates::new(array![[1.0]], array![[1.0]]);

        let internals = ConditionalPosteriorDrawInternals::new(base, whatif);

        assert_eq!(internals.base_internals.n_queries(), 1);
    }
}
