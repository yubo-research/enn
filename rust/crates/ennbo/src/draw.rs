//! Data structures for ENN internal computations and draws.

use ndarray::{Array2, Array3};

/// Internal state from a weighted posterior computation.
///
/// This holds the intermediate results from computing weighted statistics
/// for the ENN posterior, which can be used for sampling.
///
/// # Fields
///
/// * `idx` - Indices of neighbors used for each query point
/// * `w_normalized` - Normalized weights for each neighbor, shape (n_query, k, num_metrics)
/// * `l2` - L2 norm of weights for each query point, shape (n_query, num_metrics)
/// * `mu` - Predictive mean for each query point, shape (n_query, num_metrics)
/// * `se` - Predictive standard error for each query point, shape (n_query, num_metrics)
/// * `se_epi` - Epistemic standard error component, shape (n_query, num_metrics)
/// * `se_ale` - Aleatoric standard error component, shape (n_query, num_metrics)
#[derive(Debug, Clone, PartialEq)]
pub struct DrawInternals {
    /// Neighbor indices for each query point.
    pub idx: Vec<Vec<usize>>,
    /// Normalized weights for each neighbor of each query point, shape (n_query, k, num_metrics).
    pub w_normalized: Array3<f64>,
    /// L2 norm of weights for each query point, shape (n_query, num_metrics).
    pub l2: Array2<f64>,
    /// Predictive means, shape (n_query, num_metrics).
    pub mu: Array2<f64>,
    /// Predictive standard errors, shape (n_query, num_metrics).
    pub se: Array2<f64>,
    /// Epistemic standard error component, shape (n_query, num_metrics).
    pub se_epi: Array2<f64>,
    /// Aleatoric standard error component, shape (n_query, num_metrics).
    pub se_ale: Array2<f64>,
}

impl DrawInternals {
    /// Create new DrawInternals.
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

    /// Get the number of query points.
    pub fn n_queries(&self) -> usize {
        self.mu.nrows()
    }

    /// Get the number of neighbors per query.
    pub fn n_neighbors(&self) -> usize {
        self.w_normalized.shape()[1]
    }
}

/// Data about neighbors for query points.
///
/// Holds the distances, indices, and target values for neighbors
/// of multiple query points.
///
/// # Fields
///
/// * `dist2s` - Squared distances to neighbors, shape (n_query, k)
/// * `idx` - Indices of neighbors for each query point
/// * `y_neighbors` - Target values for neighbors, shape (n_query, k, num_metrics)
/// * `k` - Number of neighbors
#[derive(Debug, Clone, PartialEq)]
pub struct NeighborData {
    /// Squared distances to neighbors, shape (n_query, k).
    pub dist2s: Array2<f64>,
    /// Indices of neighbors for each query point.
    pub idx: Vec<Vec<usize>>,
    /// Target values for neighbors, shape (n_query, k, num_metrics).
    pub y_neighbors: Array2<f64>,
    /// Number of neighbors.
    pub k: usize,
}

impl NeighborData {
    /// Create new NeighborData.
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

    /// Get the number of query points.
    pub fn n_queries(&self) -> usize {
        self.dist2s.nrows()
    }

    /// Get the number of neighbors.
    pub fn k(&self) -> usize {
        self.k
    }

    /// Check if there are no neighbors.
    pub fn is_empty(&self) -> bool {
        self.k == 0
    }
}

/// Data for conditional posterior computation.
///
/// Combines training data with what-if scenarios.
///
/// # Fields
///
/// * `x` - Input features
/// * `y` - Target values
/// * `yvar` - Optional observation noise variance
#[derive(Debug, Clone, PartialEq)]
pub struct Candidates {
    /// Input features.
    pub x: Array2<f64>,
    /// Target values.
    pub y: Array2<f64>,
    /// Optional observation noise variance.
    pub yvar: Option<Array2<f64>>,
}

impl Candidates {
    /// Create new Candidates without noise.
    pub fn new(x: Array2<f64>, y: Array2<f64>) -> Self {
        Self { x, y, yvar: None }
    }

    /// Create new Candidates with noise.
    pub fn with_noise(x: Array2<f64>, y: Array2<f64>, yvar: Array2<f64>) -> Self {
        Self {
            x,
            y,
            yvar: Some(yvar),
        }
    }

    /// Get the number of candidates.
    pub fn len(&self) -> usize {
        self.x.nrows()
    }

    /// Check if there are no candidates.
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }
}

/// Internal state from conditional posterior computation.
#[derive(Debug, Clone, PartialEq)]
pub struct ConditionalPosteriorDrawInternals {
    /// Base draw internals.
    pub base_internals: DrawInternals,
    /// What-if candidate data.
    pub whatif_candidates: Candidates,
}

impl ConditionalPosteriorDrawInternals {
    /// Create new ConditionalPosteriorDrawInternals.
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
        let w = array![[[0.5, 0.5], [0.5, 0.5]], [[0.3, 0.7], [0.3, 0.7]]]; // 2 queries, 2 neighbors, 2 metrics
        let l2 = array![[0.7, 0.7], [0.5, 0.5]]; // 2 queries, 2 metrics
        let mu = array![[1.0, 1.0], [2.0, 2.0]];
        let se = array![[0.1, 0.1], [0.2, 0.2]];

        let internals = DrawInternals::new(idx.clone(), w, l2, mu, se.clone(), se, array![[0.0, 0.0], [0.0, 0.0]]);

        assert_eq!(internals.n_queries(), 2);
        assert_eq!(internals.n_neighbors(), 2);
        assert_eq!(internals.idx, idx);
    }

    #[test]
    fn test_neighbor_data() {
        let dist2s = array![[1.0, 2.0, 3.0]]; // 1 query, 3 neighbors
        let idx = vec![vec![0, 1, 2]];
        let y = array![[1.0, 2.0, 3.0]]; // Flattened (1*3, 1)

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
            array![[[1.0]]], // 1 query, 1 neighbor, 1 metric
            array![[1.0]],   // 1 query, 1 metric
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
