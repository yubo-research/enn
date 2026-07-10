//! Statistical data structures and utilities.

use ndarray::{Array2, Array3};

/// Weighted statistical summary for ENN calculations.
///
/// This struct holds the normalized weights, L2 norms, means, and
/// standard errors computed from weighted neighbor contributions.
///
/// # Fields
///
/// * `w_normalized` - Normalized weights, shape (n_query, k, num_metrics)
/// * `l2` - L2 norm values, shape (n_query, num_metrics)
/// * `mu` - Mean predictions, shape (n_query, num_metrics)
/// * `se` - Standard errors, shape (n_query, num_metrics)
/// * `se_epi` - Epistemic standard error component, shape (n_query, num_metrics)
/// * `se_ale` - Aleatoric standard error component, shape (n_query, num_metrics)
#[derive(Debug, Clone, PartialEq)]
pub struct WeightedStats {
    /// Normalized weights for each neighbor, shape (n_query, k, num_metrics).
    pub w_normalized: Array3<f64>,
    /// L2 norm values for each query point, shape (n_query, num_metrics).
    pub l2: Array2<f64>,
    /// Mean predictions for each query point, shape (n_query, num_metrics).
    pub mu: Array2<f64>,
    /// Standard errors for each query point, shape (n_query, num_metrics).
    pub se: Array2<f64>,
    /// Epistemic standard error component for each query point.
    pub se_epi: Array2<f64>,
    /// Aleatoric standard error component for each query point.
    pub se_ale: Array2<f64>,
}

impl WeightedStats {
    /// Create a new WeightedStats instance.
    ///
    /// # Arguments
    ///
    /// * `w_normalized` - Normalized weights, shape (n_query, k, num_metrics)
    /// * `l2` - L2 norms, shape (n_query, num_metrics)
    /// * `mu` - Means, shape (n_query, num_metrics)
    /// * `se` - Standard errors, shape (n_query, num_metrics)
    ///
    /// # Panics
    ///
    /// Panics if arrays don't have compatible shapes.
    pub fn new(
        w_normalized: Array3<f64>,
        l2: Array2<f64>,
        mu: Array2<f64>,
        se: Array2<f64>,
        se_epi: Array2<f64>,
        se_ale: Array2<f64>,
    ) -> Self {
        let n_query = l2.nrows();
        assert_eq!(
            w_normalized.shape()[0],
            n_query,
            "w_normalized rows must match l2 rows"
        );
        assert_eq!(mu.nrows(), n_query, "mu rows must match l2 rows");
        assert_eq!(se.nrows(), n_query, "se rows must match l2 rows");
        assert_eq!(se_epi.nrows(), n_query, "se_epi rows must match l2 rows");
        assert_eq!(se_ale.nrows(), n_query, "se_ale rows must match l2 rows");
        assert_eq!(
            w_normalized.shape()[2],
            l2.ncols(),
            "w_normalized last dim must match l2 cols"
        );
        assert_eq!(mu.ncols(), l2.ncols(), "mu cols must match l2 cols");
        assert_eq!(se.ncols(), l2.ncols(), "se cols must match l2 cols");
        assert_eq!(se_epi.ncols(), l2.ncols(), "se_epi cols must match l2 cols");
        assert_eq!(se_ale.ncols(), l2.ncols(), "se_ale cols must match l2 cols");

        Self {
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
        self.l2.nrows()
    }

    /// Get the number of neighbors per query.
    pub fn n_neighbors(&self) -> usize {
        self.w_normalized.shape()[1]
    }

    /// Get the number of metrics.
    pub fn n_metrics(&self) -> usize {
        self.l2.ncols()
    }

    /// Check if there are no query points.
    pub fn is_empty(&self) -> bool {
        self.l2.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_weighted_stats_creation() {
        let stats = WeightedStats::new(
            array![[[0.5, 0.5]]], // 1 query, 1 neighbor, 2 metrics
            array![[1.0, 1.0]],   // 1 query, 2 metrics
            array![[0.0, 0.5]],
            array![[0.1, 0.2]],
            array![[0.1, 0.2]],
            array![[0.0, 0.0]],
        );

        assert_eq!(stats.n_queries(), 1);
        assert_eq!(stats.n_neighbors(), 1);
        assert_eq!(stats.n_metrics(), 2);
        assert!(!stats.is_empty());
    }

    #[test]
    fn test_weighted_stats_multiple_queries() {
        let stats = WeightedStats::new(
            array![[[0.5, 0.5], [0.5, 0.5]], [[0.3, 0.3], [0.7, 0.7]]], // 2 queries, 2 neighbors, 2 metrics
            array![[1.0, 1.0], [0.5, 0.5]],                             // 2 queries, 2 metrics
            array![[0.0, 0.0], [1.0, 1.0]],
            array![[0.1, 0.1], [0.2, 0.2]],
            array![[0.1, 0.1], [0.2, 0.2]],
            array![[0.0, 0.0], [0.0, 0.0]],
        );

        assert_eq!(stats.n_queries(), 2);
        assert_eq!(stats.n_neighbors(), 2);
        assert_eq!(stats.n_metrics(), 2);
    }

    #[test]
    fn test_weighted_stats_empty() {
        let stats = WeightedStats::new(
            Array3::zeros((0, 0, 0)),
            Array2::zeros((0, 0)),
            Array2::zeros((0, 0)),
            Array2::zeros((0, 0)),
            Array2::zeros((0, 0)),
            Array2::zeros((0, 0)),
        );

        assert!(stats.is_empty());
        assert_eq!(stats.n_queries(), 0);
    }

    #[test]
    #[should_panic(expected = "w_normalized rows must match l2 rows")]
    fn test_mismatched_lengths() {
        WeightedStats::new(
            Array3::zeros((2, 2, 2)), // 2 queries
            Array2::zeros((1, 2)),    // 1 query - mismatch
            Array2::zeros((2, 2)),
            Array2::zeros((2, 2)),
            Array2::zeros((2, 2)),
            Array2::zeros((2, 2)),
        );
    }

    #[test]
    fn test_clone_equality() {
        let stats = WeightedStats::new(
            array![[[0.5, 0.5]]],
            array![[1.0, 1.0]],
            array![[0.0, 0.5]],
            array![[0.1, 0.2]],
            array![[0.1, 0.2]],
            array![[0.0, 0.0]],
        );
        let cloned = stats.clone();
        assert_eq!(stats, cloned);
    }
}
