
use ndarray::{Array2, Array3};

#[derive(Debug, Clone, PartialEq)]
pub struct WeightedStats {
    pub w_normalized: Array3<f64>,
    pub l2: Array2<f64>,
    pub mu: Array2<f64>,
    pub se: Array2<f64>,
    pub se_epi: Array2<f64>,
    pub se_ale: Array2<f64>,
}

impl WeightedStats {
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

    pub fn n_queries(&self) -> usize {
        self.l2.nrows()
    }

    pub fn n_neighbors(&self) -> usize {
        self.w_normalized.shape()[1]
    }

    pub fn n_metrics(&self) -> usize {
        self.l2.ncols()
    }

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
            array![[[0.5, 0.5]]],
            array![[1.0, 1.0]],
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
            array![[[0.5, 0.5], [0.5, 0.5]], [[0.3, 0.3], [0.7, 0.7]]],
            array![[1.0, 1.0], [0.5, 0.5]],
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
            Array3::zeros((2, 2, 2)),
            Array2::zeros((1, 2)),
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
