//! K-Nearest Neighbors index for ENN.
//!
//! Provides efficient KNN search using either a K-d tree implementation
//! or exact matrix-based search.

use ndarray::{Array1, Array2, ArrayView2, Axis};
use std::collections::BinaryHeap;
use thiserror::Error;

/// Errors that can occur during index operations.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum IndexError {
    /// Invalid dimensionality.
    #[error("Invalid shape: expected {expected} dims, got {got}")]
    InvalidShape { expected: usize, got: usize },
    /// Invalid search parameter.
    #[error("Invalid search parameter: {0}")]
    InvalidParameter(String),
    /// Empty index.
    #[error("Index is empty")]
    EmptyIndex,
}

/// Driver for KNN search algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum IndexDriver {
    /// Use exact search with matrix operations.
    #[default]
    Exact,
    /// Use K-d tree for approximate search.
    KDTree,
    /// Use HNSW for approximate search.
    HNSW,
}

/// K-Nearest Neighbors index for ENN.
///
/// Stores training data and provides efficient KNN search.
pub struct ENNIndex {
    /// Scaled training data.
    train_x_scaled: Array2<f64>,
    /// Number of dimensions.
    num_dim: usize,
    /// Scale factors for each dimension.
    x_scale: Array1<f64>,
    /// Whether to scale inputs.
    scale_x: bool,
    /// Search driver.
    driver: IndexDriver,
}

impl ENNIndex {
    /// Create a new ENNIndex.
    ///
    /// # Arguments
    ///
    /// * `train_x_scaled` - Pre-scaled training data
    /// * `num_dim` - Number of dimensions
    /// * `x_scale` - Scale factors for each dimension
    /// * `scale_x` - Whether to scale new inputs
    /// * `driver` - Search algorithm to use
    pub fn new(
        train_x_scaled: Array2<f64>,
        num_dim: usize,
        x_scale: Array1<f64>,
        scale_x: bool,
        driver: IndexDriver,
    ) -> Self {
        Self {
            train_x_scaled,
            num_dim,
            x_scale,
            scale_x,
            driver,
        }
    }

    /// Add new points to the index.
    ///
    /// # Arguments
    ///
    /// * `x` - New points to add, shape (n, num_dim)
    ///
    /// # Errors
    ///
    /// Returns `IndexError::InvalidShape` if dimensions don't match.
    pub fn add(&mut self, x: &ArrayView2<f64>) -> Result<(), IndexError> {
        if x.ncols() != self.num_dim {
            return Err(IndexError::InvalidShape {
                expected: self.num_dim,
                got: x.ncols(),
            });
        }

        // Scale if needed
        let x_scaled = if self.scale_x {
            x / &self.x_scale.view().insert_axis(Axis(0))
        } else {
            x.to_owned()
        };

        // Concatenate with existing data
        self.train_x_scaled = ndarray::concatenate![
            Axis(0),
            self.train_x_scaled.view(),
            x_scaled.view()
        ];

        Ok(())
    }

    /// Search for k nearest neighbors.
    ///
    /// # Arguments
    ///
    /// * `x` - Query points, shape (n_query, num_dim)
    /// * `search_k` - Number of neighbors to find
    /// * `exclude_nearest` - If true, exclude the nearest neighbor (return 1..k instead of 0..k)
    ///
    /// # Returns
    ///
    /// Tuple of (distances_squared, indices), each with shape (n_query, k).
    ///
    /// # Errors
    ///
    /// Returns `IndexError` if parameters are invalid.
    pub fn search(
        &self,
        x: &ArrayView2<f64>,
        search_k: i32,
        exclude_nearest: bool,
    ) -> Result<(Array2<f64>, Array2<i64>), IndexError> {
        if search_k <= 0 {
            return Err(IndexError::InvalidParameter(format!(
                "search_k must be > 0, got {}",
                search_k
            )));
        }

        if x.ncols() != self.num_dim {
            return Err(IndexError::InvalidShape {
                expected: self.num_dim,
                got: x.ncols(),
            });
        }

        let n_train = self.train_x_scaled.nrows();
        let k = (search_k as usize).min(n_train);

        // Scale query points
        let x_scaled = if self.scale_x {
            x / &self.x_scale.view().insert_axis(Axis(0))
        } else {
            x.to_owned()
        };

        // Compute distances based on driver
        let (mut dist2s, mut indices) = match self.driver {
            IndexDriver::Exact => self.exact_search(&x_scaled.view(), k),
            IndexDriver::KDTree => self.kdtree_search(&x_scaled.view(), k),
            IndexDriver::HNSW => self.hnsw_search(&x_scaled.view(), k),
        };

        // Exclude nearest if requested (need at least 2 to exclude 1)
        if exclude_nearest {
            if k < 2 {
                return Err(IndexError::InvalidParameter(
                    "exclude_nearest=True requires search_k >= 2".to_string(),
                ));
            }
            dist2s = dist2s.slice_axis(Axis(1), ndarray::Slice::from(1..)).to_owned();
            indices = indices.slice_axis(Axis(1), ndarray::Slice::from(1..)).to_owned();
        }

        Ok((dist2s, indices))
    }

    /// Exact search using matrix operations.
    fn exact_search(
        &self,
        x: &ArrayView2<f64>,
        k: usize,
    ) -> (Array2<f64>, Array2<i64>) {
        let n_query = x.nrows();
        let n_train = self.train_x_scaled.nrows();

        // Compute squared distances using matrix operations
        // d2[i,j] = ||x[i] - train_x[j]||^2
        //         = ||x[i]||^2 + ||train_x[j]||^2 - 2 * x[i] dot train_x[j]
        let x2: Array1<f64> = x.map_axis(Axis(1), |row| row.dot(&row));
        let y2: Array1<f64> = self
            .train_x_scaled
            .map_axis(Axis(1), |row| row.dot(&row));

        // d2 = x2[:, None] + y2[None, :] - 2 * x.dot(train_x.T)
        let xy = x.dot(&self.train_x_scaled.t());
        let d2 =
            x2.view().insert_axis(Axis(1)).to_owned() + y2.view().insert_axis(Axis(0)) - 2.0 * xy;

        // Find k smallest distances for each query
        let mut dist2s = Array2::from_elem((n_query, k), f64::INFINITY);
        let mut indices = Array2::from_elem((n_query, k), -1i64);

        for i in 0..n_query {
            // Get distances for this query
            let row = d2.row(i);

            // Keep a max-heap of size k, so the worst current neighbor is at the top.
            let mut heap: BinaryHeap<(ordered_float::OrderedFloat<f64>, usize)> =
                BinaryHeap::with_capacity(k);

            for j in 0..n_train {
                let dist = ordered_float::OrderedFloat(row[j]);
                if heap.len() < k {
                    heap.push((dist, j));
                } else if let Some(&(top_dist, _)) = heap.peek() {
                    if dist < top_dist {
                        heap.pop();
                        heap.push((dist, j));
                    }
                }
            }

            // Extract and sort by distance
            let mut neighbors: Vec<(f64, usize)> = heap
                .into_iter()
                .map(|(dist, idx)| (dist.into_inner(), idx))
                .collect();
            neighbors.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Store results
            for (j, (dist, idx)) in neighbors.iter().enumerate() {
                dist2s[[i, j]] = *dist;
                indices[[i, j]] = *idx as i64;
            }
        }

        (dist2s, indices)
    }

    /// K-d tree search (placeholder for future implementation).
    fn kdtree_search(
        &self,
        x: &ArrayView2<f64>,
        k: usize,
    ) -> (Array2<f64>, Array2<i64>) {
        // For now, delegate to exact search
        // TODO: Implement proper K-d tree using kdtree crate
        self.exact_search(x, k)
    }

    /// HNSW search (placeholder for future implementation).
    fn hnsw_search(
        &self,
        x: &ArrayView2<f64>,
        k: usize,
    ) -> (Array2<f64>, Array2<i64>) {
        // For now, delegate to exact search.
        self.exact_search(x, k)
    }

    /// Get the number of training points.
    pub fn len(&self) -> usize {
        self.train_x_scaled.nrows()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.train_x_scaled.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_index_creation() {
        let train_x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let x_scale = array![1.0, 1.0];

        let index = ENNIndex::new(train_x, 2, x_scale, false, IndexDriver::Exact);

        assert_eq!(index.len(), 3);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_index_search() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let x_scale = array![1.0, 1.0];
        let index = ENNIndex::new(train_x, 2, x_scale, false, IndexDriver::Exact);

        let query = array![[0.0, 0.0]];
        let (dist2s, indices) = index.search(&query.view(), 2, false).unwrap();

        // Nearest should be point 0 at distance 0
        assert_eq!(indices[[0, 0]], 0);
        assert_eq!(dist2s[[0, 0]], 0.0);

        // Second nearest should be either point 1 or 2 at distance 1
        assert!(dist2s[[0, 1]] > 0.0);
    }

    #[test]
    fn test_index_search_exclude_nearest() {
        let train_x = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let x_scale = array![1.0, 1.0];
        let index = ENNIndex::new(train_x, 2, x_scale, false, IndexDriver::Exact);

        let query = array![[0.0, 0.0]];
        let (dist2s, indices) = index.search(&query.view(), 2, true).unwrap();

        // Should only return 1 neighbor (excluding the nearest)
        assert_eq!(dist2s.ncols(), 1);
        assert_eq!(indices.ncols(), 1);

        // The result should be point 1 or 2, not point 0
        assert!(indices[[0, 0]] != 0);
    }

    #[test]
    fn test_index_add() {
        let train_x = array![[0.0, 0.0]];
        let x_scale = array![1.0, 1.0];
        let mut index = ENNIndex::new(train_x, 2, x_scale, false, IndexDriver::Exact);

        let new_point = array![[1.0, 1.0]];
        index.add(&new_point.view()).unwrap();

        assert_eq!(index.len(), 2);
    }

    #[test]
    fn test_invalid_search_k() {
        let train_x = array![[0.0, 0.0]];
        let x_scale = array![1.0, 1.0];
        let index = ENNIndex::new(train_x, 2, x_scale, false, IndexDriver::Exact);

        let query = array![[0.0, 0.0]];
        let result = index.search(&query.view(), 0, false);

        assert!(matches!(result, Err(IndexError::InvalidParameter(_))));
    }

    #[test]
    fn test_invalid_dimensions() {
        let train_x = array![[0.0, 0.0]];
        let x_scale = array![1.0, 1.0];
        let index = ENNIndex::new(train_x, 2, x_scale, false, IndexDriver::Exact);

        let query = array![[0.0, 0.0, 0.0]]; // Wrong dimensions
        let result = index.search(&query.view(), 1, false);

        assert!(matches!(result, Err(IndexError::InvalidShape { expected: 2, got: 3 })));
    }

    #[test]
    fn test_scaled_search() {
        // When scale_x=true, train_x_scaled should be pre-scaled
        let _train_x = array![[0.0, 0.0], [2.0, 2.0]]; // Original unscaled
        let x_scale = array![2.0, 2.0];
        // train_x_scaled = [[0,0], [1,1]]
        let index = ENNIndex::new(
            array![[0.0, 0.0], [1.0, 1.0]], // Pre-scaled
            2,
            x_scale,
            true,
            IndexDriver::Exact,
        );

        // Query [2.0, 2.0] unscaled becomes [1.0, 1.0] scaled, which matches point 1
        let query = array![[2.0, 2.0]];
        let (dist2s, indices) = index.search(&query.view(), 1, false).unwrap();

        // Should find point 1 (index 1) at distance 0
        assert_eq!(indices[[0, 0]], 1);
        assert!(dist2s[[0, 0]] < 0.0001);
    }
}
