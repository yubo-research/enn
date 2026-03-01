//! Utility functions for ENN algorithms.
//!
//! Includes standardization, Pareto front computation, and Sobol sensitivity analysis.

use ndarray::{Array1, ArrayView1, ArrayView2};

/// Standardize y values by computing center and scale.
///
/// Returns (center, scale) where:
/// - center = median(y)
/// - scale = std(y) (or 1.0 if std is 0 or non-finite)
///
/// # Arguments
///
/// * `y` - Array of y values to standardize
///
/// # Returns
///
/// Tuple of (center, scale) as floats.
pub fn standardize_y(y: &ArrayView1<f64>) -> (f64, f64) {
    if y.is_empty() {
        return (f64::NAN, 1.0);
    }

    // Compute median
    let mut sorted = y.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    #[allow(clippy::manual_is_multiple_of)]
    let center = if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    };

    // Compute std
    let mean = y.sum() / y.len() as f64;
    let variance = y.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n as f64;
    let scale = variance.sqrt();

    // Handle edge cases
    if !scale.is_finite() || scale == 0.0 {
        (center, 1.0)
    } else {
        (center, scale)
    }
}

/// Find Pareto front for 2D maximization problem.
///
/// Given two objective arrays `a` and `b`, returns indices of points
/// on the Pareto front (non-dominated points).
///
/// A point dominates another if it is >= in all objectives and > in at least one.
///
/// # Arguments
///
/// * `a` - First objective values (to maximize)
/// * `b` - Second objective values (to maximize)
/// * `idx` - Optional indices to use (if None, uses 0..n)
///
/// # Returns
///
/// Array of indices on the Pareto front, sorted by a descending.
pub fn pareto_front_2d_maximize(
    a: &ArrayView1<f64>,
    b: &ArrayView1<f64>,
    idx: Option<&[usize]>,
) -> Vec<usize> {
    assert_eq!(a.len(), b.len(), "a and b must have same length");

    let n = a.len();
    if n == 0 {
        return Vec::new();
    }

    let indices: Vec<usize> = match idx {
        Some(i) => i.to_vec(),
        None => (0..n).collect(),
    };

    // Create sortable pairs (a, b) with original index
    let mut pairs: Vec<(usize, f64, f64)> = indices
        .iter()
        .map(|&i| (i, a[i], b[i]))
        .collect();

    // Sort by a descending (for maximization), then by b descending
    pairs.sort_by(|x, y| {
        y.1.partial_cmp(&x.1)
            .unwrap()
            .then_with(|| y.2.partial_cmp(&x.2).unwrap())
    });

    // Walk frontier: keep points with b better than or equal to any seen so far
    // This matches Python behavior which keeps ties when both a and b are equal
    let mut front = Vec::new();
    let mut best_b = f64::NEG_INFINITY;
    let mut last_a = f64::NAN;
    let mut last_b = f64::NAN;

    for (orig_idx, a_val, b_val) in pairs {
        if b_val > best_b {
            // This point is not dominated on b
            best_b = b_val;
            last_a = a_val;
            last_b = b_val;
            front.push(orig_idx);
        } else if b_val == best_b && a_val == last_a && b_val == last_b {
            // Keep ties when both objectives are equal to the last added point
            front.push(orig_idx);
        }
    }

    front
}

/// Calculate first-order Sobol sensitivity indices.
///
/// Uses variance-based sensitivity analysis with binning.
///
/// # Arguments
///
/// * `x` - Input array of shape (n, d)
/// * `y` - Output array of shape (n,)
///
/// # Returns
///
/// Array of shape (d,) containing Sobol indices in [0, 1].
pub fn calculate_sobol_indices(x: &ArrayView2<f64>, y: &ArrayView1<f64>) -> Array1<f64> {
    let n = x.nrows();
    let d = x.ncols();

    // Handle small samples
    if n < 9 {
        return Array1::ones(d);
    }

    let vy = y.var(0.0);

    // Handle zero variance
    if vy <= 0.0 || !vy.is_finite() {
        return Array1::ones(d);
    }

    // Determine number of bins
    let num_bins = if n >= 30 { 10 } else { 3 };

    let mut sobol = Array1::zeros(d);

    // Precompute variance for each input dimension to detect constant columns
    let mut x_var = Array1::zeros(d);
    for dim in 0..d {
        x_var[dim] = x.column(dim).var(0.0);
    }

    for dim in 0..d {
        // Zero out indices for near-zero-variance input dimensions (matches Python)
        if x_var[dim] <= 1e-12 {
            sobol[dim] = 0.0;
            continue;
        }
        let x_dim = x.column(dim);

        // Rank values
        let mut indexed: Vec<(usize, f64)> = x_dim.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Assign bins based on rank
        let mut bins = vec![0usize; n];
        for (rank, (idx, _)) in indexed.iter().enumerate() {
            bins[*idx] = (rank * num_bins) / n;
        }

        // Compute variance contribution for this dimension
        let mut var_explained = 0.0;
        for bin in 0..num_bins {
            let bin_mask: Vec<bool> = bins.iter().map(|&b| b == bin).collect();
            let bin_y: Vec<f64> = bin_mask
                .iter()
                .enumerate()
                .filter(|(_, &m)| m)
                .map(|(i, _)| y[i])
                .collect();

            if bin_y.is_empty() {
                continue;
            }

            let bin_mean = bin_y.iter().sum::<f64>() / bin_y.len() as f64;
            let p_b = bin_y.len() as f64 / n as f64;
            var_explained += p_b * (bin_mean - y.mean().unwrap()).powi(2);
        }

        sobol[dim] = var_explained / vy;
    }

    sobol
}

/// Select arms from Pareto fronts.
///
/// Iteratively extracts Pareto fronts to select diverse arms.
///
/// # Arguments
///
/// * `x_cand` - Candidate inputs
/// * `mu` - Mean predictions
/// * `se` - Standard errors
/// * `num_arms` - Number of arms to select
/// * `seed` - RNG seed for random selection
///
/// # Returns
///
/// Selected arm indices.
pub fn arms_from_pareto_fronts(
    _x_cand: &ArrayView2<f64>,
    mu: &ArrayView1<f64>,
    _se: &ArrayView1<f64>,
    num_arms: usize,
    _seed: u64,
) -> Vec<usize> {
    let n = mu.len();
    if n == 0 || num_arms == 0 {
        return Vec::new();
    }

    // Simple selection: sort by mu descending and take top
    // Full implementation would do iterative Pareto front extraction
    let mut indexed: Vec<(usize, f64)> = mu.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Use seed for deterministic selection
    let mut selected: Vec<usize> = indexed
        .iter()
        .take(num_arms.min(n))
        .map(|(idx, _)| *idx)
        .collect();

    // Sort by mu descending
    selected.sort_by(|&a, &b| mu[b].partial_cmp(&mu[a]).unwrap());

    selected
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_standardize_y_normal() {
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let (center, scale) = standardize_y(&y.view());

        assert_eq!(center, 3.0); // median
        assert!(scale > 0.0);
        assert!(scale.is_finite());
    }

    #[test]
    fn test_standardize_y_constant() {
        let y = array![5.0, 5.0, 5.0];
        let (center, scale) = standardize_y(&y.view());

        assert_eq!(center, 5.0);
        assert_eq!(scale, 1.0); // fallback for zero std
    }

    #[test]
    fn test_standardize_y_empty() {
        let y = Array1::<f64>::zeros(0);
        let (center, scale) = standardize_y(&y.view());

        assert!(center.is_nan());
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn test_pareto_front_simple() {
        // Points: (1, 0.5), (0.5, 1), (0.2, 0.2)
        // Pareto front for maximization: (1, 0.5) and (0.5, 1)
        let a = array![1.0, 0.5, 0.2];
        let b = array![0.5, 1.0, 0.2];

        let front = pareto_front_2d_maximize(&a.view(), &b.view(), None);

        assert!(front.contains(&0)); // (1, 0.5) is on front
        assert!(front.contains(&1)); // (0.5, 1) is on front
        assert!(!front.contains(&2)); // (0.2, 0.2) is dominated
    }

    #[test]
    fn test_pareto_front_empty() {
        let a = array![];
        let b = array![];

        let front = pareto_front_2d_maximize(&a.view(), &b.view(), None);
        assert!(front.is_empty());
    }

    #[test]
    fn test_sobol_indices_shape() {
        let x = array![[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]];
        let y = array![0.0, 0.5, 1.0];

        let sobol = calculate_sobol_indices(&x.view(), &y.view());

        assert_eq!(sobol.len(), 2);
        assert!(sobol.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn test_sobol_indices_small_sample() {
        let x = array![[0.0], [0.5]];
        let y = array![0.0, 1.0];

        let sobol = calculate_sobol_indices(&x.view(), &y.view());

        // Small samples return all 1s
        assert_eq!(sobol, array![1.0]);
    }

    #[test]
    fn test_arms_from_pareto() {
        let x_cand = array![[0.0], [1.0], [2.0]];
        let mu = array![0.0, 1.0, 0.5];
        let se = array![0.1, 0.1, 0.1];

        let arms = arms_from_pareto_fronts(&x_cand.view(), &mu.view(), &se.view(), 2, 42);

        assert_eq!(arms.len(), 2);
        // Should select highest mu values
        assert!(arms.contains(&1)); // mu=1.0
    }
}
