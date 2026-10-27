//! Objective-independent projection construction and bounded-input transforms.
use crate::error::ENNError;
use ndarray::{s, Array2, ArrayView2, Axis};
use ndarray_linalg::SVD;

/// Normalize embeddings, retain the requested SVD variance, and fix column signs.
pub fn projection(embeddings: &ArrayView2<f64>, cutoff: f64) -> Result<Array2<f64>, ENNError> {
    if embeddings.is_empty()
        || embeddings.iter().any(|v| !v.is_finite())
        || !cutoff.is_finite()
        || cutoff <= 0.0
        || cutoff > 1.0
    {
        return Err(ENNError::InvalidParameter(
            "embeddings must be a finite nonempty matrix and variance_cutoff in (0, 1]".into(),
        ));
    }
    let mut vectors = embeddings.to_owned();
    for mut row in vectors.rows_mut() {
        let norm = row.dot(&row).sqrt();
        if norm > 0.0 {
            row /= norm;
        }
    }
    let mean = vectors
        .mean_axis(Axis(0))
        .ok_or_else(|| ENNError::InvalidParameter("empty embeddings".into()))?;
    vectors -= &mean;
    let (u, singular, _) = vectors
        .svd(true, false)
        .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
    let rank = singular
        .iter()
        .filter(|v| **v > vectors.nrows().max(vectors.ncols()) as f64 * f64::EPSILON)
        .count();
    let total: f64 = singular.iter().take(rank).map(|v| v * v).sum();
    let mut retained = rank;
    let mut cumulative = 0.0;
    for (j, v) in singular.iter().take(rank).enumerate() {
        cumulative += v * v;
        if cutoff < 1.0 && cumulative >= cutoff * total {
            retained = j + 1;
            break;
        }
    }
    let mut result = u
        .ok_or_else(|| ENNError::InvalidParameter("SVD returned no vectors".into()))?
        .slice(s![.., ..retained])
        .to_owned();
    for (j, mut column) in result.columns_mut().into_iter().enumerate() {
        let pivot = column
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.abs().total_cmp(&b.1.abs()))
            .map(|v| v.0)
            .unwrap_or(0);
        let sign = if column[pivot] >= 0.0 { 1.0 } else { -1.0 };
        column *= sign * singular[j];
    }
    let norm = result.iter().map(|v| v * v).sum::<f64>().sqrt();
    if norm > 0.0 {
        result *= (retained as f64).sqrt() / norm;
    }
    Ok(result)
}

/// Scale semantic energy relative to the original coordinate energy.
pub fn weighted(
    projection: &ArrayView2<f64>,
    relative_weight: f64,
) -> Result<Array2<f64>, ENNError> {
    if !relative_weight.is_finite() || relative_weight < 0.0 {
        return Err(ENNError::InvalidParameter(
            "relative_weight must be finite and nonnegative".into(),
        ));
    }
    Ok(projection
        * (relative_weight * projection.nrows() as f64 / projection.ncols().max(1) as f64).sqrt())
}

/// Normalize original coordinates and produce their semantic representation.
pub fn transform(
    x: &ArrayView2<f64>,
    bounds: &ArrayView2<f64>,
    projection: &Array2<f64>,
    only: bool,
) -> Result<Array2<f64>, ENNError> {
    validate_bounds(bounds, projection.nrows())?;
    crate::semantic::validate(Some(projection), bounds.nrows(), only)?;
    if x.ncols() != bounds.nrows() || x.iter().any(|v| !v.is_finite()) {
        return Err(ENNError::InvalidParameter(
            "inputs must be finite rows with one column per variable".into(),
        ));
    }
    let normalized = Array2::from_shape_fn(x.dim(), |(i, j)| {
        (x[[i, j]] - bounds[[j, 0]]) / (bounds[[j, 1]] - bounds[[j, 0]])
    });
    crate::semantic::features(&normalized.view(), Some(projection), only)
}

/// Require finite increasing bounds for each original coordinate.
pub fn validate_bounds(bounds: &ArrayView2<f64>, dimensions: usize) -> Result<(), ENNError> {
    if bounds.dim() != (dimensions, 2)
        || bounds
            .rows()
            .into_iter()
            .any(|r| !r[0].is_finite() || !r[1].is_finite() || r[1] <= r[0])
    {
        return Err(ENNError::InvalidParameter(
            "bounds must be finite increasing pairs, one per variable".into(),
        ));
    }
    Ok(())
}

/// Recover originals when their coordinates are included in the representation.
pub fn inverse(z: &ArrayView2<f64>, bounds: &ArrayView2<f64>) -> Result<Array2<f64>, ENNError> {
    validate_bounds(bounds, bounds.nrows())?;
    if z.ncols() < bounds.nrows() {
        return Err(ENNError::InvalidParameter(
            "semantic-only features are not invertible".into(),
        ));
    }
    Ok(Array2::from_shape_fn(
        (z.nrows(), bounds.nrows()),
        |(i, j)| (z[[i, j]] + 0.5) * (bounds[[j, 1]] - bounds[[j, 0]]) + bounds[[j, 0]],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    #[test]
    fn projection_energy_and_bounded_roundtrip() {
        let p = projection(&array![[1., 0.], [0., 1.], [1., 1.]].view(), 1.0).unwrap();
        assert_eq!(p.dim(), (3, 2));
        assert!((p.iter().map(|v| v * v).sum::<f64>() - 2.0).abs() < 1e-12);
        let p = weighted(&p.view(), 0.25).unwrap();
        let bounds = array![[0., 2.], [0., 2.], [0., 2.]];
        let x = array![[0., 1., 2.]];
        let z = transform(&x.view(), &bounds.view(), &p, false).unwrap();
        assert_eq!(inverse(&z.view(), &bounds.view()).unwrap(), x);
        assert_eq!(
            transform(&x.view(), &bounds.view(), &p, true)
                .unwrap()
                .ncols(),
            2
        );
        assert!(projection(&array![[f64::NAN]].view(), 0.9).is_err());
        assert!(weighted(&p.view(), -1.0).is_err());
        assert!(validate_bounds(&array![[1., 0.]].view(), 1).is_err());
    }
}
