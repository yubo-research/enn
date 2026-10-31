use crate::error::ENNError;
use ndarray::{s, Array2, ArrayView2, Axis};

pub fn validate(projection: Option<&Array2<f64>>, dimensions: usize, semantic_only: bool) -> Result<(), ENNError> {
    if let Some(p) = projection {
        if p.nrows() != dimensions || p.iter().any(|v| !v.is_finite()) {
            return Err(ENNError::InvalidParameter("invalid semantic projection".into()));
        }
        if semantic_only && (p.ncols() == 0 || p.iter().all(|v| *v == 0.0)) {
            return Err(ENNError::InvalidParameter("semantic-only features require nonzero rank".into()));
        }
    } else if semantic_only {
        return Err(ENNError::InvalidParameter("semantic_only requires a projection".into()));
    }
    Ok(())
}

pub fn augment(x: &ArrayView2<f64>, projection: Option<&Array2<f64>>) -> Result<Array2<f64>, ENNError> {
    let Some(p) = projection else { return Ok(x.to_owned()); };
    if x.ncols() != p.nrows() {
        return Err(ENNError::InvalidParameter("semantic input dimension mismatch".into()));
    }
    let centered = x - 0.5;
    ndarray::concatenate(Axis(1), &[centered.view(), centered.dot(p).view()])
        .map_err(|e| ENNError::InvalidParameter(e.to_string()))
}

pub fn features(x: &ArrayView2<f64>, projection: Option<&Array2<f64>>, semantic_only: bool) -> Result<Array2<f64>, ENNError> {
    let augmented = augment(x, projection)?;
    if semantic_only {
        let dimensions = projection.ok_or_else(|| ENNError::InvalidParameter("missing projection".into()))?.nrows();
        return Ok(augmented.slice(s![.., dimensions..]).to_owned());
    }
    Ok(augmented)
}

pub fn original_weights(weights: &[f64], projection: Option<&Array2<f64>>, semantic_only: bool) -> Vec<f64> {
    let Some(p) = projection else { return weights.to_vec(); };
    let offset = if semantic_only { 0 } else { p.nrows() };
    let floor = weights.iter().copied().fold(0.0_f64, f64::max).max(1.0) * 1e-12;
    (0..p.nrows()).map(|j| {
        let numerical = if semantic_only { 0.0 } else { weights[j] };
        (numerical + (0..p.ncols()).map(|k| p[[j, k]].powi(2) * weights[offset + k]).sum::<f64>()).max(floor)
    }).collect()
}

pub fn feature_count(projection: Option<&Array2<f64>>, dimensions: usize, semantic_only: bool) -> usize {
    match projection { Some(p) if semantic_only => p.ncols(), Some(p) => dimensions + p.ncols(), None => dimensions }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn fixed_features_roundtrip_and_validate() {
        let p = array![[1.0], [-1.0]];
        let x = array![[0.0, 1.0], [0.3, 0.6]];
        let z = augment(&x.view(), Some(&p)).unwrap();
        assert_eq!(z.dim(), (2, 3));
        assert_eq!(z[[0, 2]], -1.0);
        assert!(validate(Some(&p), 2, false).is_ok());
        assert!(augment(&array![[0.0]].view(), Some(&p)).is_err());
    }

    #[test]
    fn metric_pullback_and_null_directions() {
        let p = array![[2.0], [0.0], [-1.0]];
        let both = original_weights(&[1.0, 2.0, 3.0, 4.0], Some(&p), false);
        assert_eq!(both, vec![17.0, 2.0, 7.0]);
        let only = original_weights(&[4.0], Some(&p), true);
        assert_eq!(only[0], 16.0);
        assert_eq!(only[2], 4.0);
        assert!(crate::metric_weights::trust_region_sides(&only).iter().all(|v| v.is_finite()));
        assert_eq!(original_weights(&[1.0, 2.0], None, false), vec![1.0, 2.0]);
        assert!(validate(None, 3, true).is_err());
        assert!(validate(Some(&Array2::zeros((3, 0))), 3, true).is_err());
        assert!(validate(Some(&Array2::zeros((3, 1))), 3, true).is_err());
    }

}
