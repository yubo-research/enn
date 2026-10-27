//! Fixed semantic features at the surrogate boundary. Optimizer coordinates stay unit-scaled.
use crate::{error::ENNError, layout::EnnLayout};
use ndarray::{s, Array1, Array2, ArrayView2, Axis};

/// Check dimensions and prohibit observation-dependent reweighting.
pub fn validate(projection: Option<&Array2<f64>>, dimensions: usize, layout: &EnnLayout) -> Result<(), ENNError> {
    if let Some(p) = projection {
        if p.nrows() != dimensions || p.iter().any(|v| !v.is_finite()) {
            return Err(ENNError::InvalidParameter("invalid semantic projection".into()));
        }
        if layout.scale_x() || layout.metric_learning() != crate::metric_auto::MetricLearning::None {
            return Err(ENNError::InvalidParameter("semantic inputs require fixed scaling".into()));
        }
    }
    Ok(())
}

/// Append semantic coordinates, centering numerical unit inputs using fixed bounds.
pub fn augment(x: &ArrayView2<f64>, projection: Option<&Array2<f64>>) -> Result<Array2<f64>, ENNError> {
    let Some(p) = projection else { return Ok(x.to_owned()); };
    if x.ncols() != p.nrows() {
        return Err(ENNError::InvalidParameter("semantic input dimension mismatch".into()));
    }
    let centered = x - 0.5;
    ndarray::concatenate(Axis(1), &[centered.view(), centered.dot(p).view()])
        .map_err(|e| ENNError::InvalidParameter(e.to_string()))
}

/// Recover optimizer coordinates for observation access.
pub fn original_row(x: Array1<f64>, projection: Option<&Array2<f64>>) -> Array1<f64> {
    match projection {
        Some(p) => x.slice(s![..p.nrows()]).to_owned() + 0.5,
        None => x,
    }
}

/// Recover optimizer rows without exposing appended features to candidate generation.
pub fn original_rows(x: Array2<f64>, projection: Option<&Array2<f64>>) -> Array2<f64> {
    match projection {
        Some(p) => x.slice(s![.., ..p.nrows()]).to_owned() + 0.5,
        None => x,
    }
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
        assert_eq!(original_rows(z.clone(), Some(&p)), x);
        assert_eq!(original_row(z.row(0).to_owned(), Some(&p)), x.row(0));
        assert!(validate(Some(&p), 2, &EnnLayout::memory(crate::index::IndexDriver::Flat, false)).is_ok());
        assert!(augment(&array![[0.0]].view(), Some(&p)).is_err());
    }
}
