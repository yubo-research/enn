use crate::error::ENNError;
use ndarray::{s, Array2, ArrayView2, Axis};

pub fn validate(
    projection: Option<&Array2<f64>>,
    dimensions: usize,
    semantic_only: bool,
) -> Result<(), ENNError> {
    if let Some(p) = projection {
        if p.nrows() != dimensions || p.iter().any(|v| !v.is_finite()) {
            return Err(ENNError::InvalidParameter(
                "invalid semantic projection".into(),
            ));
        }
        if semantic_only && (p.ncols() == 0 || p.iter().all(|v| *v == 0.0)) {
            return Err(ENNError::InvalidParameter(
                "semantic-only features require nonzero rank".into(),
            ));
        }
    } else if semantic_only {
        return Err(ENNError::InvalidParameter(
            "semantic_only requires a projection".into(),
        ));
    }
    Ok(())
}

pub fn augment(
    x: &ArrayView2<f64>,
    projection: Option<&Array2<f64>>,
) -> Result<Array2<f64>, ENNError> {
    let Some(p) = projection else {
        return Ok(x.to_owned());
    };
    if x.ncols() != p.nrows() {
        return Err(ENNError::InvalidParameter(
            "semantic input dimension mismatch".into(),
        ));
    }
    let centered = x - 0.5;
    ndarray::concatenate(Axis(1), &[centered.view(), centered.dot(p).view()])
        .map_err(|e| ENNError::InvalidParameter(e.to_string()))
}

pub fn features(
    x: &ArrayView2<f64>,
    projection: Option<&Array2<f64>>,
    semantic_only: bool,
) -> Result<Array2<f64>, ENNError> {
    let augmented = augment(x, projection)?;
    if semantic_only {
        let dimensions = projection
            .ok_or_else(|| ENNError::InvalidParameter("missing projection".into()))?
            .nrows();
        return Ok(augmented.slice(s![.., dimensions..]).to_owned());
    }
    Ok(augmented)
}

pub fn original_weights(
    weights: &[f64],
    projection: Option<&Array2<f64>>,
    semantic_only: bool,
) -> Vec<f64> {
    let Some(p) = projection else {
        return weights.to_vec();
    };
    let offset = if semantic_only { 0 } else { p.nrows() };
    let floor = weights.iter().copied().fold(0.0_f64, f64::max).max(1.0) * 1e-12;
    (0..p.nrows())
        .map(|j| {
            let numerical = if semantic_only { 0.0 } else { weights[j] };
            (numerical
                + (0..p.ncols())
                    .map(|k| p[[j, k]].powi(2) * weights[offset + k])
                    .sum::<f64>())
            .max(floor)
        })
        .collect()
}

pub fn feature_count(
    projection: Option<&Array2<f64>>,
    dimensions: usize,
    semantic_only: bool,
) -> usize {
    match projection {
        Some(p) if semantic_only => p.ncols(),
        Some(p) => dimensions + p.ncols(),
        None => dimensions,
    }
}

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

fn fnv1a(hash: u64, bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .fold(hash, |h, b| (h ^ u64::from(*b)).wrapping_mul(FNV_PRIME))
}

fn fnv1a_matrix(hash: u64, matrix: &ArrayView2<f64>) -> u64 {
    let shape = fnv1a(
        fnv1a(hash, &matrix.nrows().to_le_bytes()),
        &matrix.ncols().to_le_bytes(),
    );
    matrix
        .iter()
        .fold(shape, |h, v| fnv1a(h, &v.to_bits().to_le_bytes()))
}

pub fn input_fingerprint(
    names: &[String],
    bounds: &ArrayView2<f64>,
    weighted_projection: &ArrayView2<f64>,
    semantic_only: bool,
) -> String {
    let mut hash = fnv1a(FNV_OFFSET, &[u8::from(semantic_only)]);
    for name in names {
        hash = fnv1a(hash, &name.len().to_le_bytes());
        hash = fnv1a(hash, name.as_bytes());
    }
    format!(
        "{:016x}",
        fnv1a_matrix(fnv1a_matrix(hash, bounds), weighted_projection)
    )
}

pub fn output_fingerprint(loadings: &ArrayView2<f64>) -> String {
    format!("{:016x}", fnv1a_matrix(FNV_OFFSET, loadings))
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
    fn input_fingerprint_is_stable_and_sees_every_part_of_the_transform() {
        let names = vec!["a".to_string(), "b".to_string()];
        let bounds = array![[0.0, 1.0], [0.0, 2.0]];
        let projection = array![[1.0], [-1.0]];
        let base = input_fingerprint(&names, &bounds.view(), &projection.view(), false);
        assert_eq!(base, "6f798e8a3bb776b0");
        let renamed = vec!["ab".to_string(), String::new()];
        let wider = array![[0.0, 1.0], [0.0, 3.0]];
        let scaled = array![[2.0], [-1.0]];
        for other in [
            input_fingerprint(&names, &bounds.view(), &projection.view(), true),
            input_fingerprint(&renamed, &bounds.view(), &projection.view(), false),
            input_fingerprint(&names, &wider.view(), &projection.view(), false),
            input_fingerprint(&names, &bounds.view(), &scaled.view(), false),
        ] {
            assert_ne!(base, other);
        }
    }

    #[test]
    fn metric_pullback_and_null_directions() {
        let p = array![[2.0], [0.0], [-1.0]];
        let both = original_weights(&[1.0, 2.0, 3.0, 4.0], Some(&p), false);
        assert_eq!(both, vec![17.0, 2.0, 7.0]);
        let only = original_weights(&[4.0], Some(&p), true);
        assert_eq!(only[0], 16.0);
        assert_eq!(only[2], 4.0);
        assert!(crate::metric_weights::trust_region_sides(&only)
            .iter()
            .all(|v| v.is_finite()));
        assert_eq!(original_weights(&[1.0, 2.0], None, false), vec![1.0, 2.0]);
        assert!(validate(None, 3, true).is_err());
        assert!(validate(Some(&Array2::zeros((3, 0))), 3, true).is_err());
        assert!(validate(Some(&Array2::zeros((3, 1))), 3, true).is_err());
    }
}
