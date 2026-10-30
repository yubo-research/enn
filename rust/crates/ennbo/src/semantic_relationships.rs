//! Fixed rank-one input projections for each output, shared by all surrogates.
use crate::error::ENNError;
use ndarray::{Array2, ArrayView2};

/// Construct columns proportional to input-output cosine similarities.
/// Rows map ordered inputs followed by outputs into deduplicated embeddings.
/// The combined squared energy equals the output count, matching the relative
/// weight convention of the x-x projection. Zero embeddings contribute no prior.
pub fn projection(
    embeddings: &ArrayView2<f64>,
    rows: &[usize],
    input_count: usize,
) -> Result<Array2<f64>, ENNError> {
    if input_count == 0
        || input_count >= rows.len()
        || embeddings.is_empty()
        || rows.iter().copied().max() != embeddings.nrows().checked_sub(1)
        || embeddings.iter().any(|v| !v.is_finite())
    {
        return Err(ENNError::InvalidParameter(
            "invalid input-output embeddings or row mapping".into(),
        ));
    }
    let mut unit = embeddings.to_owned();
    crate::semantic_projection::normalize_rows(&mut unit);
    let output_count = rows.len() - input_count;
    let mut result = Array2::from_shape_fn((input_count, output_count), |(j, i)| {
        unit.row(rows[j])
            .dot(&unit.row(rows[input_count + i]))
            .clamp(-1.0, 1.0)
    });
    let scale = result.iter().copied().map(f64::abs).fold(0.0_f64, f64::max);
    if scale > 0.0 {
        result /= scale;
        let norm = result.iter().map(|v| v * v).sum::<f64>().sqrt();
        result *= (output_count as f64).sqrt() / norm;
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn cosine_columns_preserve_relevance_sign_and_order() {
        let embeddings = array![[2., 0.], [0., 3.], [-4., 0.]];
        let p = projection(&embeddings.view(), &[0, 1, 0, 2], 2).unwrap();
        assert_eq!(p, array![[1., -1.], [0., 0.]]);
        let reordered = projection(&embeddings.view(), &[1, 0, 2, 0], 2).unwrap();
        assert_eq!(reordered, array![[0., 0.], [-1., 1.]]);
        let duplicated = projection(&array![[2., 0.]].view(), &[0, 0, 0], 2).unwrap();
        assert!((duplicated[[0, 0]] - 0.5_f64.sqrt()).abs() < 1e-15);
        assert_eq!(duplicated[[0, 0]], duplicated[[1, 0]]);
    }

    #[test]
    fn zero_and_extreme_vectors_remain_finite() {
        let embeddings = array![[f64::MAX, 0.], [0., f64::MIN_POSITIVE], [0., 0.]];
        assert_eq!(
            projection(&embeddings.slice(ndarray::s![..2, ..]), &[0, 1, 0, 1], 2).unwrap(),
            array![[1., 0.], [0., 1.]]
        );
        assert_eq!(
            projection(&embeddings.view(), &[0, 1, 2], 2).unwrap(),
            Array2::<f64>::zeros((2, 1))
        );
        assert_eq!(
            projection(&array![[0., 0.]].view(), &[0, 0], 1).unwrap(),
            array![[0.]]
        );
        assert_eq!(
            projection(&array![[1., 0.], [1e-300, 1.]].view(), &[0, 1], 1).unwrap(),
            array![[1.]]
        );
    }

    #[test]
    fn malformed_embeddings_fail_before_indexing() {
        for (embeddings, rows, inputs) in [
            (array![[1., 0.]], vec![0, 0], 0),
            (array![[1., 0.]], vec![0, 0], 2),
            (array![[1., 0.]], vec![0, 1], 1),
            (array![[1., 0.]], vec![], 1),
            (array![[f64::NAN]], vec![0, 0], 1),
            (array![[f64::INFINITY]], vec![0, 0], 1),
            (Array2::zeros((1, 0)), vec![0, 0], 1),
            (Array2::zeros((0, 2)), vec![0, 0], 1),
        ] {
            assert!(projection(&embeddings.view(), &rows, inputs).is_err());
        }
    }
}
