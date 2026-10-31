use crate::error::ENNError;
use ndarray::{Array2, ArrayView2};

pub fn projection(
    embeddings: &ArrayView2<f64>,
    rows: &[usize],
    input_count: usize,
    cutoff: f64,
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
    let result = Array2::from_shape_fn((input_count, output_count), |(j, i)| {
        unit.row(rows[j])
            .dot(&unit.row(rows[input_count + i]))
            .clamp(-1.0, 1.0)
    });
    crate::semantic_projection::centered_projection(result, cutoff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn centered_cosines_preserve_signed_geometry() {
        let embeddings = array![[2., 0.], [0., 3.], [-4., 0.]];
        let p = projection(&embeddings.view(), &[0, 1, 0, 2], 2, 1.0).unwrap();
        assert_eq!(p.dim(), (2, 1));
        assert!((p[[0, 0]] + p[[1, 0]]).abs() < 1e-12);
        assert!((p.iter().map(|v| v * v).sum::<f64>() - 1.0).abs() < 1e-12);
        let same = projection(&array![[2., 0.]].view(), &[0, 0, 0], 2, 0.9).unwrap();
        assert_eq!(same.dim(), (2, 0));
        let zero = projection(&Array2::zeros((2, 2)).view(), &[0, 1], 1, 1.0).unwrap();
        assert_eq!(zero.dim(), (1, 0));
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
            assert!(projection(&embeddings.view(), &rows, inputs, 0.9).is_err());
        }
    }
}
