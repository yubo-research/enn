//! Error types for ENN operations.

use thiserror::Error;

use crate::index::IndexError;

/// Errors that can occur in ENN operations.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum ENNError {
    /// Invalid input shape.
    #[error("Invalid shape: expected {expected:?}, got {got:?}")]
    InvalidShape { expected: Vec<usize>, got: Vec<usize> },
    /// Invalid parameter.
    #[error("Invalid parameter: {0}")]
    InvalidParameter(String),
    /// Index error.
    #[error("Index error: {0}")]
    IndexError(#[from] IndexError),
    /// Not enough observations.
    #[error("Not enough observations: got {0}, need at least {1}")]
    NotEnoughObservations(usize, usize),
    /// Shape error from ndarray.
    #[error("Shape error: {0}")]
    ShapeError(String),
}

impl From<ndarray::ShapeError> for ENNError {
    fn from(e: ndarray::ShapeError) -> Self {
        ENNError::ShapeError(e.to_string())
    }
}

/// Small constant to avoid division by zero.
pub const EPS_VAR: f64 = 1e-9;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::IndexError;

    #[test]
    fn test_enn_error_display() {
        let err = ENNError::InvalidShape {
            expected: vec![10, 5],
            got: vec![8, 5],
        };
        assert!(err.to_string().contains("Invalid shape"));

        let err = ENNError::InvalidParameter("bad param".to_string());
        assert!(err.to_string().contains("Invalid parameter"));

        let err = ENNError::NotEnoughObservations(5, 10);
        assert!(err.to_string().contains("Not enough observations"));

        let err = ENNError::ShapeError("bad shape".to_string());
        assert!(err.to_string().contains("Shape error"));
    }

    #[test]
    fn test_enn_error_from_index_error() {
        let index_err = IndexError::InvalidShape {
            expected: 5,
            got: 3,
        };
        let enn_err: ENNError = index_err.into();
        assert!(matches!(enn_err, ENNError::IndexError(_)));
    }

    #[test]
    fn test_enn_error_from_shape_error() {
        let shape_err = ndarray::Array2::from_shape_vec((2, 2), vec![1.0]).expect_err("shape err");
        let enn_err: ENNError = shape_err.into();
        assert!(matches!(enn_err, ENNError::ShapeError(_)));
    }

    #[test]
    fn test_eps_var_constant() {
        assert_eq!(EPS_VAR, 1e-9);
    }
}
