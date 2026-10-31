//! Supervised linear and quadratic geometry for bounded replay epochs.
use crate::error::ENNError;
use ndarray::{s, Array2, Axis};
use ndarray_linalg::{Eigh, Inverse, SVD, UPLO};

pub(crate) fn invalid(error: impl std::fmt::Display) -> ENNError {
    ENNError::InvalidParameter(error.to_string())
}

pub(crate) fn ridge(z: &Array2<f64>, target: &Array2<f64>) -> Result<Array2<f64>, ENNError> {
    let gram = z.t().dot(z) + Array2::<f64>::eye(z.ncols()) * (z.nrows() as f64 * 0.03);
    Ok(gram.inv().map_err(invalid)?.dot(&z.t().dot(target)))
}

fn normalize(mut direction: Array2<f64>, dimensions: usize) -> Array2<f64> {
    let norm = direction
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt()
        .max(1e-12);
    direction *= (dimensions as f64).sqrt() / norm;
    direction
}

pub(crate) fn fixed_rank(
    direction: Array2<f64>,
    rank: Option<usize>,
) -> Result<Array2<f64>, ENNError> {
    let dimensions = direction.nrows();
    let direction = normalize(direction, dimensions);
    let Some(rank) = rank else {
        return Ok(direction);
    };
    let (u, singular, _) = direction.svd(true, false).map_err(invalid)?;
    let count = rank.min(singular.len());
    let mut retained = Array2::zeros((dimensions, rank));
    retained
        .slice_mut(s![.., ..count])
        .assign(&(u.unwrap().slice(s![.., ..count]).to_owned() * singular.slice(s![..count])));
    Ok(normalize(retained, dimensions))
}

pub(crate) fn quadratic(z: &Array2<f64>, target: &Array2<f64>) -> Result<Array2<f64>, ENNError> {
    let d = z.ncols();
    let design = ndarray::concatenate(Axis(1), &[z.view(), z.mapv(|v| v * v - 1.0).view()])
        .map_err(invalid)?;
    let coefficients = ridge(&design, target)?;
    let mut moment = Array2::zeros((d, d));
    for row in z.rows() {
        let gradient = Array2::from_shape_fn((d, target.ncols()), |(j, m)| {
            coefficients[[j, m]] + 2.0 * row[j] * coefficients[[d + j, m]]
        });
        moment += &gradient.dot(&gradient.t());
    }
    root(&(moment / z.nrows() as f64))
}

pub(crate) fn root(gram: &Array2<f64>) -> Result<Array2<f64>, ENNError> {
    let (values, vectors) = gram.eigh(UPLO::Lower).map_err(invalid)?;
    Ok((&vectors * &values.mapv(|v| v.max(0.0).sqrt())).dot(&vectors.t()))
}
