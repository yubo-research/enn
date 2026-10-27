//! Deterministic categorical distances shared by Rust and Python callers.
use crate::error::ENNError;
use ndarray::{Array2, ArrayView2};

fn hash(category: i64, variable: usize, seed: u64) -> u64 {
    let mut x = category as u64 ^ seed ^ (variable as u64).wrapping_mul(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

/// Validate a code width before shifting category hashes.
pub fn width(value: usize, upper: usize) -> Result<usize, ENNError> {
    if value == 0 || value > upper {
        return Err(ENNError::InvalidParameter(format!(
            "width must be in 1..{upper}"
        )));
    }
    Ok(value)
}

/// Choose enough signs to limit collisions, subject to a caller-provided cap.
pub fn log_code_width(card: usize, cap: usize) -> Result<usize, ENNError> {
    width(cap, 64).map_err(|_| ENNError::InvalidParameter("dim_cap must be in 1..64".into()))?;
    if card == 0 {
        return Err(ENNError::InvalidParameter("card must be >= 1".into()));
    }
    Ok(if card == 1 {
        1
    } else {
        ((2.0 * (card as f64).log2() + 7.0).ceil() as usize).min(cap)
    })
}

/// Encode each label as a fixed-length sign vector with norm `separation`.
pub fn rademacher(
    categories: &[i64],
    variable: usize,
    dimensions: usize,
    seed: u64,
    separation: f64,
) -> Result<Array2<f64>, ENNError> {
    width(dimensions, 64)?;
    if !separation.is_finite() || separation <= 0.0 {
        return Err(ENNError::InvalidParameter(
            "separation must be positive and finite".into(),
        ));
    }
    let scale = separation / (dimensions as f64).sqrt();
    Ok(Array2::from_shape_fn(
        (categories.len(), dimensions),
        |(i, j)| {
            if (hash(categories[i], variable, seed) >> j) & 1 == 1 {
                scale
            } else {
                -scale
            }
        },
    ))
}

/// Encode all categorical variables in a shared normalized sparse hash vector.
pub fn hash_columns(
    categories: &ArrayView2<i64>,
    dimensions: usize,
    seed: u64,
) -> Result<Array2<f64>, ENNError> {
    width(dimensions, usize::MAX)?;
    let mut out = Array2::zeros((categories.nrows(), dimensions));
    let scale = 1.0 / (categories.ncols().max(1) as f64).sqrt();
    for ((i, j), &category) in categories.indexed_iter() {
        let h = hash(category, j, seed);
        out[[i, (h % dimensions as u64) as usize]] += if h >> 63 == 1 { scale } else { -scale };
    }
    Ok(out)
}

/// Expand category ids, rejecting labels outside the declared domain.
pub fn one_hot(categories: &[i64], card: usize) -> Result<Array2<f64>, ENNError> {
    if card == 0 || categories.iter().any(|&v| v < 0 || v as usize >= card) {
        return Err(ENNError::InvalidParameter(
            "one-hot ids must be in 0..card-1".into(),
        ));
    }
    let mut out = Array2::zeros((categories.len(), card));
    for (i, &id) in categories.iter().enumerate() {
        out[[i, id as usize]] = 1.0;
    }
    Ok(out)
}

/// Recover the most similar allowed category. Domain order resolves exact ties.
pub fn snap(
    block: &ArrayView2<f64>,
    domain: &[i64],
    variable: usize,
    seed: u64,
    separation: f64,
    onehot: bool,
) -> Result<Vec<i64>, ENNError> {
    if domain.is_empty() {
        return Err(ENNError::InvalidParameter(
            "domain must be non-empty".into(),
        ));
    }
    let table = if onehot {
        one_hot(domain, block.ncols())?
    } else {
        rademacher(domain, variable, block.ncols(), seed, separation)?
    };
    Ok(block
        .rows()
        .into_iter()
        .map(|row| {
            let mut best = 0;
            let mut score = f64::NEG_INFINITY;
            for (j, code) in table.rows().into_iter().enumerate() {
                let candidate = row.dot(&code);
                if candidate > score {
                    best = j;
                    score = candidate;
                }
            }
            domain[best]
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    #[test]
    fn categorical_codes_roundtrip_and_validate() {
        let domain = [-4, 2, 7];
        let code = rademacher(&domain, 2, 48, 0, 10.0).unwrap();
        assert_eq!(
            snap(&code.view(), &domain, 2, 0, 10.0, false).unwrap(),
            domain
        );
        let one = one_hot(&[0, 2, 1], 3).unwrap();
        assert_eq!(
            snap(&one.view(), &[0, 1, 2], 0, 0, 1.0, true).unwrap(),
            [0, 2, 1]
        );
        assert_eq!(log_code_width(1, 48).unwrap(), 1);
        assert_eq!(log_code_width(100, 48).unwrap(), 21);
        assert!(one_hot(&[-1], 3).is_err());
        assert!(rademacher(&domain, 0, 65, 0, 1.0).is_err());
        let x = array![[1, 2], [1, 2]];
        let h = hash_columns(&x.view(), 8, 3).unwrap();
        assert_eq!(h.row(0), h.row(1));
    }
}
