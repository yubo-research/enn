use crate::error::{invalid, ENNError};
use crate::semantic_projection::{normalize_rows, validate_bounds};
use ndarray::{Array2, ArrayView2, ArrayViewD, Axis, Ix1, Ix2};
use std::collections::HashSet;

const ORTHONORMAL_RELATIVE_TOLERANCE: f64 = 1e-5;
const ORTHONORMAL_ABSOLUTE_TOLERANCE: f64 = 1e-8;

fn names_are_valid(names: &[String]) -> bool {
    let unique: HashSet<&String> = names.iter().collect();
    !names.is_empty() && unique.len() == names.len() && names.iter().all(|n| !n.trim().is_empty())
}

pub fn validate_output_names(names: &[String]) -> Result<(), ENNError> {
    if names.len() < 2 || !names_are_valid(names) {
        return Err(invalid(
            "at least two unique nonempty output names are required",
        ));
    }
    Ok(())
}

pub fn validate_relationship_outputs(names: &[String]) -> Result<(), ENNError> {
    if !names_are_valid(names) {
        return Err(invalid("output names must be nonempty and unique"));
    }
    Ok(())
}

pub fn validate_cutoff(cutoff: f64) -> Result<(), ENNError> {
    if !cutoff.is_finite() || cutoff <= 0.0 || cutoff > 1.0 {
        return Err(invalid("variance_cutoff must be finite and in (0, 1]"));
    }
    Ok(())
}

pub fn validate_input_definition(
    names: &[String],
    bounds: &ArrayViewD<f64>,
    cutoff: f64,
    relative_weight: f64,
) -> Result<(), ENNError> {
    if !names_are_valid(names) {
        return Err(invalid("variable names must be nonempty and unique"));
    }
    let bounds = bounds
        .view()
        .into_dimensionality::<Ix2>()
        .map_err(|_| invalid("bounds must be finite increasing pairs, one per variable"))?;
    validate_bounds(&bounds, names.len())?;
    validate_cutoff(cutoff)?;
    if !relative_weight.is_finite() || relative_weight < 0.0 {
        return Err(invalid("relative_weight must be finite and nonnegative"));
    }
    Ok(())
}

pub fn validate_cached_projection(
    projection: &ArrayViewD<f64>,
    variables: usize,
) -> Result<(), ENNError> {
    if projection.ndim() != 2
        || projection.shape()[0] != variables
        || projection.iter().any(|v| !v.is_finite())
    {
        return Err(invalid("invalid cached projection"));
    }
    Ok(())
}

pub fn normalized_embeddings(
    embeddings: &ArrayViewD<f64>,
    outputs: usize,
) -> Result<Array2<f64>, ENNError> {
    let message = "embeddings must be finite nonempty rows, one per output";
    let values = embeddings
        .view()
        .into_dimensionality::<Ix2>()
        .map_err(|_| invalid(message))?;
    if values.nrows() != outputs || values.ncols() == 0 || values.iter().any(|v| !v.is_finite()) {
        return Err(invalid(message));
    }
    let mut unit = values.to_owned();
    normalize_rows(&mut unit);
    Ok(unit)
}

fn has_orthonormal_columns(loadings: &ArrayView2<f64>) -> bool {
    let gram = loadings.t().dot(loadings);
    gram.indexed_iter().all(|((i, j), value)| {
        let target = if i == j { 1.0 } else { 0.0 };
        (value - target).abs()
            <= ORTHONORMAL_ABSOLUTE_TOLERANCE + ORTHONORMAL_RELATIVE_TOLERANCE * target
    })
}

pub fn checked_loadings(
    loadings: &ArrayViewD<f64>,
    outputs: usize,
) -> Result<Array2<f64>, ENNError> {
    let matrix = loadings
        .view()
        .into_dimensionality::<Ix2>()
        .map_err(|_| invalid("invalid output loadings"))?;
    if matrix.nrows() != outputs
        || matrix.ncols() >= outputs
        || matrix.iter().any(|v| !v.is_finite())
    {
        return Err(invalid("invalid output loadings"));
    }
    if !has_orthonormal_columns(&matrix) {
        return Err(invalid("output loadings must have orthonormal columns"));
    }
    Ok(matrix.to_owned())
}

pub fn embedding_matrix<'a>(
    encoded: &'a ArrayViewD<'_, f64>,
) -> Result<ArrayView2<'a, f64>, ENNError> {
    encoded
        .view()
        .into_dimensionality::<Ix2>()
        .map_err(|_| invalid("encoder must return a two-dimensional embedding matrix"))
}

pub fn output_embeddings(
    encoded: &ArrayViewD<f64>,
    texts: usize,
    rows: &[usize],
) -> Result<Array2<f64>, ENNError> {
    let matrix = embedding_matrix(encoded)?;
    if matrix.nrows() != texts {
        return Err(invalid("encoder must return one embedding row per text"));
    }
    Ok(matrix.select(Axis(0), rows))
}

fn column_order(names: &[String], given: &[String]) -> Result<Vec<usize>, ENNError> {
    let mismatch = || invalid("input names must match the problem definition");
    if names.len() != given.len() {
        return Err(mismatch());
    }
    names
        .iter()
        .map(|name| given.iter().position(|g| g == name).ok_or_else(mismatch))
        .collect()
}

pub fn ordered_rows(
    x: &ArrayViewD<f64>,
    names: &[String],
    given: Option<&[String]>,
) -> Result<(Array2<f64>, bool), ENNError> {
    let message = "inputs must be finite rows with one column per variable";
    let single = x.ndim() == 1;
    let rows = if single {
        x.view()
            .into_dimensionality::<Ix1>()
            .map_err(invalid)?
            .insert_axis(Axis(0))
            .to_owned()
    } else {
        x.view()
            .into_dimensionality::<Ix2>()
            .map_err(|_| invalid(message))?
            .to_owned()
    };
    if rows.ncols() != names.len() || rows.iter().any(|v| !v.is_finite()) {
        return Err(invalid(message));
    }
    match given {
        Some(given) => Ok((rows.select(Axis(1), &column_order(names, given)?), single)),
        None => Ok((rows, single)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1};

    fn names(values: &[&str]) -> Vec<String> {
        values.iter().map(|v| (*v).to_owned()).collect()
    }

    #[test]
    fn output_names_require_two_unique_nonblank_entries() {
        assert!(validate_output_names(&names(&["a", "b"])).is_ok());
        for bad in [vec!["a"], vec!["a", "a"], vec!["a", " "]] {
            assert!(validate_output_names(&names(&bad)).is_err());
        }
        assert!(validate_relationship_outputs(&names(&["a"])).is_ok());
        assert!(validate_relationship_outputs(&[]).is_err());
    }

    #[test]
    fn cutoff_boundaries() {
        assert!(validate_cutoff(1.0).is_ok());
        for bad in [0.0, -1.0, 1.5, f64::NAN, f64::INFINITY] {
            assert!(validate_cutoff(bad).is_err());
        }
    }

    #[test]
    fn input_definition_rejects_each_invalid_field() {
        let good = names(&["a"]);
        let bounds = array![[0.0, 1.0]].into_dyn();
        let view = bounds.view();
        assert!(validate_input_definition(&good, &view, 0.9, 0.25).is_ok());
        assert!(validate_input_definition(&names(&["a", "a"]), &view, 0.9, 0.25).is_err());
        assert!(validate_input_definition(&good, &view, 0.0, 0.25).is_err());
        assert!(validate_input_definition(&good, &view, 0.9, -1.0).is_err());
        let flat = array![0.0, 1.0].into_dyn();
        assert!(validate_input_definition(&good, &flat.view(), 0.9, 0.25).is_err());
    }

    #[test]
    fn cached_projection_must_match_variables_and_be_finite() {
        let projection = array![[1.0], [0.0]].into_dyn();
        assert!(validate_cached_projection(&projection.view(), 2).is_ok());
        assert!(validate_cached_projection(&projection.view(), 3).is_err());
        let bad = array![[f64::NAN]].into_dyn();
        assert!(validate_cached_projection(&bad.view(), 1).is_err());
    }

    #[test]
    fn embeddings_normalize_rows_and_keep_zero_rows() {
        let raw = array![[3.0, 4.0], [0.0, 0.0]].into_dyn();
        let unit = normalized_embeddings(&raw.view(), 2).unwrap();
        assert!((unit[[0, 0]] - 0.6).abs() < 1e-12);
        assert_eq!(unit.row(1), Array1::<f64>::zeros(2));
        assert!(normalized_embeddings(&raw.view(), 3).is_err());
        let flat = array![1.0, 2.0].into_dyn();
        assert!(normalized_embeddings(&flat.view(), 2).is_err());
    }

    #[test]
    fn loadings_must_be_orthonormal_and_narrower_than_outputs() {
        let good = array![[1.0], [0.0]].into_dyn();
        assert!(checked_loadings(&good.view(), 2).is_ok());
        let scaled = array![[2.0], [0.0]].into_dyn();
        assert!(checked_loadings(&scaled.view(), 2).is_err());
        let square = array![[1.0, 0.0], [0.0, 1.0]].into_dyn();
        assert!(checked_loadings(&square.view(), 2).is_err());
    }

    #[test]
    fn encoder_output_shape_is_checked_before_row_selection() {
        let encoded = array![[1.0], [2.0]].into_dyn();
        let view = encoded.view();
        assert_eq!(embedding_matrix(&view).unwrap().dim(), (2, 1));
        let flat = array![1.0].into_dyn();
        assert!(embedding_matrix(&flat.view()).is_err());
        assert_eq!(
            output_embeddings(&view, 2, &[1, 1, 0]).unwrap()[[0, 0]],
            2.0
        );
        assert!(output_embeddings(&view, 3, &[0]).is_err());
    }

    #[test]
    fn ordered_rows_reorders_named_columns() {
        let order = names(&["a", "b"]);
        let given = names(&["b", "a"]);
        let row = array![1.0, 2.0].into_dyn();
        let (rows, single) = ordered_rows(&row.view(), &order, Some(&given)).unwrap();
        assert!(single);
        assert_eq!(rows, array![[2.0, 1.0]]);
        let matrix = array![[1.0, 2.0]].into_dyn();
        assert!(!ordered_rows(&matrix.view(), &order, None).unwrap().1);
        assert!(ordered_rows(&matrix.view(), &order, Some(&names(&["a", "a"]))).is_err());
        assert!(ordered_rows(&matrix.view(), &names(&["a"]), None).is_err());
        assert_eq!(column_order(&order, &given).unwrap(), vec![1, 0]);
    }
}
