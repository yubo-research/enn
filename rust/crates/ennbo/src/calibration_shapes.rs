use crate::calibration::{map_draws_with, map_mu_with};
use crate::error::ENNError;
use ndarray::{Array2, ArrayD, ArrayViewD, Axis, IxDyn};

fn metric_rows(values: &ArrayViewD<f64>) -> Result<Array2<f64>, ENNError> {
    let columns = values.shape().last().copied().unwrap_or(1).max(1);
    let columns = if values.ndim() == 1 { 1 } else { columns };
    Ok(Array2::from_shape_vec(
        (values.len() / columns, columns),
        values.iter().copied().collect(),
    )?)
}

fn aligned_rows(values: &ArrayViewD<f64>, shape: &[usize]) -> Result<Array2<f64>, ENNError> {
    let mut view = values.view();
    while view.ndim() < shape.len() {
        let axis = view.ndim().saturating_sub(1);
        view = view.insert_axis(Axis(axis));
    }
    let broadcast = view.broadcast(IxDyn(shape)).ok_or_else(|| {
        ENNError::ShapeError(format!(
            "cannot broadcast {:?} to {shape:?}",
            values.shape()
        ))
    })?;
    metric_rows(&broadcast)
}

fn restored(rows: Array2<f64>, shape: &[usize]) -> Result<ArrayD<f64>, ENNError> {
    Ok(rows.into_shape_with_order(IxDyn(shape))?)
}

pub fn map_mu_any_rank(
    a: &[f64],
    b: &[f64],
    mu: &ArrayViewD<f64>,
    y_bounds: Option<&Array2<f64>>,
    se: Option<&ArrayViewD<f64>>,
) -> Result<ArrayD<f64>, ENNError> {
    let rows = metric_rows(mu)?;
    let se_rows = se.map(|v| aligned_rows(v, mu.shape())).transpose()?;
    restored(
        map_mu_with(a, b, rows.view(), y_bounds, se_rows.as_ref())?,
        mu.shape(),
    )
}

pub struct AffineCoefficients<'a> {
    pub a: &'a [f64],
    pub b: &'a [f64],
    pub c: &'a [f64],
}

pub fn map_draws_any_rank(
    coefficients: &AffineCoefficients<'_>,
    draws: &ArrayViewD<f64>,
    mu: &ArrayViewD<f64>,
    y_bounds: Option<&Array2<f64>>,
    se: Option<&ArrayViewD<f64>>,
) -> Result<ArrayD<f64>, ENNError> {
    let rows = metric_rows(draws)?;
    let mu_rows = aligned_rows(mu, draws.shape())?;
    let se_rows = se.map(|v| aligned_rows(v, draws.shape())).transpose()?;
    restored(
        map_draws_with(
            coefficients.a,
            coefficients.b,
            coefficients.c,
            rows.view(),
            mu_rows.view(),
            y_bounds,
            se_rows.as_ref(),
        )?,
        draws.shape(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array3};

    #[test]
    fn metric_rows_flatten_all_but_the_metric_axis() {
        let vector = array![1.0, 2.0].into_dyn();
        assert_eq!(metric_rows(&vector.view()).unwrap().dim(), (2, 1));
        let cube = Array3::<f64>::zeros((2, 3, 4)).into_dyn();
        assert_eq!(metric_rows(&cube.view()).unwrap().dim(), (6, 4));
    }

    #[test]
    fn aligned_rows_insert_the_draw_axis_before_metrics() {
        let mu = array![[1.0], [2.0]].into_dyn();
        let rows = aligned_rows(&mu.view(), &[2, 3, 1]).unwrap();
        assert_eq!(rows.dim(), (6, 1));
        assert_eq!(rows[[3, 0]], 2.0);
        assert!(aligned_rows(&mu.view(), &[3, 2, 1]).is_err());
    }

    #[test]
    fn map_mu_preserves_rank() {
        let mu = array![1.0, 2.0].into_dyn();
        let out = map_mu_any_rank(&[1.0], &[2.0], &mu.view(), None, None).unwrap();
        assert_eq!(out.shape(), &[2]);
        assert_eq!(out[[1]], 5.0);
    }

    #[test]
    fn map_draws_broadcasts_mean_and_preserves_rank() {
        let draws = Array3::<f64>::from_elem((2, 3, 1), 3.0).into_dyn();
        let mu = array![[1.0], [1.0]].into_dyn();
        let coefficients = AffineCoefficients {
            a: &[0.0],
            b: &[1.0],
            c: &[1.0],
        };
        let out = map_draws_any_rank(&coefficients, &draws.view(), &mu.view(), None, None).unwrap();
        assert_eq!(out.shape(), &[2, 3, 1]);
        assert_eq!(out[[1, 1, 0]], 3.0);
    }
}
