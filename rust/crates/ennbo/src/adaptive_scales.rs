use crate::adaptive_geometry::invalid;
use crate::error::ENNError;
use ndarray::{Array1, Array2, ArrayView1};
use ndarray_linalg::Inverse;

pub(crate) const STATIONARY_FORGET: f64 = 1.0;
const SHRINK_ROWS: f64 = 12.0;
const RIDGE_RATE: f64 = 0.03;
const METADATA_MULTIPLIER: f64 = 1.0;

#[derive(Clone)]
pub(crate) struct ColumnScales {
    projection: Array2<f64>,
    gram: Array2<f64>,
    cross: Array2<f64>,
    sum_s: Array1<f64>,
    y_sum: Array1<f64>,
    y_sumsq: Array1<f64>,
    mass: f64,
    forget: f64,
}

impl ColumnScales {
    pub(crate) fn new(projection: Array2<f64>, outputs: usize, forget: f64) -> Self {
        let factors = projection.ncols();
        Self {
            projection,
            gram: Array2::zeros((factors, factors)),
            cross: Array2::zeros((factors, outputs)),
            sum_s: Array1::zeros(factors),
            y_sum: Array1::zeros(outputs),
            y_sumsq: Array1::zeros(outputs),
            mass: 0.0,
            forget,
        }
    }

    pub(crate) fn observe(&mut self, factor: ArrayView1<f64>, target: ArrayView1<f64>) {
        self.gram *= self.forget;
        self.cross *= self.forget;
        self.sum_s *= self.forget;
        self.y_sum *= self.forget;
        self.y_sumsq *= self.forget;
        self.mass *= self.forget;
        for (i, &left) in factor.iter().enumerate() {
            for (j, &right) in factor.iter().enumerate() {
                self.gram[[i, j]] += left * right;
            }
            for (k, &value) in target.iter().enumerate() {
                self.cross[[i, k]] += left * value;
            }
            self.sum_s[i] += left;
        }
        for (k, &value) in target.iter().enumerate() {
            self.y_sum[k] += value;
            self.y_sumsq[k] += value * value;
        }
        self.mass += 1.0;
    }

    pub(crate) fn observe_rows(
        &mut self,
        x: &Array2<f64>,
        y: &Array2<f64>,
        bounds: &Array2<f64>,
    ) -> Result<(), ENNError> {
        if x.ncols() != self.projection.nrows() || y.ncols() != self.cross.ncols() {
            return Err(invalid(
                "column-scale observations must match the projection and outputs",
            ));
        }
        let factors = centered(x, bounds).dot(&self.projection);
        for row in 0..x.nrows() {
            self.observe(factors.row(row), y.row(row));
        }
        Ok(())
    }

    pub(crate) fn coefficients(&self) -> Array1<f64> {
        let factors = self.gram.nrows();
        let prior = Array1::from_elem(factors, METADATA_MULTIPLIER);
        if self.mass <= 0.0 || factors == 0 {
            return prior;
        }
        let response = self.standardized_cross();
        let mut system = self.gram.clone();
        let ridge = RIDGE_RATE * self.mass;
        for index in 0..factors {
            system[[index, index]] += ridge;
        }
        let Ok(coefficients) = system.inv().map(|inverse| inverse.dot(&response)) else {
            return prior;
        };
        let energy: Vec<f64> = coefficients
            .rows()
            .into_iter()
            .map(|row| row.dot(&row))
            .collect();
        let total = energy.iter().sum::<f64>();
        if total <= 1e-24 {
            return prior;
        }
        let mean_energy = total / factors as f64;
        let denom = self.mass + SHRINK_ROWS;
        Array1::from_shape_fn(factors, |index| {
            (self.mass * energy[index] / mean_energy + SHRINK_ROWS * METADATA_MULTIPLIER) / denom
        })
    }

    fn standardized_cross(&self) -> Array2<f64> {
        let mean = &self.y_sum / self.mass;
        let mut response = self.cross.clone();
        for factor in 0..response.nrows() {
            for output in 0..response.ncols() {
                response[[factor, output]] -= self.sum_s[factor] * mean[output];
            }
        }
        for output in 0..response.ncols() {
            let variance = (self.y_sumsq[output] / self.mass - mean[output].powi(2)).max(0.0);
            let scale = variance.sqrt();
            if scale > 1e-12 {
                response
                    .column_mut(output)
                    .mapv_inplace(|value| value / scale);
            } else {
                response.column_mut(output).fill(0.0);
            }
        }
        response
    }
}

pub(crate) fn centered(x: &Array2<f64>, bounds: &Array2<f64>) -> Array2<f64> {
    Array2::from_shape_fn(x.dim(), |(row, column)| {
        (x[[row, column]] - bounds[[column, 0]]) / (bounds[[column, 1]] - bounds[[column, 0]]) - 0.5
    })
}

pub(crate) fn distance_scales(coefficients: &Array1<f64>, dimensions: usize) -> Array1<f64> {
    let mut scale = Array1::ones(dimensions + coefficients.len());
    for (index, &coefficient) in coefficients.iter().enumerate() {
        scale[dimensions + index] = 1.0 / coefficient.max(1e-12);
    }
    scale
}

pub(crate) fn bake(features: &mut Array2<f64>, coefficients: &Array1<f64>) {
    let offset = features.ncols() - coefficients.len();
    for mut row in features.rows_mut() {
        for (index, &coefficient) in coefficients.iter().enumerate() {
            row[offset + index] *= coefficient;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn one_row_keeps_metadata_multipliers_and_forgetting_is_rank_one() {
        let mut once = ColumnScales::new(Array2::zeros((1, 2)), 1, STATIONARY_FORGET);
        once.observe(array![3.0, -4.0].view(), array![9.0].view());
        assert_eq!(once.coefficients(), array![1.0, 1.0]);
        assert_eq!(once.gram.dim(), (2, 2));
        let mut discounted = ColumnScales::new(Array2::zeros((1, 1)), 1, 0.5);
        discounted.observe(array![2.0].view(), array![1.0].view());
        discounted.observe(array![2.0].view(), array![1.0].view());
        assert!((discounted.gram[[0, 0]] - 6.0).abs() < 1e-12);
        assert!((discounted.cross[[0, 0]] - 3.0).abs() < 1e-12);
        assert!((discounted.mass - 1.5).abs() < 1e-12);
    }

    #[test]
    fn predictive_factor_outranks_an_unused_factor_without_storing_rows() {
        let mut state = ColumnScales::new(Array2::zeros((1, 2)), 1, STATIONARY_FORGET);
        for index in 0..400 {
            let value = index as f64 / 20.0;
            state.observe(
                array![value.sin(), (index as f64 * 0.17).cos()].view(),
                array![value.sin()].view(),
            );
        }
        let coefficients = state.coefficients();
        assert!(coefficients[0] > coefficients[1]);
        assert!(coefficients[1] < 0.1);
        assert_eq!(state.gram.nrows(), 2);
    }
}
