use ndarray::{Array1, Array2, ArrayViewD};

use super::EpistemicNearestNeighbors;
use crate::error::ENNError;
use crate::output_blend::{OutputBlend, SemanticOutputs};
use crate::params::ENNParams;

pub struct OutputMoments {
    center: Array1<f64>,
    scale: Array1<f64>,
    constant: Vec<bool>,
}

impl OutputMoments {
    fn from_sums(n: usize, sum: &Array1<f64>, sumsq: &Array1<f64>) -> Self {
        let nf = n.max(1) as f64;
        let center = sum / nf;
        let variance = (sumsq / nf - &center * &center).mapv(|v| v.max(0.0));
        let tolerance = (&center * &center) * (nf * f64::EPSILON);
        let constant: Vec<bool> = variance
            .iter()
            .zip(tolerance.iter())
            .map(|(v, t)| n >= 2 && v <= t)
            .collect();
        let scale = variance
            .iter()
            .zip(&constant)
            .map(|(v, c)| if *c || *v <= 0.0 { 1.0 } else { v.sqrt() })
            .collect();
        Self {
            center,
            scale,
            constant,
        }
    }

    pub(crate) fn standardized(&self, rows: &Array2<f64>) -> Array2<f64> {
        let mut z = (rows - &self.center) / &self.scale;
        for (j, constant) in self.constant.iter().enumerate() {
            if *constant {
                z.column_mut(j).fill(0.0);
            }
        }
        z
    }

    pub(crate) fn into_blend(
        self,
        alpha: Array1<f64>,
        projector: Array2<f64>,
    ) -> Result<OutputBlend, ENNError> {
        let alpha = alpha
            .iter()
            .zip(&self.constant)
            .map(|(a, c)| if *c { 0.0 } else { a.clamp(0.0, 1.0) })
            .collect();
        OutputBlend::new(alpha, projector, self.center, self.scale)
    }
}

impl EpistemicNearestNeighbors {
    pub fn set_semantic_outputs(
        &mut self,
        loadings: ArrayViewD<'_, f64>,
        alpha: Option<f64>,
    ) -> Result<(), ENNError> {
        let outputs = SemanticOutputs::new(loadings, alpha, self.num_metrics)?;
        self.semantic_outputs = (outputs.loadings.ncols() > 0).then_some(outputs);
        Ok(())
    }

    pub fn semantic_outputs(&self) -> Option<&SemanticOutputs> {
        self.semantic_outputs.as_ref()
    }

    pub(crate) fn output_moments(&self) -> OutputMoments {
        OutputMoments::from_sums(self.num_obs, &self.y_sum, &self.y_sumsq)
    }

    pub(crate) fn warped_y_sums(&self) -> (&Array1<f64>, &Array1<f64>) {
        (&self.y_sum, &self.y_sumsq)
    }

    pub(crate) fn checked_blend<'p>(
        &self,
        params: &'p ENNParams,
    ) -> Result<Option<&'p OutputBlend>, ENNError> {
        let blend = params.output_blend.as_deref();
        if let Some(b) = blend {
            if b.num_outputs() != self.num_metrics {
                return Err(ENNError::InvalidShape {
                    expected: vec![self.num_metrics],
                    got: vec![b.num_outputs()],
                });
            }
        }
        Ok(blend)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn moments_flag_a_constant_output_despite_rounding() {
        let rows = 1000;
        let sum = array![0.1 * rows as f64, 0.0];
        let sumsq = array![0.1 * 0.1 * rows as f64, rows as f64];
        let moments = OutputMoments::from_sums(rows, &sum, &sumsq);
        assert_eq!(moments.constant, vec![true, false]);
        assert_eq!(moments.scale[0], 1.0);
        assert!((moments.scale[1] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn moments_of_one_row_flag_nothing_constant() {
        let moments = OutputMoments::from_sums(1, &array![2.0], &array![4.0]);
        assert_eq!(moments.constant, vec![false]);
        assert_eq!(moments.scale[0], 1.0);
        assert_eq!(moments.center[0], 2.0);
    }

    #[test]
    fn constant_output_gets_zero_alpha_and_zero_standardized_value() {
        let moments = OutputMoments::from_sums(4, &array![4.0, 2.0], &array![4.0, 2.0]);
        let z = moments.standardized(&array![[1.0, 1.0]]);
        assert_eq!(z[[0, 0]], 0.0);
        let blend = moments
            .into_blend(array![0.5, 0.5], Array2::eye(2))
            .unwrap();
        assert_eq!(blend.alpha[0], 0.0);
        assert_eq!(blend.alpha[1], 0.5);
    }
}
