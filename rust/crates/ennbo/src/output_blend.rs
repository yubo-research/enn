use ndarray::{Array1, Array2, Array3, ArrayViewD, ArrayViewMut2, Axis, Ix2};

use crate::error::{invalid, ENNError};
use crate::params::ENNNormal;
use crate::semantic_definition::checked_loadings;

#[derive(Clone)]
pub struct SemanticOutputs {
    pub loadings: Array2<f64>,
    pub alpha: Option<f64>,
}

impl SemanticOutputs {
    pub fn new(
        loadings: ArrayViewD<'_, f64>,
        alpha: Option<f64>,
        num_metrics: usize,
    ) -> Result<Self, ENNError> {
        if alpha.is_some_and(|a| !(0.0..=1.0).contains(&a)) {
            return Err(invalid("output_alpha must be in [0, 1]"));
        }
        Ok(Self {
            loadings: checked_loadings(&loadings, num_metrics)?,
            alpha,
        })
    }
}

#[derive(Debug, PartialEq)]
pub struct OutputBlend {
    pub alpha: Array1<f64>,
    pub projector: Array2<f64>,
    pub center: Array1<f64>,
    pub scale: Array1<f64>,
}

impl OutputBlend {
    pub fn new(
        alpha: Array1<f64>,
        projector: Array2<f64>,
        center: Array1<f64>,
        scale: Array1<f64>,
    ) -> Result<Self, ENNError> {
        let m = alpha.len();
        if projector.dim() != (m, m) || center.len() != m || scale.len() != m {
            return Err(invalid(format!(
                "output blend arrays must agree on {m} outputs: projector {:?}, center {}, scale {}",
                projector.dim(),
                center.len(),
                scale.len()
            )));
        }
        if alpha.iter().any(|a| !(0.0..=1.0).contains(a)) {
            return Err(invalid("output blend alpha must be in [0, 1]"));
        }
        if projector
            .iter()
            .chain(center.iter())
            .any(|v| !v.is_finite())
        {
            return Err(invalid("output blend projector and center must be finite"));
        }
        if scale.iter().any(|v| !v.is_finite() || *v <= 0.0) {
            return Err(invalid("output blend scale must be positive and finite"));
        }
        Ok(Self {
            alpha,
            projector,
            center,
            scale,
        })
    }

    pub fn num_outputs(&self) -> usize {
        self.alpha.len()
    }

    fn operator(&self) -> Array2<f64> {
        let eye = Array2::<f64>::eye(self.num_outputs());
        (&self.projector - &eye) * &self.alpha + eye
    }

    fn blend_mean(&self, operator: &Array2<f64>, mut mean: ArrayViewMut2<'_, f64>) {
        let z = (&mean.view() - &self.center) / &self.scale;
        mean.assign(&(z.dot(operator) * &self.scale + &self.center));
    }

    fn blend_se(&self, magnitude: &Array2<f64>, mut se: ArrayViewMut2<'_, f64>) {
        let z = &se.view() / &self.scale;
        se.assign(&(z.dot(magnitude) * &self.scale));
    }

    fn blend_views(&self, mu: ArrayViewMut2<'_, f64>, ses: [ArrayViewMut2<'_, f64>; 3]) {
        let operator = self.operator();
        let magnitude = operator.mapv(f64::abs);
        self.blend_mean(&operator, mu);
        for se in ses {
            self.blend_se(&magnitude, se);
        }
    }

    pub(crate) fn apply_normal(&self, out: &mut ENNNormal) -> Result<(), ENNError> {
        let ENNNormal {
            mu,
            se,
            se_epi,
            se_ale,
            ..
        } = out;
        self.blend_views(
            mu.view_mut().into_dimensionality::<Ix2>()?,
            [
                se.view_mut().into_dimensionality::<Ix2>()?,
                se_epi.view_mut().into_dimensionality::<Ix2>()?,
                se_ale.view_mut().into_dimensionality::<Ix2>()?,
            ],
        );
        Ok(())
    }

    pub(crate) fn apply_batch_entry(
        &self,
        out: &mut ENNNormal,
        entry: usize,
    ) -> Result<(), ENNError> {
        let ENNNormal {
            mu,
            se,
            se_epi,
            se_ale,
            ..
        } = out;
        self.blend_views(
            mu.index_axis_mut(Axis(0), entry)
                .into_dimensionality::<Ix2>()?,
            [
                se.index_axis_mut(Axis(0), entry)
                    .into_dimensionality::<Ix2>()?,
                se_epi
                    .index_axis_mut(Axis(0), entry)
                    .into_dimensionality::<Ix2>()?,
                se_ale
                    .index_axis_mut(Axis(0), entry)
                    .into_dimensionality::<Ix2>()?,
            ],
        );
        Ok(())
    }

    pub(crate) fn apply_draws(&self, draws: &mut Array3<f64>) {
        let operator = self.operator();
        for draw in draws.axis_iter_mut(Axis(0)) {
            self.blend_mean(&operator, draw);
        }
    }

    #[cfg(test)]
    pub(crate) fn mean_of(&self, mean: ndarray::ArrayView2<'_, f64>) -> Array2<f64> {
        let mut out = mean.to_owned();
        self.blend_mean(&self.operator(), out.view_mut());
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn pair_blend(alpha: Array1<f64>) -> OutputBlend {
        OutputBlend::new(
            alpha,
            Array2::from_elem((2, 2), 0.5),
            array![1.0, -1.0],
            array![2.0, 4.0],
        )
        .unwrap()
    }

    #[test]
    fn zero_alpha_is_the_identity() {
        let blend = pair_blend(array![0.0, 0.0]);
        let mean = array![[3.0, 7.0], [-1.0, 0.5]];
        let got = blend.mean_of(mean.view());
        assert!((got - &mean).iter().all(|d| d.abs() < 1e-12));
    }

    #[test]
    fn mean_and_se_follow_the_hand_computed_operator() {
        let blend = pair_blend(array![0.5, 0.0]);
        let mut normal = ENNNormal::new(
            array![[3.0, 3.0]].into_dyn(),
            array![[2.0, 4.0]].into_dyn(),
            array![[2.0, 0.0]].into_dyn(),
            array![[0.0, 4.0]].into_dyn(),
            None,
        );
        blend.apply_normal(&mut normal).unwrap();
        // O = [[0.75, 0], [0.25, 1]], z = [1, 1], se/scale = [1, 1]
        assert!((normal.mu[[0, 0]] - (1.0 * 2.0 + 1.0)).abs() < 1e-12);
        assert!((normal.mu[[0, 1]] - 3.0).abs() < 1e-12);
        assert!((normal.se[[0, 0]] - 2.0).abs() < 1e-12);
        assert!((normal.se[[0, 1]] - 4.0).abs() < 1e-12);
        assert!((normal.se_epi[[0, 0]] - 1.5).abs() < 1e-12);
        assert!((normal.se_ale[[0, 0]] - 0.5).abs() < 1e-12);
    }

    #[test]
    fn new_rejects_inconsistent_or_invalid_arrays() {
        let eye = Array2::<f64>::eye(2);
        let one = array![1.0, 1.0];
        let zero = array![0.0, 0.0];
        let reject = |alpha: Array1<f64>, projector: Array2<f64>, center, scale| {
            assert!(OutputBlend::new(alpha, projector, center, scale).is_err());
        };
        reject(array![0.1], eye.clone(), zero.clone(), one.clone());
        reject(array![1.5, 0.0], eye.clone(), zero.clone(), one.clone());
        reject(
            array![f64::NAN, 0.0],
            eye.clone(),
            zero.clone(),
            one.clone(),
        );
        reject(
            zero.clone(),
            eye.clone() * f64::NAN,
            zero.clone(),
            one.clone(),
        );
        reject(
            zero.clone(),
            eye.clone(),
            array![f64::INFINITY, 0.0],
            one.clone(),
        );
        reject(zero.clone(), eye, zero.clone(), array![0.0, 1.0]);
    }

    #[test]
    fn draws_use_the_mean_rule_per_seed() {
        let blend = pair_blend(array![1.0, 1.0]);
        let mut draws = Array3::from_shape_vec((2, 1, 2), vec![3.0, 3.0, 1.0, -1.0]).unwrap();
        let expected_first = blend.mean_of(array![[3.0, 3.0]].view());
        blend.apply_draws(&mut draws);
        assert!((draws[[0, 0, 0]] - expected_first[[0, 0]]).abs() < 1e-12);
        assert!((draws[[1, 0, 0]] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn semantic_outputs_validate_alpha_and_loadings() {
        let loadings = array![[0.5f64.sqrt()], [0.5f64.sqrt()]].into_dyn();
        assert!(SemanticOutputs::new(loadings.view(), Some(0.3), 2).is_ok());
        assert!(SemanticOutputs::new(loadings.view(), Some(1.1), 2).is_err());
        assert!(SemanticOutputs::new(loadings.view(), Some(f64::NAN), 2).is_err());
        assert!(SemanticOutputs::new(loadings.view(), None, 3).is_err());
    }
}
