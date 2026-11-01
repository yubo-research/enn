use crate::adaptive_basis::{Basis, Metadata};
use crate::adaptive_geometry::invalid;
use crate::adaptive_scales::distance_scales;
use crate::error::ENNError;
use crate::index::IndexDriver;
use crate::model::EpistemicNearestNeighbors;
use crate::params::{ENNParams, PosteriorFlags};
use ndarray::{Array1, Array2, Ix2};

#[derive(Clone, Debug, PartialEq)]
pub struct Choice {
    pub geometry: String,
    pub weight: f64,
    pub output: String,
    pub params: ENNParams,
    pub alpha: f64,
}

pub(crate) struct Predictor {
    pub basis: Basis,
    pub model: EpistemicNearestNeighbors,
    pub choice: Choice,
    pub linear_mean: bool,
}

impl Predictor {
    pub fn new(
        x: &Array2<f64>,
        y: &Array2<f64>,
        metadata: &Metadata,
        choice: Choice,
        linear_mean: bool,
        column_scale: Option<&Array1<f64>>,
    ) -> Result<Self, ENNError> {
        let mut basis = Basis::new(x, y, metadata)?;
        if let Some(scale) = column_scale {
            basis.column_scale = scale.clone();
        }
        let features = basis.features(x, &choice.geometry, choice.weight)?;
        let target = basis.target(x, y, linear_mean);
        let scaled = choice.geometry == "scales";
        let mut model =
            EpistemicNearestNeighbors::new(features, target, None, scaled, IndexDriver::Flat)?;
        if scaled {
            install_column_scale(&mut model, &basis.column_scale)?;
        }
        Ok(Self {
            basis,
            model,
            choice,
            linear_mean,
        })
    }

    fn params(&self) -> Result<ENNParams, ENNError> {
        let chosen = self.choice.params;
        ENNParams::new(
            chosen.k_num_neighbors.min(self.model.len() as i32),
            chosen.epistemic_variance_scale,
            chosen.aleatoric_variance_scale,
        )
        .map_err(invalid)
    }

    pub fn standardized_mean(&self, x: &Array2<f64>) -> Result<Array2<f64>, ENNError> {
        let features = self
            .basis
            .features(x, &self.choice.geometry, self.choice.weight)?;
        let params = self.params()?;
        let posterior =
            self.model
                .posterior(&features.view(), &params, &PosteriorFlags::default())?;
        let mut mean = posterior.mu.into_dimensionality::<Ix2>().map_err(invalid)?;
        if self.linear_mean {
            mean += &self.basis.trend(x);
        }
        Ok(mean)
    }

    pub fn posterior_blend(
        &self,
        x: &Array2<f64>,
        blend: Option<&(Array2<f64>, Array1<f64>)>,
    ) -> Result<(Array2<f64>, Array2<f64>), ENNError> {
        let features = self
            .basis
            .features(x, &self.choice.geometry, self.choice.weight)?;
        let params = self.params()?;
        let posterior =
            self.model
                .posterior(&features.view(), &params, &PosteriorFlags::default())?;
        let mut mean = posterior.mu.into_dimensionality::<Ix2>().map_err(invalid)?;
        if self.linear_mean {
            mean += &self.basis.trend(x);
        }
        let operator = match blend {
            Some((projector, alpha)) => {
                let eye = Array2::<f64>::eye(mean.ncols());
                (projector - &eye) * alpha + eye
            }
            None => {
                let projector = &self
                    .basis
                    .operators
                    .iter()
                    .find(|(name, _)| *name == self.choice.output)
                    .unwrap()
                    .1;
                Array2::<f64>::eye(mean.ncols()) * (1.0 - self.choice.alpha)
                    + projector * self.choice.alpha
            }
        };
        let mut mu = mean.dot(&operator) * &self.basis.y_scale + &self.basis.y_mean;
        let mut se = posterior
            .se
            .into_dimensionality::<Ix2>()
            .map_err(invalid)?
            .dot(&operator.mapv(f64::abs))
            * &self.basis.y_scale;
        for (j, constant) in self.basis.constant.iter().enumerate() {
            if *constant {
                mu.column_mut(j).fill(self.basis.y_mean[j]);
                se.column_mut(j).fill(0.0);
            }
        }
        Ok((mu, se))
    }
}

pub(crate) fn install_column_scale(
    model: &mut EpistemicNearestNeighbors,
    coefficients: &Array1<f64>,
) -> Result<(), ENNError> {
    if !model.scale_x || coefficients.is_empty() || model.num_dim < coefficients.len() {
        return Err(invalid("column scales require a semantic factor"));
    }
    let dimensions = model.num_dim - coefficients.len();
    model.metric_fixed = true;
    model.x_scale = distance_scales(coefficients, dimensions);
    model.backend.mark_index_stale();
    model.ensure_index_sync()
}

pub(crate) struct Epoch {
    pub predictor: Predictor,
    pub blend: Option<(Array2<f64>, Array1<f64>)>,
    pub floor: Array1<f64>,
}
impl Epoch {
    pub fn posterior(&self, x: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>), ENNError> {
        let (mu, mut se) = self.predictor.posterior_blend(x, self.blend.as_ref())?;
        for ((_, j), value) in se.indexed_iter_mut() {
            *value = (value.powi(2).max(self.floor[j].powi(2))).sqrt();
        }
        Ok((mu, se))
    }
}
