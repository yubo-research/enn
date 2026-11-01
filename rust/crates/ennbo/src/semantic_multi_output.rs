use crate::adaptive_geometry::invalid;
use crate::error::ENNError;
use crate::index::IndexDriver;
use crate::model::EpistemicNearestNeighbors;
use crate::numpy_pcg::NumpyPcg64;
use crate::params::{ENNParams, PosteriorFlags};
use ndarray::{s, Array1, Array2, ArrayView2, Axis, Ix2};

const SCALE_FLOOR: f64 = 1e-12;
const RELATIVE_FLOOR: f64 = 1e-6;
const CALIBRATION_FLOOR: f64 = 1e-12;
const MIN_TRAINING_ROWS: usize = 4;

struct OutputModels {
    mean: Array1<f64>,
    scale: Array1<f64>,
    constant: Vec<bool>,
    loadings: Array2<f64>,
    latent_scale: Option<Array1<f64>>,
    params: ENNParams,
    model: EpistemicNearestNeighbors,
}

fn output_statistics(y: &Array2<f64>) -> (Array1<f64>, Array1<f64>, Vec<bool>) {
    let mean = y.mean_axis(Axis(0)).unwrap();
    let constant: Vec<bool> = y
        .columns()
        .into_iter()
        .map(|c| c.iter().all(|v| *v == c[0]))
        .collect();
    let mut scale = y.std_axis(Axis(0), 0.0);
    for (value, flat) in scale.iter_mut().zip(&constant) {
        if *flat {
            *value = 1.0;
        }
    }
    (mean, scale, constant)
}

fn latent_scale(latent: &Array2<f64>) -> Array1<f64> {
    let count = latent.nrows() as f64;
    Array1::from_shape_fn(latent.ncols(), |j| {
        let column = latent.column(j);
        let mean = column.sum() / count;
        let second = column.dot(&column) / count;
        let std = (second - mean * mean).max(0.0).sqrt();
        if std.is_finite() && std > 0.0 {
            std
        } else {
            1.0
        }
    })
}

impl OutputModels {
    fn new(
        x: Array2<f64>,
        y: &Array2<f64>,
        loadings: &Array2<f64>,
        params: &ENNParams,
    ) -> Result<Self, ENNError> {
        let (mean, scale, constant) = output_statistics(y);
        let z = (y - &mean) / &scale;
        let latent_scale = (loadings.ncols() > 0).then(|| latent_scale(&z.dot(loadings)));
        let neighbors = (params.k_num_neighbors as usize).min(x.nrows()) as i32;
        Ok(Self {
            params: ENNParams::new(
                neighbors,
                params.epistemic_variance_scale,
                params.aleatoric_variance_scale,
            )
            .map_err(invalid)?,
            model: EpistemicNearestNeighbors::new(x, z, None, false, IndexDriver::Flat)?,
            mean,
            scale,
            constant,
            loadings: loadings.clone(),
            latent_scale,
        })
    }

    fn predict(
        &self,
        x: &ArrayView2<f64>,
        alpha: f64,
    ) -> Result<(Array2<f64>, Array2<f64>), ENNError> {
        let posterior = self
            .model
            .posterior(x, &self.params, &PosteriorFlags::default())?;
        let mut mean = posterior.mu.into_dimensionality::<Ix2>().map_err(invalid)?;
        let mut se = posterior.se.into_dimensionality::<Ix2>().map_err(invalid)?;
        if let (Some(latent), true) = (&self.latent_scale, alpha > 0.0) {
            let unit = self.model.y_scale_row()[[0, 0]];
            let factor = se.slice(s![.., ..1]).mapv(|v| v / unit);
            let shared_se = (factor * latent).dot(&self.loadings.t().mapv(f64::abs));
            let shared = mean.dot(&self.loadings).dot(&self.loadings.t());
            mean = mean * (1.0 - alpha) + shared * alpha;
            se = se * (1.0 - alpha) + shared_se * alpha;
        }
        for (j, constant) in self.constant.iter().enumerate() {
            if *constant {
                mean.column_mut(j).fill(0.0);
                se.column_mut(j).fill(0.0);
            }
        }
        Ok((mean * &self.scale + &self.mean, se * &self.scale))
    }
}

struct FittedOutputs {
    models: OutputModels,
    dimension: usize,
    floor: Array1<f64>,
    calibration: Array1<f64>,
}

pub struct OutOfFold {
    pub mean: Array2<f64>,
    pub variance: Array2<f64>,
    pub fold_ids: Vec<usize>,
}

pub struct SemanticMultiOutputENN {
    loadings: Array2<f64>,
    alpha: f64,
    params: ENNParams,
    folds: usize,
    seed: u64,
    fitted: Option<FittedOutputs>,
    out_of_fold: Option<OutOfFold>,
}

impl SemanticMultiOutputENN {
    pub fn new(
        loadings: Array2<f64>,
        alpha: f64,
        params: ENNParams,
        folds: usize,
        seed: u64,
    ) -> Result<Self, ENNError> {
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err(invalid("alpha must be finite and in [0, 1]"));
        }
        if folds < 2 {
            return Err(invalid("folds must be an integer at least two"));
        }
        Ok(Self {
            loadings,
            alpha,
            params,
            folds,
            seed,
            fitted: None,
            out_of_fold: None,
        })
    }

    pub fn is_fitted(&self) -> bool {
        self.fitted.is_some()
    }

    pub fn out_of_fold(&self) -> Option<&OutOfFold> {
        self.out_of_fold.as_ref()
    }

    fn active_loadings(&self) -> Array2<f64> {
        if self.alpha > 0.0 {
            self.loadings.clone()
        } else {
            self.loadings.slice(s![.., ..0]).to_owned()
        }
    }

    fn validate_training(&self, x: &Array2<f64>, y: &Array2<f64>) -> Result<(), ENNError> {
        if x.ncols() == 0
            || y.dim() != (x.nrows(), self.loadings.nrows())
            || x.nrows() < MIN_TRAINING_ROWS
            || x.iter().chain(y.iter()).any(|v| !v.is_finite())
        {
            return Err(invalid(
                "training data must be finite matching matrices with at least four rows",
            ));
        }
        if self.folds > x.nrows() {
            return Err(invalid("fold count cannot exceed training row count"));
        }
        Ok(())
    }

    fn fold_ids(&self, count: usize) -> Vec<usize> {
        let mut labels = vec![0; count];
        let order = NumpyPcg64::from_seed(self.seed).permutation(count);
        for (position, row) in order.into_iter().enumerate() {
            labels[row] = position % self.folds;
        }
        labels
    }

    fn predict_out_of_fold(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
        labels: &[usize],
        loadings: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>), ENNError> {
        let mut mean = Array2::zeros(y.raw_dim());
        let mut se = Array2::zeros(y.raw_dim());
        for fold in 0..self.folds {
            let (held, kept): (Vec<usize>, Vec<usize>) =
                (0..x.nrows()).partition(|row| labels[*row] == fold);
            let models = OutputModels::new(
                x.select(Axis(0), &kept),
                &y.select(Axis(0), &kept),
                loadings,
                &self.params,
            )?;
            let (fold_mean, fold_se) =
                models.predict(&x.select(Axis(0), &held).view(), self.alpha)?;
            for (position, row) in held.into_iter().enumerate() {
                mean.row_mut(row).assign(&fold_mean.row(position));
                se.row_mut(row).assign(&fold_se.row(position));
            }
        }
        Ok((mean, se))
    }

    pub fn fit(&mut self, x: Array2<f64>, y: Array2<f64>) -> Result<(), ENNError> {
        self.validate_training(&x, &y)?;
        let loadings = self.active_loadings();
        let labels = self.fold_ids(x.nrows());
        let (mean, se) = self.predict_out_of_fold(&x, &y, &labels, &loadings)?;
        let floor = y
            .std_axis(Axis(0), 0.0)
            .mapv(|v| (v * RELATIVE_FLOOR).max(SCALE_FLOOR));
        let raw_variance =
            Array2::from_shape_fn(se.dim(), |(i, j)| se[[i, j]].powi(2).max(floor[j].powi(2)));
        let ratios = (&y - &mean).mapv(|v| v * v) / &raw_variance;
        let calibration = ratios
            .mean_axis(Axis(0))
            .unwrap()
            .mapv(|v| v.max(CALIBRATION_FLOOR));
        let dimension = x.ncols();
        let models = OutputModels::new(x, &y, &loadings, &self.params)?;
        self.out_of_fold = Some(OutOfFold {
            mean,
            variance: raw_variance * &calibration,
            fold_ids: labels,
        });
        self.fitted = Some(FittedOutputs {
            models,
            dimension,
            floor,
            calibration,
        });
        Ok(())
    }

    fn fitted_for_query(&self, x: &Array2<f64>) -> Result<&FittedOutputs, ENNError> {
        let fitted = self
            .fitted
            .as_ref()
            .ok_or_else(|| invalid("fit must be called before posterior"))?;
        if x.ncols() != fitted.dimension || x.iter().any(|v| !v.is_finite()) {
            return Err(invalid(
                "query must be a finite matrix with the fitted input dimension",
            ));
        }
        Ok(fitted)
    }

    pub fn uncalibrated_posterior(
        &self,
        x: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>), ENNError> {
        self.fitted_for_query(x)?
            .models
            .predict(&x.view(), self.alpha)
    }

    pub fn posterior(&self, x: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>), ENNError> {
        let fitted = self.fitted_for_query(x)?;
        if x.nrows() == 0 {
            let empty = Array2::zeros((0, self.loadings.nrows()));
            return Ok((empty.clone(), empty));
        }
        let (mean, se) = fitted.models.predict(&x.view(), self.alpha)?;
        let calibrated = Array2::from_shape_fn(se.dim(), |(i, j)| {
            (se[[i, j]].powi(2).max(fitted.floor[j].powi(2)) * fitted.calibration[j]).sqrt()
        });
        Ok((mean, calibrated))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn rows() -> (Array2<f64>, Array2<f64>) {
        let x = Array2::from_shape_fn((18, 2), |(i, j)| ((i * (j + 3)) % 7) as f64 / 7.0);
        let y = Array2::from_shape_fn((18, 3), |(i, j)| match j {
            0 => (3.0 * x[[i, 0]]).sin(),
            1 => -(3.0 * x[[i, 0]]).sin(),
            _ => x[[i, 1]],
        });
        (x, y)
    }

    fn model(alpha: f64) -> SemanticMultiOutputENN {
        let root = 0.5_f64.sqrt();
        let loadings = array![[root], [root], [0.0]];
        let params = ENNParams::new(3, 1.0, 0.0).unwrap();
        SemanticMultiOutputENN::new(loadings, alpha, params, 3, 4).unwrap()
    }

    #[test]
    fn output_statistics_marks_constant_columns_with_unit_scale() {
        let (mean, scale, constant) = output_statistics(&array![[1.0, 7.0], [3.0, 7.0]]);
        assert_eq!(mean, array![2.0, 7.0]);
        assert_eq!(scale, array![1.0, 1.0]);
        assert_eq!(constant, vec![false, true]);
    }

    #[test]
    fn latent_scale_is_population_std_with_unit_fallback() {
        let scale = latent_scale(&array![[1.0, 2.0], [3.0, 2.0]]);
        assert_eq!(scale, array![1.0, 1.0]);
        assert_eq!(latent_scale(&array![[0.0, 4.0], [0.0, 0.0]])[1], 2.0);
    }

    #[test]
    fn calibration_makes_out_of_fold_variance_gaussian() {
        let (x, y) = rows();
        let mut fitted = model(0.3);
        fitted.fit(x.clone(), y.clone()).unwrap();
        let out = fitted.out_of_fold().unwrap();
        let standardized = (&y - &out.mean).mapv(|v| v * v) / &out.variance;
        for value in standardized.mean_axis(Axis(0)).unwrap() {
            assert!((value - 1.0).abs() < 1e-9, "{value}");
        }
        let (mean, se) = fitted.posterior(&x).unwrap();
        assert_eq!(mean.dim(), y.dim());
        assert!(se.iter().all(|v| v.is_finite() && *v > 0.0));
        assert!(fitted.uncalibrated_posterior(&x).is_ok());
    }

    #[test]
    fn held_rows_do_not_influence_their_own_predictions() {
        let (x, y) = rows();
        let mut base = model(0.3);
        base.fit(x.clone(), y.clone()).unwrap();
        let labels = base.out_of_fold().unwrap().fold_ids.clone();
        let mut changed = y.clone();
        for (row, label) in labels.iter().enumerate() {
            if *label == 0 {
                changed.row_mut(row).mapv_inplace(|v| v + 100.0);
            }
        }
        let mut other = model(0.3);
        other.fit(x, changed).unwrap();
        for (row, label) in labels.iter().enumerate() {
            if *label == 0 {
                assert_eq!(
                    base.out_of_fold().unwrap().mean.row(row),
                    other.out_of_fold().unwrap().mean.row(row)
                );
            }
        }
    }

    #[test]
    fn invalid_configuration_and_inputs_are_rejected() {
        let params = ENNParams::new(3, 1.0, 0.0).unwrap();
        let loadings = Array2::zeros((3, 1));
        assert!(SemanticMultiOutputENN::new(loadings.clone(), f64::NAN, params, 3, 0).is_err());
        assert!(SemanticMultiOutputENN::new(loadings, 0.5, params, 1, 0).is_err());
        let (x, y) = rows();
        let mut fitted = model(0.3);
        assert!(!fitted.is_fitted());
        assert!(fitted.posterior(&x).is_err());
        assert!(fitted
            .fit(
                x.slice(s![..2, ..]).to_owned(),
                y.slice(s![..2, ..]).to_owned()
            )
            .is_err());
        assert!(fitted
            .fit(x.clone(), y.slice(s![.., ..2]).to_owned())
            .is_err());
        fitted.fit(x, y).unwrap();
        assert!(fitted.is_fitted());
        assert!(fitted.posterior(&array![[f64::NAN, 1.0]]).is_err());
        assert_eq!(
            fitted.posterior(&Array2::zeros((0, 2))).unwrap().0.dim(),
            (0, 3)
        );
    }
}
