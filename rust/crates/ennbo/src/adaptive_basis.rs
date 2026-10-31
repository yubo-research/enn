use crate::adaptive_geometry::{fixed_rank, invalid, quadratic, ridge, root};
use crate::error::ENNError;
use ndarray::{s, Array1, Array2, Axis};
use ndarray_linalg::SVD;

#[derive(Clone, Default)]
pub struct Metadata {
    pub bounds: Option<Array2<f64>>,
    pub input: Option<Array2<f64>>,
    pub output: Option<Array2<f64>>,
}

pub(crate) struct Basis {
    pub x_mean: Array1<f64>,
    pub x_scale: Array1<f64>,
    pub y_mean: Array1<f64>,
    pub y_scale: Array1<f64>,
    pub constant: Vec<bool>,
    pub ridge: Array2<f64>,
    pub linear: Array2<f64>,
    pub quadratic: Array2<f64>,
    pub operators: Vec<(&'static str, Array2<f64>)>,
    pub metadata: Metadata,
    pub column_scale: Array1<f64>,
}

fn statistics(x: &Array2<f64>) -> (Array1<f64>, Array1<f64>, Vec<bool>) {
    let mean = x.mean_axis(Axis(0)).unwrap();
    let centered = x - &mean;
    let constant: Vec<_> = x
        .columns()
        .into_iter()
        .map(|c| c.iter().all(|v| *v == c[0]))
        .collect();
    let scale = Array1::from_shape_fn(x.ncols(), |j| {
        if constant[j] {
            1.0
        } else {
            (centered.column(j).mapv(|v| v * v).sum() / x.nrows() as f64)
                .sqrt()
                .max(1e-12)
        }
    });
    (mean, scale, constant)
}

impl Basis {
    pub fn new(x: &Array2<f64>, y: &Array2<f64>, metadata: &Metadata) -> Result<Self, ENNError> {
        let (x_mean, x_scale, _) = statistics(x);
        let (y_mean, y_scale, constant) = statistics(y);
        let z = (x - &x_mean) / &x_scale;
        let target = (y - &y_mean) / &y_scale;
        let ridge = ridge(&z, &target)?;
        let rank = metadata.input.as_ref().map(Array2::ncols);
        let linear = fixed_rank(ridge.clone(), rank)?;
        let quadratic = fixed_rank(quadratic(&z, &target)?, rank)?;
        let operators = output_operators(&target, metadata.output.as_ref())?;
        Ok(Self {
            x_mean,
            x_scale,
            y_mean,
            y_scale,
            constant,
            ridge,
            linear,
            quadratic,
            operators,
            metadata: metadata.clone(),
            column_scale: Array1::zeros(0),
        })
    }

    pub fn features(
        &self,
        x: &Array2<f64>,
        geometry: &str,
        weight: f64,
    ) -> Result<Array2<f64>, ENNError> {
        let d = x.ncols();
        if geometry == "scales" {
            let bounds = self.metadata.bounds.as_ref().ok_or_else(|| {
                invalid("column scales require input bounds")
            })?;
            let projection = self.metadata.input.as_ref().ok_or_else(|| {
                invalid("column scales require a projection")
            })?;
            let z = crate::adaptive_scales::centered(x, bounds);
            let factors = z.dot(projection);
            return ndarray::concatenate(Axis(1), &[z.view(), factors.view()]).map_err(invalid);
        }
        if geometry == "raw" || geometry == "fixed" {
            let Some(bounds) = &self.metadata.bounds else {
                return Ok(x.clone());
            };
            let z = Array2::from_shape_fn(x.dim(), |(i, j)| {
                (x[[i, j]] - bounds[[j, 0]]) / (bounds[[j, 1]] - bounds[[j, 0]])
            });
            if geometry == "raw" {
                return Ok(z);
            }
            let direction = self.metadata.input.as_ref().unwrap();
            return Ok(z.dot(&root(
                &(Array2::<f64>::eye(d) + direction.dot(&direction.t())),
            )?));
        }
        let z = (x - &self.x_mean) / &self.x_scale;
        let direction = if geometry == "linear" {
            &self.linear
        } else {
            &self.quadratic
        };
        Ok(z.dot(&root(
            &(Array2::<f64>::eye(d) + direction.dot(&direction.t()) * weight),
        )?))
    }

    pub fn target(&self, x: &Array2<f64>, y: &Array2<f64>, trend: bool) -> Array2<f64> {
        let target = (y - &self.y_mean) / &self.y_scale;
        if trend {
            target - self.trend(x)
        } else {
            target
        }
    }

    pub fn trend(&self, x: &Array2<f64>) -> Array2<f64> {
        ((x - &self.x_mean) / &self.x_scale).dot(&self.ridge)
    }
}

fn output_operators(
    target: &Array2<f64>,
    loading: Option<&Array2<f64>>,
) -> Result<Vec<(&'static str, Array2<f64>)>, ENNError> {
    let identity = Array2::eye(target.ncols());
    let mut operators = vec![("independent", identity.clone())];
    if let Some(loading) = loading {
        let semantic = if loading.ncols() == 0 {
            identity.clone()
        } else {
            loading.dot(&loading.t())
        };
        let (_, _, vt) = target.svd(false, true).map_err(invalid)?;
        let vt = vt.unwrap();
        let retained = vt.slice(s![..loading.ncols().min(vt.nrows()), ..]);
        let learned = if loading.ncols() == 0 {
            identity
        } else {
            retained.t().dot(&retained)
        };
        operators.extend([
            ("semantic", semantic),
            ("learned", learned.clone()),
            ("lowrank", learned),
        ]);
    }
    Ok(operators)
}
