use ndarray::{Array2, ArrayView2};

use super::EpistemicNearestNeighbors;
use crate::error::ENNError;
use crate::y_bounds::resolve_y_bounds;

impl EpistemicNearestNeighbors {
    pub(super) fn validate_shapes(
        train_x: &ArrayView2<f64>,
        train_y: &ArrayView2<f64>,
        train_yvar: Option<&ArrayView2<f64>>,
    ) -> Result<(), ENNError> {
        if train_x.nrows() != train_y.nrows() {
            return Err(ENNError::InvalidShape {
                expected: vec![train_y.nrows(), train_x.ncols()],
                got: vec![train_x.nrows(), train_x.ncols()],
            });
        }
        if let Some(yvar) = train_yvar {
            if yvar.shape() != train_y.shape() {
                return Err(ENNError::InvalidShape {
                    expected: train_y.shape().to_vec(),
                    got: yvar.shape().to_vec(),
                });
            }
        }
        validate_finite_xy(train_x, train_y, train_yvar)
    }

    pub(crate) fn validate_first_rows(
        train_x: &ArrayView2<f64>,
        train_y: &ArrayView2<f64>,
        train_yvar: Option<&ArrayView2<f64>>,
        y_bounds: Option<&Array2<f64>>,
    ) -> Result<(), ENNError> {
        Self::validate_shapes(train_x, train_y, train_yvar)?;
        let bounds = resolve_y_bounds(y_bounds, train_y.ncols())?;
        let owned_yvar = train_yvar.map(|v| v.to_owned());
        Self::ingress_warp_owned(train_y.to_owned(), owned_yvar, &bounds).map(|_| ())
    }

    fn validate_add(
        &self,
        x: &ArrayView2<f64>,
        y: &ArrayView2<f64>,
        yvar: Option<&ArrayView2<f64>>,
    ) -> Option<ENNError> {
        if x.nrows() != y.nrows() || x.ncols() != self.num_dim || y.ncols() != self.num_metrics {
            return Some(ENNError::InvalidShape {
                expected: vec![y.nrows(), self.num_metrics],
                got: vec![x.nrows(), x.ncols()],
            });
        }
        if y.ncols() != self.num_metrics {
            return Some(ENNError::InvalidParameter(format!(
                "y has {} metric columns but model expects {}",
                y.ncols(),
                self.num_metrics
            )));
        }
        if let Err(e) = validate_finite_xy(x, y, yvar) {
            return Some(e);
        }
        match (yvar, self.rows().row_yvar(0).ok().flatten().is_some()) {
            (Some(yv), _) if yv.shape() != y.shape() => Some(ENNError::InvalidShape {
                expected: y.shape().to_vec(),
                got: yv.shape().to_vec(),
            }),
            (Some(_), false) if self.num_obs > 0 => Some(ENNError::InvalidParameter(
                "yvar provided but model has no existing yvar".to_string(),
            )),
            (None, true) if self.num_obs > 0 => Some(ENNError::InvalidParameter(
                "yvar must be provided if model has existing yvar".to_string(),
            )),
            _ => None,
        }
    }

    fn validate_original(
        &self,
        train_x: &ArrayView2<f64>,
        original_x: Option<&ArrayView2<f64>>,
    ) -> Result<(), ENNError> {
        match (original_x, self.original_x.is_some()) {
            (None, true) => Err(ENNError::InvalidParameter(
                "this store keeps original inputs, so add() requires them".into(),
            )),
            (Some(_), false) => Err(ENNError::InvalidParameter(
                "this store keeps no original inputs".into(),
            )),
            (Some(original), true) if original.nrows() != train_x.nrows() => {
                Err(ENNError::InvalidShape {
                    expected: vec![train_x.nrows(), original.ncols()],
                    got: vec![original.nrows(), original.ncols()],
                })
            }
            _ => Ok(()),
        }
    }

    pub(crate) fn prepare_new_rows(
        &self,
        train_x: &ArrayView2<f64>,
        original_x: Option<&ArrayView2<f64>>,
        train_y: &ArrayView2<f64>,
        train_yvar: Option<&ArrayView2<f64>>,
    ) -> Result<(Array2<f64>, Option<Array2<f64>>), ENNError> {
        if let Some(err) = self.validate_add(train_x, train_y, train_yvar) {
            return Err(err);
        }
        self.validate_original(train_x, original_x)?;
        self.warp_observations(train_y, train_yvar)
    }
}

fn validate_finite_xy(
    x: &ArrayView2<f64>,
    y: &ArrayView2<f64>,
    yvar: Option<&ArrayView2<f64>>,
) -> Result<(), ENNError> {
    if x.iter().any(|v| !v.is_finite()) {
        return Err(ENNError::InvalidParameter(
            "x must contain only finite values".to_string(),
        ));
    }
    if y.iter().any(|v| !v.is_finite()) {
        return Err(ENNError::InvalidParameter(
            "y must contain only finite values".to_string(),
        ));
    }
    if let Some(yv) = yvar {
        if yv.iter().any(|v| !v.is_finite()) {
            return Err(ENNError::InvalidParameter(
                "yvar must contain only finite values".to_string(),
            ));
        }
        if yv.iter().any(|&v| v < 0.0) {
            return Err(ENNError::InvalidParameter(
                "yvar must be non-negative".to_string(),
            ));
        }
    }
    Ok(())
}
