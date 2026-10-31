//! Bounded uniform priority replay with transactional epoch refits.
use crate::adaptive_basis::Metadata;
use crate::adaptive_geometry::invalid;
use crate::adaptive_predictor::{Choice, Epoch};
use crate::adaptive_scales::{ColumnScales, STATIONARY_FORGET};
use crate::adaptive_selection::fit_epoch;
use crate::error::ENNError;
use ndarray::{Array2, Axis};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};

/// Online semantic adaptation with bounded replay and held-out calibration.
pub struct AdaptiveSemanticENN {
    metadata: Metadata,
    capacity: usize,
    linear_mean: bool,
    adaptation: String,
    rng: StdRng,
    sample: Option<(Array2<f64>, Array2<f64>, Vec<f64>)>,
    scales: Option<ColumnScales>,
    epoch: Option<Epoch>,
    /// Total accepted observations.
    pub num_seen: usize,
    /// Number of completed epochs.
    pub num_refits: usize,
}
impl AdaptiveSemanticENN {
    /// Configure immutable metadata and replay policy.
    pub fn new(
        metadata: Metadata,
        capacity: usize,
        seed: u64,
        linear_mean: bool,
        adaptation: String,
    ) -> Result<Self, ENNError> {
        if capacity < 12 || !["all", "inputs", "outputs", "none"].contains(&adaptation.as_str()) {
            return Err(invalid(
                "capacity must be at least 12 and adaptation must be all, inputs, outputs, or none",
            ));
        }
        if metadata.bounds.is_some() != metadata.input.is_some() {
            return Err(invalid(
                "input bounds and projection must be supplied together",
            ));
        }
        Ok(Self {
            metadata,
            capacity,
            linear_mean,
            adaptation,
            rng: StdRng::seed_from_u64(seed),
            sample: None,
            scales: None,
            epoch: None,
            num_seen: 0,
            num_refits: 0,
        })
    }
    /// Current replay size.
    pub fn num_retained(&self) -> usize {
        self.sample.as_ref().map_or(0, |s| s.0.nrows())
    }
    /// Selected candidate, available after the first successful update.
    pub fn choice(&self) -> Option<&Choice> {
        self.epoch.as_ref().map(|e| &e.predictor.choice)
    }

    /// Refit an epoch, committing replay and random state only after success.
    pub fn add(
        &mut self,
        x: Array2<f64>,
        y: Array2<f64>,
        weight: Option<f64>,
    ) -> Result<(), ENNError> {
        self.validate(&x, &y, weight)?;
        let count = x.nrows();
        let scales = self.updated_scales(&x, &y)?;
        let learned = scales.as_ref().map(ColumnScales::coefficients);
        let mut rng = self.rng.clone();
        let mut keys: Vec<f64> = (0..count).map(|_| rng.gen()).collect();
        let (x, y) = match &self.sample {
            Some((old_x, old_y, old_keys)) => {
                keys = [old_keys.as_slice(), &keys].concat();
                (
                    ndarray::concatenate(Axis(0), &[old_x.view(), x.view()]).map_err(invalid)?,
                    ndarray::concatenate(Axis(0), &[old_y.view(), y.view()]).map_err(invalid)?,
                )
            }
            None => (x, y),
        };
        let mut keep: Vec<_> = (0..keys.len()).collect();
        keep.sort_by(|a, b| keys[*a].total_cmp(&keys[*b]).then(a.cmp(b)));
        keep.truncate(self.capacity);
        let sample = (
            x.select(Axis(0), &keep),
            y.select(Axis(0), &keep),
            keep.iter().map(|i| keys[*i]).collect(),
        );
        let mut order: Vec<_> = (0..keep.len()).collect();
        order.shuffle(&mut rng);
        let epoch = fit_epoch(
            &sample.0.select(Axis(0), &order),
            &sample.1.select(Axis(0), &order),
            &self.metadata,
            self.linear_mean,
            &self.adaptation,
            weight,
            learned.as_ref(),
        )?;
        self.sample = Some(sample);
        self.scales = scales;
        self.epoch = Some(epoch);
        self.rng = rng;
        self.num_seen += count;
        self.num_refits += 1;
        Ok(())
    }

    fn updated_scales(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
    ) -> Result<Option<ColumnScales>, ENNError> {
        if !matches!(self.adaptation.as_str(), "all" | "inputs") {
            return Ok(None);
        }
        let Some(projection) = self.metadata.input.clone() else {
            return Ok(None);
        };
        if projection.ncols() == 0 {
            return Ok(self.scales.clone());
        }
        let Some(bounds) = &self.metadata.bounds else {
            return Ok(None);
        };
        let mut state = self.scales.clone().unwrap_or_else(|| {
            ColumnScales::new(projection, y.ncols(), STATIONARY_FORGET)
        });
        state.observe_rows(x, y, bounds)?;
        Ok(Some(state))
    }

    /// Learned multipliers, when input adaptation is active.
    #[cfg(test)]
    pub(crate) fn column_scale(&self) -> Option<ndarray::Array1<f64>> {
        self.scales.as_ref().map(ColumnScales::coefficients)
    }

    fn validate(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
        weight: Option<f64>,
    ) -> Result<(), ENNError> {
        if weight.is_some_and(|v| !v.is_finite() || v <= 0.0) {
            return Err(invalid("geometry_weight must be positive and finite"));
        }
        if x.nrows() == 0
            || x.ncols() == 0
            || y.ncols() == 0
            || x.nrows() != y.nrows()
            || x.iter().chain(y.iter()).any(|v| !v.is_finite())
        {
            return Err(invalid(
                "observations must be finite, nonempty matching matrices",
            ));
        }
        if let Some((old_x, old_y, _)) = &self.sample {
            if old_x.ncols() != x.ncols() || old_y.ncols() != y.ncols() {
                return Err(invalid(
                    "observation dimensions must match previous batches",
                ));
            }
        } else if x.nrows() < 12 {
            return Err(invalid("initial batch must have at least 12 rows"));
        }
        if let Some(bounds) = &self.metadata.bounds {
            crate::semantic_projection::validate_bounds(&bounds.view(), x.ncols())?;
            crate::semantic::validate(self.metadata.input.as_ref(), x.ncols(), false)?;
        }
        if self.metadata.output.as_ref().is_some_and(|p| {
            p.nrows() != y.ncols() || p.ncols() >= y.ncols() || p.iter().any(|v| !v.is_finite())
        }) {
            return Err(invalid("output columns must match the semantic projection"));
        }
        Ok(())
    }

    /// Predict calibrated marginal means and standard errors.
    pub fn posterior(&self, x: &Array2<f64>) -> Result<(Array2<f64>, Array2<f64>), ENNError> {
        let epoch = self
            .epoch
            .as_ref()
            .ok_or_else(|| invalid("add must be called before posterior"))?;
        if x.ncols() != epoch.predictor.basis.x_mean.len() || x.iter().any(|v| !v.is_finite()) {
            return Err(invalid(
                "query must be finite with the fitted input dimension",
            ));
        }
        if x.nrows() == 0 {
            let empty = Array2::zeros((0, epoch.predictor.basis.y_mean.len()));
            return Ok((empty.clone(), empty));
        }
        epoch.posterior(x)
    }
}
