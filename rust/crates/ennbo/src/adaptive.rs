use crate::adaptive_basis::Metadata;
use crate::adaptive_fitter::AdaptiveInputs;
use crate::adaptive_geometry::invalid;
use crate::adaptive_predictor::{Choice, Epoch};
use crate::adaptive_scales::centered;
use crate::adaptive_weight::{GeometryWeight, PRIOR_WEIGHT};
use crate::error::ENNError;
use crate::fit_samples::ScaleSearch;
use crate::fitter::ENNFitter;
use crate::online_input_scales::InputScales;
use crate::online_output_weights::OutputWeights;
use ndarray::{Array1, Array2, Axis};
use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};

const FITTER_NEIGHBORS: i32 = 10;
const SCALE_NEIGHBORS: usize = 10;
const SCALE_RESERVOIR: usize = 300;
const SCALE_SEED: u64 = 0x5ca1e;
const MIN_LEARNED_ROWS: usize = 12;

pub struct AdaptiveSemanticENN {
    metadata: Metadata,
    capacity: usize,
    linear_mean: bool,
    adaptation: String,
    rng: StdRng,
    fitter: ENNFitter,
    sample: Option<(Array2<f64>, Array2<f64>, Vec<f64>)>,
    scales: Option<Vec<InputScales>>,
    output_weights: Option<OutputWeights>,
    weight_state: Option<GeometryWeight>,
    direction: Option<Array2<f64>>,
    epoch: Option<Epoch>,
    pub num_seen: usize,
    pub num_refits: usize,
}
impl AdaptiveSemanticENN {
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
            fitter: ENNFitter::new(FITTER_NEIGHBORS),
            sample: None,
            scales: None,
            output_weights: None,
            weight_state: None,
            direction: None,
            epoch: None,
            num_seen: 0,
            num_refits: 0,
        })
    }
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    pub fn num_retained(&self) -> usize {
        self.sample.as_ref().map_or(0, |s| s.0.nrows())
    }
    pub fn choice(&self) -> Option<&Choice> {
        self.epoch.as_ref().map(|e| &e.predictor.choice)
    }

    pub fn add(
        &mut self,
        x: Array2<f64>,
        y: Array2<f64>,
        weight: Option<f64>,
    ) -> Result<(), ENNError> {
        self.validate(&x, &y, weight)?;
        let count = x.nrows();
        let mut next_weight = self.weight_state.clone();
        if weight.is_none() {
            let state =
                next_weight.get_or_insert_with(|| GeometryWeight::new(x.ncols(), y.ncols()));
            let direction = self.direction.clone();
            for row in 0..count {
                state.observe(x.row(row), y.row(row), direction.as_ref());
            }
        }
        let geometry_weight = weight.unwrap_or_else(|| {
            next_weight
                .as_ref()
                .map_or(PRIOR_WEIGHT, GeometryWeight::weight)
        });
        let scales = self.updated_scales(&x, &y)?;
        let learned = scales.as_deref().map(coefficients);
        let output_weights = self.updated_output_weights(&x, &y);
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
        let (fit_x, fit_y) = (
            sample.0.select(Axis(0), &order),
            sample.1.select(Axis(0), &order),
        );
        let mut epoch = self.fitter.ask_adaptive(
            &AdaptiveInputs {
                x: &fit_x,
                y: &fit_y,
                metadata: &self.metadata,
                trend: self.linear_mean,
                adaptation: &self.adaptation,
                weight: geometry_weight,
                column_scale: learned.as_ref(),
            },
            &ScaleSearch::default(),
            &mut rng,
        )?;
        if let Some(weights) = output_weights
            .as_ref()
            .filter(|w| w.rows() >= MIN_LEARNED_ROWS)
        {
            epoch.blend = Some((weights.weighted_projector(), weights.alpha()));
        }
        self.output_weights = output_weights;
        self.sample = Some(sample);
        self.scales = scales;
        self.weight_state = next_weight;
        self.direction = Some(active_direction(&epoch));
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
    ) -> Result<Option<Vec<InputScales>>, ENNError> {
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
        if x.ncols() != projection.nrows() {
            return Err(invalid(
                "column-scale observations must match the projection and outputs",
            ));
        }
        let mut learners = self.scales.clone().unwrap_or_else(|| {
            (0..y.ncols())
                .map(|j| {
                    InputScales::new(
                        &projection,
                        SCALE_NEIGHBORS,
                        SCALE_RESERVOIR,
                        SCALE_SEED + j as u64,
                    )
                })
                .collect()
        });
        let z = centered(x, bounds);
        let factors = z.dot(&projection);
        for row in 0..x.nrows() {
            for (j, learner) in learners.iter_mut().enumerate() {
                learner.observe(z.row(row), factors.row(row), y[[row, j]]);
            }
        }
        Ok(Some(learners))
    }

    fn updated_output_weights(&self, x: &Array2<f64>, y: &Array2<f64>) -> Option<OutputWeights> {
        if !matches!(self.adaptation.as_str(), "all" | "outputs") {
            return None;
        }
        let loading = self.metadata.output.as_ref().filter(|p| p.ncols() > 0)?;
        let mut weights = self
            .output_weights
            .clone()
            .unwrap_or_else(|| OutputWeights::new(loading.clone()));
        if let Some(epoch) = &self.epoch {
            let basis = &epoch.predictor.basis;
            if let Ok(mean) = epoch.predictor.standardized_mean(x) {
                let target = (y - &basis.y_mean) / &basis.y_scale;
                for (mut a, mut t) in mean
                    .rows()
                    .into_iter()
                    .zip(target.rows())
                    .map(|(a, t)| (a.to_owned(), t.to_owned()))
                {
                    for (j, constant) in basis.constant.iter().enumerate() {
                        if *constant {
                            a[j] = 0.0;
                            t[j] = 0.0;
                        }
                    }
                    weights.observe(a.view(), t.view());
                }
            }
        }
        Some(weights)
    }

    #[cfg(test)]
    pub(crate) fn column_scale(&self) -> Option<ndarray::Array1<f64>> {
        self.scales.as_deref().map(coefficients)
    }

    #[cfg(test)]
    pub(crate) fn learned_factor_weights(&self) -> Option<Array1<f64>> {
        self.output_weights
            .as_ref()
            .map(OutputWeights::factor_weights)
    }

    #[cfg(test)]
    pub(crate) fn learned_alpha(&self) -> Option<Array1<f64>> {
        self.epoch.as_ref()?.blend.as_ref().map(|b| b.1.clone())
    }

    #[cfg(test)]
    pub(crate) fn learned_weight(&self) -> f64 {
        self.weight_state
            .as_ref()
            .map_or(PRIOR_WEIGHT, GeometryWeight::weight)
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

fn coefficients(learners: &[InputScales]) -> Array1<f64> {
    let mut total = learners[0].scale().mapv(f64::sqrt);
    for learner in &learners[1..] {
        total += &learner.scale().mapv(f64::sqrt);
    }
    total / learners.len() as f64
}

fn active_direction(epoch: &Epoch) -> Array2<f64> {
    let basis = &epoch.predictor.basis;
    if epoch.predictor.choice.geometry == "quadratic" {
        basis.quadratic.clone()
    } else {
        basis.linear.clone()
    }
}
