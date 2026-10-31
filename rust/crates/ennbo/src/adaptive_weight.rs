use ndarray::{Array1, Array2, ArrayView1};

pub(crate) const PRIOR_WEIGHT: f64 = 3.0;
pub(crate) const STEP_GAIN: f64 = 0.1;
const FORGET: f64 = 0.99;
const SHRINK_ROWS: f64 = 12.0;
const MIN_WEIGHT: f64 = 0.3;
const MAX_WEIGHT: f64 = 300.0;
const UNIT_STRENGTH_WEIGHT: f64 = 30.0;
const Z_LIMIT: f64 = 5.0;

#[derive(Clone)]
pub(crate) struct GeometryWeight {
    w: f64,
    stat_mass: f64,
    moment_mass: f64,
    x_sum: Array1<f64>,
    x_sumsq: Array1<f64>,
    y_sum: Array1<f64>,
    sum_q: f64,
    sum_q2: f64,
    sum_e: f64,
    sum_e2: f64,
    sum_qe: f64,
}

impl GeometryWeight {
    pub(crate) fn new(dimensions: usize, outputs: usize) -> Self {
        Self {
            w: PRIOR_WEIGHT,
            stat_mass: 0.0,
            moment_mass: 0.0,
            x_sum: Array1::zeros(dimensions),
            x_sumsq: Array1::zeros(dimensions),
            y_sum: Array1::zeros(outputs),
            sum_q: 0.0,
            sum_q2: 0.0,
            sum_e: 0.0,
            sum_e2: 0.0,
            sum_qe: 0.0,
        }
    }

    pub(crate) fn weight(&self) -> f64 {
        self.w
    }

    pub(crate) fn target(&self) -> f64 {
        if self.moment_mass <= 0.0 {
            return PRIOR_WEIGHT;
        }
        let n = self.moment_mass;
        let mean_q = self.sum_q / n;
        let mean_e = self.sum_e / n;
        let var_q = (self.sum_q2 / n - mean_q * mean_q).max(0.0);
        let var_e = (self.sum_e2 / n - mean_e * mean_e).max(0.0);
        if var_q <= 1e-18 || var_e <= 1e-18 {
            return PRIOR_WEIGHT;
        }
        let cov = self.sum_qe / n - mean_q * mean_e;
        let explained = if cov > 0.0 {
            (cov * cov / (var_q * var_e)).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let slope = cov / var_q;
        let strength = slope.max(0.0) * mean_q.max(0.0) * explained;
        let shrunk = n * strength / (n + SHRINK_ROWS);
        let expansion = UNIT_STRENGTH_WEIGHT / PRIOR_WEIGHT - 1.0;
        (PRIOR_WEIGHT * (1.0 + expansion * shrunk)).clamp(MIN_WEIGHT, MAX_WEIGHT)
    }

    pub(crate) fn observe(
        &mut self,
        x: ArrayView1<f64>,
        y: ArrayView1<f64>,
        direction: Option<&Array2<f64>>,
    ) {
        if self.stat_mass >= 1.0 {
            if let Some(direction) = direction {
                if direction.nrows() == x.len() && direction.ncols() > 0 {
                    self.absorb(x, y, direction);
                }
            }
        }
        self.x_sum *= FORGET;
        self.x_sumsq *= FORGET;
        self.y_sum *= FORGET;
        self.stat_mass *= FORGET;
        self.x_sum += &x;
        self.x_sumsq += &x.mapv(|value| value * value);
        self.y_sum += &y;
        self.stat_mass += 1.0;
    }

    fn absorb(&mut self, x: ArrayView1<f64>, y: ArrayView1<f64>, direction: &Array2<f64>) {
        let z = self.standardize(x);
        let residual = &y.to_owned() - &(&self.y_sum / self.stat_mass);
        let score = direction.t().dot(&z);
        let q = score.dot(&score);
        let energy = residual.dot(&residual);
        if !q.is_finite() || !energy.is_finite() || q > 1.0e12 || energy > 1.0e12 {
            return;
        }
        self.sum_q *= FORGET;
        self.sum_q2 *= FORGET;
        self.sum_e *= FORGET;
        self.sum_e2 *= FORGET;
        self.sum_qe *= FORGET;
        self.moment_mass *= FORGET;
        self.sum_q += q;
        self.sum_q2 += q * q;
        self.sum_e += energy;
        self.sum_e2 += energy * energy;
        self.sum_qe += q * energy;
        self.moment_mass += 1.0;
        let target = self.target();
        self.w = (self.w + STEP_GAIN * (target - self.w)).clamp(MIN_WEIGHT, MAX_WEIGHT);
    }

    fn standardize(&self, x: ArrayView1<f64>) -> Array1<f64> {
        let mean = &self.x_sum / self.stat_mass;
        Array1::from_shape_fn(x.len(), |column| {
            let variance =
                (self.x_sumsq[column] / self.stat_mass - mean[column] * mean[column]).max(0.0);
            let scale = if variance <= 1e-24 {
                1.0
            } else {
                variance.sqrt()
            };
            ((x[column] - mean[column]) / scale).clamp(-Z_LIMIT, Z_LIMIT)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array1};
    use rand::{rngs::StdRng, Rng, SeedableRng};

    fn stream(aligned: bool, seed: u64) -> GeometryWeight {
        let mut state = GeometryWeight::new(1, 1);
        let mut rng = StdRng::seed_from_u64(seed);
        let direction = array![[1.0]];
        for _ in 0..800 {
            let x = rng.gen::<f64>() * 2.0 - 1.0;
            let y = if aligned {
                x
            } else {
                rng.gen::<f64>() * 2.0 - 1.0
            };
            state.observe(array![x].view(), array![y].view(), Some(&direction));
        }
        state
    }

    #[test]
    fn one_row_moves_weight_by_at_most_the_gain_times_the_gap() {
        let mut state = GeometryWeight::new(1, 1);
        let direction = array![[1.0]];
        let mut rng = StdRng::seed_from_u64(7);
        let mut moved = false;
        for _ in 0..80 {
            let x = rng.gen::<f64>() * 2.0 - 1.0;
            let before = state.weight();
            state.observe(array![x].view(), array![3.0 * x].view(), Some(&direction));
            let gap = (state.target() - before).abs();
            let step = (state.weight() - before).abs();
            assert!(step <= STEP_GAIN * gap + 1e-9);
            moved |= step > 1e-6 && gap > step + 1e-6;
        }
        assert!(moved);
    }

    #[test]
    fn subspace_outputs_raise_weight_and_independent_outputs_stay_near_the_prior() {
        let aligned: Vec<f64> = (0..12).map(|seed| stream(true, seed).weight()).collect();
        let independent: Vec<f64> = (0..12).map(|seed| stream(false, seed).weight()).collect();
        assert!(aligned.iter().all(|weight| *weight > 10.0));
        assert!(independent
            .iter()
            .all(|weight| (*weight - PRIOR_WEIGHT).abs() < 0.5));
    }

    #[test]
    fn weight_state_stores_no_rows() {
        let mut state = GeometryWeight::new(4, 2);
        let direction = Array2::from_shape_fn((4, 1), |(row, _)| if row == 0 { 1.0 } else { 0.0 });
        for index in 0..250 {
            let x = Array1::from_shape_fn(4, |column| ((index * (column + 1)) as f64 * 0.13).sin());
            let y = array![x[0], -x[0]];
            state.observe(x.view(), y.view(), Some(&direction));
        }
        assert_eq!(state.x_sum.len(), 4);
        assert_eq!(state.y_sum.len(), 2);
        assert!(state.moment_mass > 50.0);
        assert!(state.moment_mass < 110.0);
    }
}
