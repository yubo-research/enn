use ndarray::{Array1, Array2, ArrayView1};

pub(crate) const PRIOR_ALPHA: f64 = 0.3;
const SHRINK_ROWS: f64 = 12.0;
const TINY: f64 = 1e-12;

#[derive(Clone)]
pub struct OutputWeights {
    loadings: Array2<f64>,
    projector: Array2<f64>,
    num_output: Array1<f64>,
    den_output: Array1<f64>,
    num_pooled: f64,
    den_pooled: f64,
    num_factor: Array1<f64>,
    den_factor: Array1<f64>,
    rows: f64,
}

impl OutputWeights {
    pub fn new(loadings: Array2<f64>) -> Self {
        let outputs = loadings.nrows();
        let factors = loadings.ncols();
        let projector = loadings.dot(&loadings.t());
        Self {
            loadings,
            projector,
            num_output: Array1::zeros(outputs),
            den_output: Array1::from_elem(outputs, TINY),
            num_pooled: 0.0,
            den_pooled: TINY,
            num_factor: Array1::zeros(factors),
            den_factor: Array1::from_elem(factors, TINY),
            rows: 0.0,
        }
    }

    pub fn loadings(&self) -> &Array2<f64> {
        &self.loadings
    }

    pub fn projector(&self) -> &Array2<f64> {
        &self.projector
    }

    pub fn rows(&self) -> usize {
        self.rows as usize
    }

    fn trust(&self) -> f64 {
        self.rows / (self.rows + SHRINK_ROWS)
    }

    pub fn observe(&mut self, a: ArrayView1<f64>, t: ArrayView1<f64>) {
        let gap = a.dot(&self.projector) - a;
        let miss = &t - &a;
        self.num_output += &(&gap * &miss);
        self.den_output += &gap.mapv(|v| v * v);
        self.num_pooled += gap.dot(&miss);
        self.den_pooled += gap.dot(&gap);
        let shared_a = a.dot(&self.loadings);
        let shared_t = t.dot(&self.loadings);
        for f in 0..shared_a.len() {
            self.num_factor[f] += shared_a[f] * shared_t[f];
            self.den_factor[f] += shared_a[f] * shared_a[f];
        }
        self.rows += 1.0;
    }

    pub fn pooled_alpha(&self) -> f64 {
        let lam = self.trust();
        lam * (self.num_pooled / self.den_pooled).clamp(0.0, 1.0) + (1.0 - lam) * PRIOR_ALPHA
    }

    pub fn alpha(&self) -> Array1<f64> {
        let lam = self.trust();
        let pooled = self.pooled_alpha();
        (&self.num_output / &self.den_output)
            .mapv(|v| lam * v.clamp(0.0, 1.0) + (1.0 - lam) * pooled)
    }

    pub fn factor_weights(&self) -> Array1<f64> {
        let lam = self.trust();
        (&self.num_factor / &self.den_factor).mapv(|v| lam * v.clamp(0.0, 1.0) + (1.0 - lam))
    }

    pub fn weighted_projector(&self) -> Array2<f64> {
        let weights = self.factor_weights();
        let scaled = &self.loadings * &weights;
        scaled.dot(&self.loadings.t())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::{rngs::StdRng, Rng, SeedableRng};

    fn noise(rng: &mut StdRng, scale: f64) -> f64 {
        (rng.gen::<f64>() - 0.5) * 2.0 * scale
    }

    fn feed(signs: &[f64], loadings: Array2<f64>, rows: usize, seed: u64) -> OutputWeights {
        let mut state = OutputWeights::new(loadings);
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..rows {
            let signal = noise(&mut rng, 1.0);
            let a: Array1<f64> = signs
                .iter()
                .map(|s| s * signal + noise(&mut rng, 0.6))
                .collect();
            let t: Array1<f64> = signs
                .iter()
                .map(|s| s * signal + noise(&mut rng, 0.3))
                .collect();
            state.observe(a.view(), t.view());
        }
        state
    }

    fn one_group() -> Array2<f64> {
        Array2::from_elem((3, 1), 1.0 / 3f64.sqrt())
    }

    #[test]
    fn prior_before_any_row() {
        let state = OutputWeights::new(one_group());
        assert_eq!(state.rows(), 0);
        assert!((state.pooled_alpha() - PRIOR_ALPHA).abs() < 1e-12);
        assert!(state
            .alpha()
            .iter()
            .all(|v| (v - PRIOR_ALPHA).abs() < 1e-12));
        assert_eq!(state.factor_weights(), array![1.0]);
        assert_eq!(state.weighted_projector(), state.projector().clone());
    }

    #[test]
    fn shared_signal_raises_alpha_and_independent_outputs_lower_it() {
        let shared = feed(&[1.0, 1.0, 1.0], one_group(), 800, 1);
        assert!(
            shared.alpha().iter().all(|v| *v > 0.5),
            "{:?}",
            shared.alpha()
        );
        let mut state = OutputWeights::new(one_group());
        let mut rng = StdRng::seed_from_u64(2);
        for _ in 0..800 {
            let a: Array1<f64> = (0..3).map(|_| noise(&mut rng, 1.0)).collect();
            let t: Array1<f64> = a.iter().map(|v| v + noise(&mut rng, 0.2)).collect();
            state.observe(a.view(), t.view());
        }
        assert!(
            state.alpha().iter().all(|v| *v < 0.1),
            "{:?}",
            state.alpha()
        );
    }

    #[test]
    fn per_output_alpha_protects_the_opposed_output() {
        let state = feed(&[-1.0, 1.0, 1.0], one_group(), 800, 3);
        let alpha = state.alpha();
        assert!(alpha[0] < 0.2, "{alpha:?}");
        assert!(alpha[0] < 0.5 * alpha[1].min(alpha[2]), "{alpha:?}");
        assert!(alpha[1] > 0.3 && alpha[2] > 0.3, "{alpha:?}");
        assert!(state.pooled_alpha() > alpha[0]);
    }

    #[test]
    fn misleading_factor_weight_falls_and_good_factor_stays() {
        let s = 0.5f64.sqrt();
        let loadings = array![[s, 0.0], [s, 0.0], [0.0, s], [0.0, s]];
        let state = feed(&[1.0, 1.0, 1.0, -1.0], loadings, 800, 4);
        let g = state.factor_weights();
        assert!(g[0] > 0.7, "{g:?}");
        assert!(g[1] < 0.3, "{g:?}");
        let p = state.weighted_projector();
        assert!(p[[2, 2]] < 0.5 * state.projector()[[2, 2]]);
    }

    #[test]
    fn state_does_not_grow_with_rows() {
        let state = feed(&[1.0, 1.0, 1.0], one_group(), 2000, 5);
        assert_eq!(state.rows(), 2000);
        assert_eq!(state.num_output.len(), 3);
        assert_eq!(state.num_factor.len(), 1);
    }
}
