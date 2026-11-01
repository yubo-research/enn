use ndarray::{Array1, Array2, ArrayView1};
use ndarray_linalg::Solve;
use rand::{rngs::StdRng, Rng, SeedableRng};

const RELATIVE_WEIGHTS: [f64; 7] = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0];
const PRIOR_WEIGHT: f64 = 0.3;
const SHRINK_ROWS: f64 = 12.0;
const SWITCH_MARGIN: f64 = 0.03;
const PAIRS_PER_ROW: usize = 10;
const MAX_RATIO: f64 = 1e3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Candidate {
    Prefix { r: f64, m: usize },
    Shape { r: f64 },
}

#[derive(Clone)]
struct Stored {
    z: Array1<f64>,
    s: Array1<f64>,
    y: f64,
}

#[derive(Clone)]
pub struct InputScales {
    dims: f64,
    factors: usize,
    column_energy: Array1<f64>,
    neighbors: usize,
    capacity: usize,
    rng: StdRng,
    reservoir: Vec<Stored>,
    candidates: Vec<Candidate>,
    loss: Vec<f64>,
    mass: f64,
    current: Option<usize>,
    switches: usize,
    gram: Array2<f64>,
    cross: Array1<f64>,
    shape: Array1<f64>,
    base: Array1<f64>,
    scale: Array1<f64>,
    seen: usize,
    y_mean: f64,
    y_m2: f64,
}

impl InputScales {
    pub fn new(projection: &Array2<f64>, neighbors: usize, capacity: usize, seed: u64) -> Self {
        let factors = projection.ncols();
        let column_energy: Array1<f64> = projection
            .columns()
            .into_iter()
            .map(|c| c.dot(&c))
            .collect();
        let mut candidates = vec![Candidate::Prefix { r: 0.0, m: 0 }];
        for r in RELATIVE_WEIGHTS {
            for m in 1..=factors {
                candidates.push(Candidate::Prefix { r, m });
            }
        }
        for r in RELATIVE_WEIGHTS {
            candidates.push(Candidate::Shape { r });
        }
        let mut state = Self {
            dims: projection.nrows() as f64,
            factors,
            column_energy,
            neighbors: neighbors.max(1),
            capacity: capacity.max(neighbors + 1),
            rng: StdRng::seed_from_u64(seed),
            reservoir: Vec::new(),
            loss: vec![0.0; candidates.len()],
            candidates,
            mass: 0.0,
            current: None,
            switches: 0,
            gram: Array2::zeros((factors + 2, factors + 2)),
            cross: Array1::zeros(factors + 2),
            shape: Array1::zeros(factors),
            base: Array1::zeros(factors),
            scale: Array1::zeros(factors),
            seen: 0,
            y_mean: 0.0,
            y_m2: 0.0,
        };
        state.base = state.prefix_scale(PRIOR_WEIGHT, factors);
        state.shape = state.base.clone();
        state.scale = state.base.clone();
        state
    }

    pub fn scale(&self) -> &Array1<f64> {
        &self.scale
    }

    pub fn switches(&self) -> usize {
        self.switches
    }

    pub fn stored_rows(&self) -> usize {
        self.reservoir.len()
    }

    pub fn current(&self) -> Option<Candidate> {
        self.current.map(|i| self.candidates[i])
    }

    fn prefix_scale(&self, r: f64, m: usize) -> Array1<f64> {
        let mut out = Array1::zeros(self.factors);
        let energy: f64 = self.column_energy.iter().take(m).sum();
        if m > 0 && energy > 1e-12 {
            out.iter_mut()
                .take(m)
                .for_each(|v| *v = r * self.dims / energy);
        }
        out
    }

    fn shape_scale(&self, r: f64) -> Array1<f64> {
        let energy = self.shape.dot(&self.column_energy);
        if energy > 1e-12 {
            &self.shape * (r * self.dims / energy)
        } else {
            Array1::zeros(self.factors)
        }
    }

    fn candidate_scale(&self, candidate: Candidate) -> Array1<f64> {
        match candidate {
            Candidate::Prefix { r, m } => self.prefix_scale(r, m),
            Candidate::Shape { r } => self.shape_scale(r),
        }
    }

    pub fn observe(&mut self, z: ArrayView1<f64>, s: ArrayView1<f64>, y: f64) {
        self.seen += 1;
        let delta = y - self.y_mean;
        self.y_mean += delta / self.seen as f64;
        self.y_m2 += delta * (y - self.y_mean);
        let variance = (self.y_m2 / self.seen as f64).max(1e-9);
        self.regress(&z, &s, y, variance);
        if self.reservoir.len() >= self.neighbors {
            self.score(&z, &s, y);
        }
        self.keep(z, s, y);
    }

    fn regress(&mut self, z: &ArrayView1<f64>, s: &ArrayView1<f64>, y: f64, variance: f64) {
        if self.reservoir.is_empty() {
            return;
        }
        let width = self.factors + 2;
        for _ in 0..PAIRS_PER_ROW {
            let other = &self.reservoir[self.rng.gen_range(0..self.reservoir.len())];
            let mut x = Array1::ones(width);
            x[0] = (&other.z - z).mapv(|v| v * v).sum();
            for f in 0..self.factors {
                x[f + 1] = (other.s[f] - s[f]).powi(2);
            }
            let t = (other.y - y).powi(2) / variance;
            for i in 0..width {
                self.cross[i] += x[i] * t;
                for j in 0..width {
                    self.gram[[i, j]] += x[i] * x[j];
                }
            }
        }
        if self.seen < 10 || !self.seen.is_multiple_of(self.factors.max(1)) {
            return;
        }
        let ridge = 1e-6 * self.gram.diag().sum() / width as f64;
        let mut system = self.gram.clone();
        for i in 0..width {
            system[[i, i]] += ridge;
        }
        let Ok(solution) = system.solve(&self.cross) else {
            return;
        };
        let rows = self.seen as f64;
        let anchor = solution[0].max(1e-3);
        for f in 0..self.factors {
            let ratio = (solution[f + 1].max(0.0) / anchor).min(MAX_RATIO);
            self.shape[f] = (rows * ratio + SHRINK_ROWS * self.base[f]) / (rows + SHRINK_ROWS);
        }
    }

    fn score(&mut self, z: &ArrayView1<f64>, s: &ArrayView1<f64>, y: f64) {
        let n = self.reservoir.len();
        let mut mean = 0.0;
        let mut m2 = 0.0;
        for (i, row) in self.reservoir.iter().enumerate() {
            let delta = row.y - mean;
            mean += delta / (i + 1) as f64;
            m2 += delta * (row.y - mean);
        }
        let variance = (m2 / n as f64).max(1e-9);
        let base: Vec<f64> = self
            .reservoir
            .iter()
            .map(|r| (&r.z - z).mapv(|v| v * v).sum())
            .collect();
        let gaps: Vec<Array1<f64>> = self
            .reservoir
            .iter()
            .map(|r| (&r.s - s).mapv(|v| v * v))
            .collect();
        let scales: Vec<Array1<f64>> = self
            .candidates
            .iter()
            .map(|c| self.candidate_scale(*c))
            .collect();
        let k = self.neighbors.min(n);
        for (c, scale) in scales.iter().enumerate() {
            let mut d2: Vec<(f64, usize)> =
                (0..n).map(|i| (base[i] + gaps[i].dot(scale), i)).collect();
            d2.select_nth_unstable_by(k - 1, |a, b| a.0.total_cmp(&b.0));
            let (mut num, mut den) = (0.0, 0.0);
            for (dist, i) in &d2[..k] {
                let w = 1.0 / (1e-12 + dist);
                num += w * self.reservoir[*i].y;
                den += w;
            }
            self.loss[c] += (num / den - y).powi(2) / variance;
        }
        self.mass += 1.0;
        if self.mass < SHRINK_ROWS {
            return;
        }
        let best = (0..self.loss.len())
            .min_by(|a, b| self.loss[*a].total_cmp(&self.loss[*b]))
            .unwrap();
        match self.current {
            None => self.current = Some(best),
            Some(cur) if self.loss[best] < (1.0 - SWITCH_MARGIN) * self.loss[cur] => {
                if best != cur {
                    self.switches += 1;
                }
                self.current = Some(best);
            }
            _ => {}
        }
        self.scale = self.candidate_scale(self.candidates[self.current.unwrap()]);
    }

    fn keep(&mut self, z: ArrayView1<f64>, s: ArrayView1<f64>, y: f64) {
        let row = Stored {
            z: z.to_owned(),
            s: s.to_owned(),
            y,
        };
        if self.reservoir.len() < self.capacity {
            self.reservoir.push(row);
        } else {
            let slot = self.rng.gen_range(0..self.seen);
            if slot < self.capacity {
                self.reservoir[slot] = row;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn projection() -> Array2<f64> {
        // 4 inputs; factor 0 mixes x0 and x1, factor 1 mixes x2 and x3.
        let mut w = Array2::zeros((4, 2));
        w[[0, 0]] = 1.0;
        w[[1, 0]] = 1.0;
        w[[2, 1]] = 1.0;
        w[[3, 1]] = 1.0;
        w
    }

    fn run(learner: &mut InputScales, rows: usize, seed: u64, f: impl Fn(&Array1<f64>) -> f64) {
        let w = projection();
        let mut rng = StdRng::seed_from_u64(seed);
        for _ in 0..rows {
            let z: Array1<f64> = (0..4).map(|_| rng.gen::<f64>() - 0.5).collect();
            let s = z.dot(&w);
            let y = f(&z) + (rng.gen::<f64>() - 0.5) * 0.2;
            learner.observe(z.view(), s.view(), y);
        }
    }

    #[test]
    fn prefix_scale_has_relative_weight_energy() {
        let learner = InputScales::new(&projection(), 8, 50, 0);
        for (r, m) in [(0.3, 1), (3.0, 2), (30.0, 2)] {
            let c = learner.prefix_scale(r, m);
            let energy = c.dot(&learner.column_energy);
            assert!((energy - r * 4.0).abs() < 1e-12, "{r} {m} {energy}");
        }
        assert_eq!(learner.prefix_scale(1.0, 0), array![0.0, 0.0]);
    }

    #[test]
    fn starts_at_the_prior_and_stores_a_bounded_reservoir() {
        let mut learner = InputScales::new(&projection(), 8, 60, 1);
        assert_eq!(learner.scale(), &learner.prefix_scale(PRIOR_WEIGHT, 2));
        assert!(learner.current().is_none());
        run(&mut learner, 500, 1, |z| z[2] + z[3]);
        assert_eq!(learner.stored_rows(), 60);
        assert!(learner.current().is_some());
    }

    #[test]
    fn learns_to_use_the_informative_factor() {
        // Response depends on the second factor's direction only.
        let signal = |z: &Array1<f64>| (6.0 * (z[2] + z[3])).sin();
        let mut learner = InputScales::new(&projection(), 8, 150, 2);
        run(&mut learner, 1200, 2, signal);
        let c = learner.scale();
        assert!(c[1] > 1.3 * c[0], "{c:?} {:?}", learner.current());
        let energy = c.dot(&learner.column_energy);
        assert!(energy > 4.0 * PRIOR_WEIGHT, "{energy}");
    }

    #[test]
    fn learns_to_drop_a_response_free_embedding() {
        // Response ignores both factor directions: x0 - x1 and x2 - x3 are not factor axes.
        let signal = |z: &Array1<f64>| (6.0 * (z[0] - z[1])).sin();
        let mut learner = InputScales::new(&projection(), 8, 150, 3);
        run(&mut learner, 1200, 3, signal);
        let energy = learner.scale().dot(&learner.column_energy);
        assert!(
            energy < 4.0 * PRIOR_WEIGHT + 1e-9,
            "{energy} {:?}",
            learner.current()
        );
    }

    #[test]
    fn switching_rule_limits_changes() {
        let mut learner = InputScales::new(&projection(), 8, 150, 4);
        run(&mut learner, 1500, 4, |z| (6.0 * (z[2] + z[3])).sin());
        assert!(learner.switches() < 30, "{}", learner.switches());
    }

    #[test]
    fn same_seed_gives_the_same_scale() {
        let signal = |z: &Array1<f64>| (6.0 * (z[2] + z[3])).sin();
        let mut a = InputScales::new(&projection(), 8, 80, 9);
        let mut b = InputScales::new(&projection(), 8, 80, 9);
        run(&mut a, 400, 9, signal);
        run(&mut b, 400, 9, signal);
        assert_eq!(a.scale(), b.scale());
    }

    #[test]
    fn no_factors_and_few_rows_are_handled() {
        let empty = Array2::<f64>::zeros((4, 0));
        let mut none = InputScales::new(&empty, 3, 20, 0);
        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..60 {
            let z: Array1<f64> = (0..4).map(|_| rng.gen::<f64>()).collect();
            none.observe(z.view(), Array1::zeros(0).view(), z[0]);
        }
        assert_eq!(none.scale().len(), 0);
        let mut few = InputScales::new(&projection(), 8, 50, 0);
        run(&mut few, 5, 0, |z| z[0]);
        assert!(few.current().is_none());
        assert_eq!(few.stored_rows(), 5);
    }

    fn knn_error(
        train: &[(Array1<f64>, Array1<f64>, f64)],
        test: &[(Array1<f64>, Array1<f64>, f64)],
        scale: &Array1<f64>,
    ) -> f64 {
        let mut total = 0.0;
        for (z, s, y) in test {
            let mut d: Vec<(f64, f64)> = train
                .iter()
                .map(|(tz, ts, ty)| {
                    let gap = (tz - z).mapv(|v| v * v).sum() + (ts - s).mapv(|v| v * v).dot(scale);
                    (gap, *ty)
                })
                .collect();
            d.sort_by(|a, b| a.0.total_cmp(&b.0));
            let (num, den) = d[..8].iter().fold((0.0, 0.0), |(n, w), (g, v)| {
                let k = 1.0 / (1e-12 + g);
                (n + k * v, w + k)
            });
            total += (num / den - y).powi(2);
        }
        (total / test.len() as f64).sqrt()
    }

    #[test]
    fn learned_scale_beats_the_prior_on_held_out_rows() {
        let w = projection();
        let signal = |z: &Array1<f64>| (6.0 * (z[2] + z[3])).sin();
        let mut learner = InputScales::new(&w, 8, 150, 11);
        run(&mut learner, 1000, 11, signal);
        let mut rng = StdRng::seed_from_u64(99);
        let mut draw = |n: usize| -> Vec<(Array1<f64>, Array1<f64>, f64)> {
            (0..n)
                .map(|_| {
                    let z: Array1<f64> = (0..4).map(|_| rng.gen::<f64>() - 0.5).collect();
                    let s = z.dot(&w);
                    let y = signal(&z);
                    (z, s, y)
                })
                .collect()
        };
        let (train, test) = (draw(600), draw(200));
        let learned = knn_error(&train, &test, learner.scale());
        let prior = knn_error(&train, &test, &learner.prefix_scale(PRIOR_WEIGHT, 2));
        assert!(learned < 0.9 * prior, "learned {learned} prior {prior}");
    }
}
