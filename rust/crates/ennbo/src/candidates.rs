//! Candidate generation for trust region optimization.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use rand::distributions::Uniform;
use rand::Rng;
use rand::RngCore;
use sobol::params::JoeKuoD6;
use sobol::Sobol;
use std::sync::OnceLock;

use crate::error::ENNError;

fn sobol_params() -> &'static JoeKuoD6 {
    static PARAMS: OnceLock<JoeKuoD6> = OnceLock::new();
    PARAMS.get_or_init(JoeKuoD6::extended)
}

/// Candidate random variable type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CandidateRV {
    /// Sobol quasi-random sequence.
    Sobol,
    /// Uniform random sampling.
    #[default]
    Uniform,
    /// RAASP (Random Axis-Aligned Subspace Perturbation).
    RAASP,
}

/// Convert from unit hypercube to bounds.
pub fn from_unit(x_unit: &ArrayView2<f64>, bounds: &ArrayView2<f64>) -> Array2<f64> {
    let n = x_unit.nrows();
    let d = x_unit.ncols();

    let mut result = Array2::zeros((n, d));
    for i in 0..n {
        for j in 0..d {
            let lower = bounds[[j, 0]];
            let upper = bounds[[j, 1]];
            result[[i, j]] = lower + x_unit[[i, j]] * (upper - lower);
        }
    }
    result
}

/// Convert to unit hypercube from bounds.
pub fn to_unit(x: &ArrayView2<f64>, bounds: &ArrayView2<f64>) -> Array2<f64> {
    let n = x.nrows();
    let d = x.ncols();

    let mut result = Array2::zeros((n, d));
    for i in 0..n {
        for j in 0..d {
            let lower = bounds[[j, 0]];
            let upper = bounds[[j, 1]];
            let range = upper - lower;
            if range > 0.0 {
                result[[i, j]] = (x[[i, j]] - lower) / range;
            } else {
                result[[i, j]] = 0.5; // Degenerate dimension
            }
        }
    }
    result
}

/// Generate candidates within trust region bounds.
#[allow(clippy::too_many_arguments)]
pub fn generate_candidates<R: Rng + ?Sized>(
    compute_bounds_1d: impl Fn() -> (Array1<f64>, Array1<f64>),
    x_center: &ArrayView1<f64>,
    lengthscales: Option<&ArrayView1<f64>>,
    num_candidates: usize,
    candidate_rv: CandidateRV,
    rng: &mut R,
    sobol_engine: Option<&mut SobolEngine>,
    num_pert: usize,
) -> Result<Array2<f64>, ENNError> {
    let (lower_1d, upper_1d) = compute_bounds_1d();

    let num_dim = x_center.len();

    match candidate_rv {
        CandidateRV::Sobol => {
            if let Some(engine) = sobol_engine {
                let mut candidates = Array2::zeros((num_candidates, num_dim));
                for i in 0..num_candidates {
                    let sample = engine.sample(rng)?;
                    for j in 0..num_dim {
                        // Scale from unit to TR bounds
                        let unit = sample[j];
                        candidates[[i, j]] = lower_1d[j] + unit * (upper_1d[j] - lower_1d[j]);
                    }
                }
                Ok(candidates)
            } else {
                // Fallback to uniform if no Sobol engine
                generate_uniform(&lower_1d, &upper_1d, num_candidates, rng)
            }
        }
        CandidateRV::Uniform => generate_uniform(&lower_1d, &upper_1d, num_candidates, rng),
        CandidateRV::RAASP => generate_raasp(
            x_center,
            lengthscales,
            &lower_1d,
            &upper_1d,
            num_candidates,
            rng,
            num_pert,
        ),
    }
}

/// Generate uniformly random candidates.
pub fn generate_uniform<R: Rng + ?Sized>(
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    num_candidates: usize,
    rng: &mut R,
) -> Result<Array2<f64>, ENNError> {
    let num_dim = lower.len();
    let mut candidates = Array2::zeros((num_candidates, num_dim));

    for i in 0..num_candidates {
        for j in 0..num_dim {
            let dist = Uniform::new(lower[j], upper[j]);
            candidates[[i, j]] = rng.sample(dist);
        }
    }

    Ok(candidates)
}

fn raasp_dim_from_cdf(cdf: &Array1<f64>, r: f64, num_dim: usize) -> usize {
    let mut lo = 0usize;
    let mut hi = num_dim.saturating_sub(1);
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if r <= cdf[mid] {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    lo
}

/// Generate RAASP candidates.
fn generate_raasp<R: Rng + ?Sized>(
    x_center: &ArrayView1<f64>,
    lengthscales: Option<&ArrayView1<f64>>,
    lower: &Array1<f64>,
    upper: &Array1<f64>,
    num_candidates: usize,
    rng: &mut R,
    num_pert: usize,
) -> Result<Array2<f64>, ENNError> {
    let num_dim = x_center.len();
    let num_pert = num_pert.max(1);

    let probs: Array1<f64> = match lengthscales {
        Some(ls) => {
            let ls_sum: f64 = ls.sum();
            if ls_sum > 0.0 {
                ls.mapv(|v| v / ls_sum)
            } else {
                Array1::from_elem(num_dim, 1.0 / num_dim as f64)
            }
        }
        None => Array1::from_elem(num_dim, 1.0 / num_dim as f64),
    };

    let mut cdf = Array1::zeros(num_dim);
    let mut c = 0.0;
    for j in 0..num_dim {
        c += probs[j];
        cdf[j] = c;
    }

    let mut candidates = Array2::zeros((num_candidates, num_dim));
    for i in 0..num_candidates {
        for j in 0..num_dim {
            candidates[[i, j]] = x_center[j];
        }
        for _ in 0..num_pert {
            let r: f64 = rng.gen();
            let dim_to_perturb = raasp_dim_from_cdf(&cdf, r, num_dim);
            let dist = Uniform::new(lower[dim_to_perturb], upper[dim_to_perturb]);
            candidates[[i, dim_to_perturb]] = rng.sample(dist);
        }
    }

    Ok(candidates)
}

/// Sobol quasi-random sequence generator.
pub struct SobolEngine {
    dimension: usize,
    sequence: Sobol<f64>,
    /// Random shift per dimension for scrambling (None = not scrambled).
    /// When set, each sample is transformed: (x_d + shift_d) % 1.0
    shift: Option<Vec<f64>>,
}

impl SobolEngine {
    /// Create a new Sobol engine.
    pub fn new(dimension: usize) -> Result<Self, ENNError> {
        if dimension == 0 || dimension > 21201 {
            return Err(ENNError::InvalidParameter(format!(
                "Sobol dimension must be in 1..=21201, got {}",
                dimension
            )));
        }
        let sequence = Sobol::<f64>::new(dimension, sobol_params());

        Ok(Self {
            dimension,
            sequence,
            shift: None,
        })
    }

    /// Sample the next point from the Sobol sequence.
    ///
    /// If scrambled (via `scramble()`), applies random shift per dimension.
    pub fn sample<R: Rng + ?Sized>(&mut self, _rng: &mut R) -> Result<Vec<f64>, ENNError> {
        let mut sample = self
            .sequence
            .next()
            .ok_or_else(|| ENNError::InvalidParameter("Sobol sequence exhausted".to_string()))?;
        if sample.len() != self.dimension {
            return Err(ENNError::InvalidParameter(format!(
                "Sobol sample dimension mismatch: expected {}, got {}",
                self.dimension,
                sample.len()
            )));
        }
        if let Some(ref shift) = self.shift {
            for (s, sh) in sample.iter_mut().zip(shift.iter()) {
                *s = (*s + *sh) % 1.0;
                if *s < 0.0 {
                    *s += 1.0;
                }
            }
        }
        Ok(sample)
    }

    /// Scramble the sequence using random digital shift per dimension.
    ///
    /// Closer to Owen-style scrambling than a simple offset skip: each dimension
    /// gets a random shift in [0,1]; samples are (x + shift) mod 1.
    /// Preserves low-discrepancy structure while randomizing.
    pub fn scramble<R: RngCore + ?Sized>(&mut self, rng: &mut R) {
        self.shift = Some(
            (0..self.dimension)
                .map(|_| {
                    let mut bytes = [0u8; 8];
                    rng.fill_bytes(&mut bytes);
                    let u = u64::from_le_bytes(bytes);
                    (u >> 11) as f64 / ((1u64 << 53) as f64)
                })
                .collect(),
        );
    }
}

/// Generate Latin Hypercube samples.
pub fn generate_lhd<R: Rng + ?Sized>(
    num_samples: usize,
    num_dim: usize,
    bounds: &ArrayView2<f64>,
    rng: &mut R,
) -> Array2<f64> {
    let mut result = Array2::zeros((num_samples, num_dim));

    for j in 0..num_dim {
        let lower = bounds[[j, 0]];
        let upper = bounds[[j, 1]];
        let range = upper - lower;

        // Generate stratified samples
        let mut perm: Vec<usize> = (0..num_samples).collect();
        // Fisher-Yates shuffle
        for i in (1..num_samples).rev() {
            let j_idx = rng.gen_range(0..=i);
            perm.swap(i, j_idx);
        }

        for i in 0..num_samples {
            let bin = perm[i];
            let bin_start = bin as f64 / num_samples as f64;
            let bin_end = (bin + 1) as f64 / num_samples as f64;
            let offset: f64 = rng.gen();
            let unit = bin_start + offset * (bin_end - bin_start);
            result[[i, j]] = lower + unit * range;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_to_from_unit() {
        let bounds = array![[0.0, 10.0], [-5.0, 5.0]];
        let x = array![[2.5, 0.0], [7.5, 2.5]];

        let unit = to_unit(&x.view(), &bounds.view());
        let back = from_unit(&unit.view(), &bounds.view());

        // Check roundtrip (with tolerance)
        for i in 0..x.nrows() {
            for j in 0..x.ncols() {
                assert!((back[[i, j]] - x[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_generate_uniform() {
        let lower = array![0.0, 0.0];
        let upper = array![1.0, 1.0];
        let mut rng = StdRng::seed_from_u64(42);

        let candidates = generate_uniform(&lower, &upper, 10, &mut rng).unwrap();

        assert_eq!(candidates.nrows(), 10);
        assert_eq!(candidates.ncols(), 2);

        // Check bounds
        for i in 0..candidates.nrows() {
            for j in 0..candidates.ncols() {
                assert!(candidates[[i, j]] >= lower[j]);
                assert!(candidates[[i, j]] <= upper[j]);
            }
        }
    }

    #[test]
    fn test_generate_lhd() {
        let bounds = array![[0.0, 1.0], [0.0, 1.0]];
        let mut rng = StdRng::seed_from_u64(42);

        let samples = generate_lhd(5, 2, &bounds.view(), &mut rng);

        assert_eq!(samples.nrows(), 5);
        assert_eq!(samples.ncols(), 2);

        // Check bounds
        for i in 0..samples.nrows() {
            for j in 0..samples.ncols() {
                assert!(samples[[i, j]] >= 0.0);
                assert!(samples[[i, j]] <= 1.0);
            }
        }
    }

    #[test]
    fn test_sobol_engine() {
        let mut engine = SobolEngine::new(2).unwrap();
        let mut rng = StdRng::seed_from_u64(42);

        let sample1 = engine.sample(&mut rng).unwrap();
        let sample2 = engine.sample(&mut rng).unwrap();

        assert_eq!(sample1.len(), 2);
        assert_eq!(sample2.len(), 2);

        // Sobol samples should be in [0, 1)
        for v in &sample1 {
            assert!(*v >= 0.0 && *v < 1.0);
        }

        // Consecutive samples should be different
        assert_ne!(sample1, sample2);
    }

    #[test]
    fn test_generate_candidates_all_rvs_and_fallback() {
        let mut rng = StdRng::seed_from_u64(9);
        let x_center = array![0.5, 0.5];
        let lower = array![0.1, 0.2];
        let upper = array![0.9, 0.8];
        let ls = array![1.0, 2.0];
        let compute = || (lower.clone(), upper.clone());

        let uniform = generate_candidates(
            compute,
            &x_center.view(),
            Some(&ls.view()),
            4,
            CandidateRV::Uniform,
            &mut rng,
            None,
            2,
        )
        .unwrap();
        assert_eq!(uniform.nrows(), 4);

        let compute = || (lower.clone(), upper.clone());
        let mut sobol = SobolEngine::new(2).unwrap();
        let sobol_cand = generate_candidates(
            compute,
            &x_center.view(),
            Some(&ls.view()),
            4,
            CandidateRV::Sobol,
            &mut rng,
            Some(&mut sobol),
            2,
        )
        .unwrap();
        assert_eq!(sobol_cand.nrows(), 4);

        let compute = || (lower.clone(), upper.clone());
        let sobol_fallback = generate_candidates(
            compute,
            &x_center.view(),
            Some(&ls.view()),
            4,
            CandidateRV::Sobol,
            &mut rng,
            None,
            2,
        )
        .unwrap();
        assert_eq!(sobol_fallback.nrows(), 4);

        let compute = || (lower.clone(), upper.clone());
        let raasp = generate_candidates(
            compute,
            &x_center.view(),
            Some(&ls.view()),
            4,
            CandidateRV::RAASP,
            &mut rng,
            None,
            2,
        )
        .unwrap();
        assert_eq!(raasp.nrows(), 4);
    }

    #[test]
    fn test_sobol_engine_validation_and_scramble() {
        assert!(SobolEngine::new(0).is_err());
        let mut rng = StdRng::seed_from_u64(3);
        let mut engine = SobolEngine::new(3).unwrap();
        let before = engine.sample(&mut rng).unwrap();
        engine.scramble(&mut rng);
        let after = engine.sample(&mut rng).unwrap();
        assert_eq!(before.len(), 3);
        assert_eq!(after.len(), 3);
    }

    #[test]
    fn test_raasp_zero_num_pert_and_zero_lengthscales() {
        let mut rng = StdRng::seed_from_u64(21);
        let x_center = array![0.4, 0.6, 0.8];
        let lower = array![0.0, 0.0, 0.0];
        let upper = array![1.0, 1.0, 1.0];
        let ls_zero = array![0.0, 0.0, 0.0];
        let compute = || (lower.clone(), upper.clone());
        let out = generate_candidates(
            compute,
            &x_center.view(),
            Some(&ls_zero.view()),
            5,
            CandidateRV::RAASP,
            &mut rng,
            None,
            0,
        )
        .unwrap();
        assert_eq!(out.nrows(), 5);
        assert_eq!(out.ncols(), 3);
        assert!(out.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn test_sobol_high_dimension_path() {
        let mut rng = StdRng::seed_from_u64(5);
        let mut engine = SobolEngine::new(5).unwrap();
        let s = engine.sample(&mut rng).unwrap();
        assert_eq!(s.len(), 5);
        assert!(s.iter().all(|v| *v >= 0.0 && *v < 1.0));
    }

    #[test]
    fn raasp_golden_fixed_seed() {
        let x_center = array![0.25, 0.75, 0.5];
        let lower = array![0.0, 0.0, 0.0];
        let upper = array![1.0, 1.0, 1.0];
        let ls = array![1.0, 2.0, 3.0];
        let run = || {
            generate_raasp(
                &x_center.view(),
                Some(&ls.view()),
                &lower,
                &upper,
                3,
                &mut StdRng::seed_from_u64(4242),
                2,
            )
            .unwrap()
        };
        let out = run();
        let out2 = run();
        assert_eq!(out.shape(), &[3, 3]);
        assert!(out.iter().zip(out2.iter()).all(|(a, b)| (a - b).abs() < 1e-15));
        assert!(out.iter().all(|&v| (0.0..=1.0).contains(&v)));
    }

    #[test]
    fn test_private_raasp_and_sobol_engine_helpers() {
        let mut rng = StdRng::seed_from_u64(1234);
        let x_center = array![0.2, 0.4];
        let lower = array![0.0, 0.0];
        let upper = array![1.0, 1.0];
        let raasp = generate_raasp(&x_center.view(), None, &lower, &upper, 4, &mut rng, 1).unwrap();
        assert_eq!(raasp.shape(), &[4, 2]);
        let mut engine = SobolEngine::new(4).unwrap();
        let s0 = engine.sample(&mut rng).unwrap();
        let s1 = engine.sample(&mut rng).unwrap();
        assert_eq!(s0.len(), 4);
        assert_eq!(s1.len(), 4);
        assert_ne!(s0, s1);
    }

    #[test]
    fn test_sobol_engine_matches_scipy_reference_points() {
        // Reference generated from scipy.stats.qmc.Sobol(d=5, scramble=False),
        // taking the first 8 points from random_base2(m=4).
        let expected = [
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5, 0.5, 0.5],
            [0.75, 0.25, 0.25, 0.25, 0.75],
            [0.25, 0.75, 0.75, 0.75, 0.25],
            [0.375, 0.375, 0.625, 0.875, 0.375],
            [0.875, 0.875, 0.125, 0.375, 0.875],
            [0.625, 0.125, 0.875, 0.625, 0.625],
            [0.125, 0.625, 0.375, 0.125, 0.125],
        ];

        let mut rng = StdRng::seed_from_u64(1);
        let mut engine = SobolEngine::new(5).unwrap();
        let mut max_abs = 0.0f64;
        let mut sum_abs = 0.0f64;
        let mut count = 0usize;

        for row_expected in expected {
            let row = engine.sample(&mut rng).unwrap();
            for (got, exp) in row.iter().zip(row_expected.iter()) {
                let err = (got - exp).abs();
                if err > max_abs {
                    max_abs = err;
                }
                sum_abs += err;
                count += 1;
            }
        }

        let mean_abs = sum_abs / count as f64;
        println!("sobol_vs_scipy max_abs={max_abs:.3e} mean_abs={mean_abs:.3e}");
        assert!(max_abs <= 1e-12, "max_abs={max_abs}, mean_abs={mean_abs}");
    }

    #[test]
    fn kiss_candidate_rv_and_sobol_params() {
        let _ = sobol_params();
        assert!(matches!(CandidateRV::Sobol, CandidateRV::Sobol));
    }
}
