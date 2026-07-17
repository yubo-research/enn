//! Correctness, metamorphic, and fuzz tests for the squared-L2 kernel
//! `l2_sq_f32`, which was rewritten to use lane accumulators for SIMD.
//!
//! These guard against the reduction being silently broken (e.g. dropping the
//! `chunks_exact` remainder for lengths that are not a multiple of the lane
//! count) or overfit to a single length.

use bpann::distance::l2_sq_f32;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// Independent, high-precision reference in f64. Deliberately a plain scalar
/// reduction so it shares no structure with the lane-accumulator kernel.
fn ref_sq_l2_f64(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| {
            let d = f64::from(x) - f64::from(y);
            d * d
        })
        .sum()
}

fn rand_vec(rng: &mut ChaCha8Rng, len: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|_| (rng.gen::<f32>() - 0.5) * 2.0 * scale)
        .collect()
}

/// Relative tolerance appropriate for an f32 reduction of `len` terms compared
/// against an f64 reference.
fn close(candidate: f32, reference: f64, len: usize) -> bool {
    let rel = 1e-4 * (1.0 + len as f64);
    (f64::from(candidate) - reference).abs() <= rel * reference.abs() + 1e-3
}

#[test]
fn l2_matches_scalar_reference_across_lengths() {
    // Cover lengths straddling the lane width (8): exact multiples, one below,
    // one above, empty, and the D=1000 production dimension.
    let seed: u64 = rand::random();
    println!("l2_matches_scalar_reference_across_lengths seed={seed}");
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let lengths = [0usize, 1, 7, 8, 9, 15, 16, 17, 31, 100, 127, 128, 1000];
    for &len in &lengths {
        for _ in 0..64 {
            let a = rand_vec(&mut rng, len, 10.0);
            let b = rand_vec(&mut rng, len, 10.0);
            let got = l2_sq_f32(&a, &b);
            let want = ref_sq_l2_f64(&a, &b);
            assert!(
                close(got, want, len),
                "len={len} got={got} want={want} seed={seed}"
            );
        }
    }
}

#[test]
fn l2_fuzz_random_lengths() {
    // Fully randomized lengths (including many non-multiples of the lane width,
    // which exercise the remainder path) and magnitudes. Must hold for any seed.
    let seed: u64 = rand::random();
    println!("l2_fuzz_random_lengths seed={seed}");
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for _ in 0..2000 {
        let len = rng.gen_range(0..300);
        let scale = rng.gen_range(1e-3f32..1e3);
        let a = rand_vec(&mut rng, len, scale);
        let b = rand_vec(&mut rng, len, scale);
        let got = l2_sq_f32(&a, &b);
        let want = ref_sq_l2_f64(&a, &b);
        assert!(
            got >= 0.0,
            "squared distance must be non-negative: got={got} len={len} seed={seed}"
        );
        assert!(
            close(got, want, len),
            "len={len} scale={scale} got={got} want={want} seed={seed}"
        );
    }
}

#[test]
fn l2_metamorphic_properties() {
    let seed: u64 = rand::random();
    println!("l2_metamorphic_properties seed={seed}");
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    for _ in 0..500 {
        let len = rng.gen_range(1..200);
        let a = rand_vec(&mut rng, len, 5.0);
        let b = rand_vec(&mut rng, len, 5.0);
        let base = l2_sq_f32(&a, &b);

        // Identity: distance from a vector to itself is exactly zero.
        assert_eq!(l2_sq_f32(&a, &a), 0.0, "self-distance nonzero seed={seed}");

        // Symmetry: (x-y)^2 == (y-x)^2 elementwise, so the result is exact.
        assert_eq!(
            base,
            l2_sq_f32(&b, &a),
            "asymmetric distance seed={seed} len={len}"
        );

        // Translation invariance: shifting both vectors by the same constant
        // leaves the squared distance unchanged (up to f32 rounding).
        let c = (rng.gen::<f32>() - 0.5) * 20.0;
        let a_shift: Vec<f32> = a.iter().map(|&v| v + c).collect();
        let b_shift: Vec<f32> = b.iter().map(|&v| v + c).collect();
        let shifted = l2_sq_f32(&a_shift, &b_shift);
        assert!(
            close(shifted, f64::from(base), len),
            "translation changed distance: base={base} shifted={shifted} c={c} seed={seed}"
        );

        // Positive scaling: scaling both vectors by k scales the squared
        // distance by k^2.
        let k = rng.gen_range(0.25f32..4.0);
        let a_scale: Vec<f32> = a.iter().map(|&v| v * k).collect();
        let b_scale: Vec<f32> = b.iter().map(|&v| v * k).collect();
        let scaled = l2_sq_f32(&a_scale, &b_scale);
        let want = f64::from(base) * f64::from(k) * f64::from(k);
        assert!(
            close(scaled, want, len),
            "scaling law violated: scaled={scaled} want={want} k={k} seed={seed}"
        );
    }
}
