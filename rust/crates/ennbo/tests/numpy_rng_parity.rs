use ennbo::numpy_pcg::NumpyPcg64;
use ennbo::ndtri::ndtri;
use ennbo::numpy_seed::seed_sequence_u64;
use ennbo::philox::NumpyPhilox;

#[test]
fn philox_seed_123_matches_numpy_key_and_random() {
    let key = seed_sequence_u64(123, 2);
    assert_eq!(key, [12_770_025_807_176_811_766, 11_695_957_281_888_622_767]);
    let mut rng = NumpyPhilox::from_seed(123);
    let got = [rng.random(), rng.random(), rng.random(), rng.random()];
    let expect = [
        0.900_076_506_487_439_5,
        0.905_983_613_685_421_7,
        0.246_851_198_368_481_02,
        0.759_671_731_113_301_2,
    ];
    for (a, b) in got.iter().zip(expect) {
        assert_eq!(*a, b);
    }
}

#[test]
fn pcg64_seed_0_integers_match_numpy() {
    let mut rng = NumpyPcg64::from_seed(0);
    let got: Vec<u64> = (0..8).map(|_| rng.integers_high(10)).collect();
    assert_eq!(got, vec![8, 6, 5, 2, 3, 0, 0, 0]);
}

#[test]
fn pcg64_seed_18_standard_normal_matches_numpy() {
    let mut rng = NumpyPcg64::from_seed(18);
    let expect = [
        -0.432_340_005_541_102_25,
        -1.130_096_771_890_423,
        0.673_843_657_927_979_7,
        -1.107_814_598_274_859_5,
        2.013_915_583_091_144,
        0.924_112_036_701_145_7,
        -0.359_262_938_083_582_3,
        0.570_515_731_275_182_6,
    ];
    for value in expect {
        assert_eq!(rng.standard_normal(), value);
    }
}

#[test]
fn ndtri_matches_scipy_samples() {
    assert_eq!(ndtri(0.5), 0.0);
    let y = ndtri(1e-10);
    assert!((y + 6.361_340_902_404_056).abs() < 1e-12);
}
