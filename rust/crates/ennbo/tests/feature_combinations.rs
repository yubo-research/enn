//! Native Rust feature composition has the same behavior as explicit model inputs.
use ennbo::categorical_encoder::CatEncoder;
use ennbo::categorical_plan::{CatEncoding, CatOptions};
use ennbo::surrogate::{ENNSurrogate, ENNSurrogateConfig, Surrogate};
use ennbo::{EnnLayout, EpistemicNearestNeighbors, FitSamples, NeighborCount};
use ndarray::{s, Array2};
use rand::{rngs::StdRng, SeedableRng};

#[test]
fn native_feature_combinations_match_explicit_model_features() {
    let root = tempfile::tempdir().unwrap();
    let raw = Array2::from_shape_fn((18, 3), |(i, j)| {
        if j == 2 {
            (i % 3) as f64
        } else {
            ((i * (j + 3) + j) % 19) as f64 / 19.0
        }
    });
    let y = raw.slice(s![.., ..1]).mapv(|v| (v * 3.0).sin());
    let mut sequence = 0;
    for encoding in [
        None,
        Some(CatEncoding::OneHot),
        Some(CatEncoding::Rademacher),
    ] {
        let x = match encoding {
            Some(encoding) => CatEncoder::new(
                vec![false, false, true],
                Some(vec![None, None, Some(3)]),
                CatOptions {
                    encoding,
                    ..Default::default()
                },
            )
            .unwrap()
            .encode(&raw.view())
            .unwrap(),
            None => raw.clone(),
        };
        for weighting in [false, true] {
            let ties = if weighting {
                vec![vec![], vec![vec![0, 1]]]
            } else {
                vec![vec![]]
            };
            for tied in ties {
                let dir = root.path().join(format!("surrogate-{sequence}"));
                let layout = EnnLayout::new(dir, weighting).unwrap();
                let config = ENNSurrogateConfig {
                    tied_dims: tied.clone(),
                    fit_samples: FitSamples::Frozen,
                    k: NeighborCount::new(3).unwrap(),
                    ..ENNSurrogateConfig::new(layout)
                };
                compare(
                    config,
                    &x,
                    &y,
                    root.path().join(format!("reference-{sequence}")),
                );
                sequence += 1;
            }
        }
    }
    assert_eq!(sequence, 9);
}

fn compare(
    config: ENNSurrogateConfig,
    x: &Array2<f64>,
    y: &Array2<f64>,
    reference_dir: std::path::PathBuf,
) {
    config.validate(x.ncols()).unwrap();
    let z = x;
    let reference_layout = ennbo::EnnLayout::new(reference_dir, config.layout.weighting()).unwrap();
    let mut reference = EpistemicNearestNeighbors::new_configured(
        z.slice(s![..12, ..]).to_owned(),
        y.slice(s![..12, ..]).to_owned(),
        None,
        reference_layout,
        None,
        config.tied_dims.clone(),
    )
    .unwrap();
    assert_eq!(reference.metric_learning_auto(), config.layout.weighting());
    let mut surrogate = ENNSurrogate::new(config);
    let mut rng = StdRng::seed_from_u64(4);
    surrogate
        .fit(
            &x.slice(s![..12, ..]),
            &y.slice(s![..12, ..]),
            None,
            &mut rng,
        )
        .unwrap();
    surrogate
        .fit_append(
            &x.slice(s![12.., ..]),
            &y.slice(s![12.., ..]),
            None,
            &mut rng,
        )
        .unwrap();
    reference
        .add(&z.slice(s![12.., ..]), &y.slice(s![12.., ..]), None)
        .unwrap();
    let actual = surrogate.predict(&x.slice(s![..3, ..])).unwrap();
    let expected = reference
        .posterior(
            &z.slice(s![..3, ..]),
            &ennbo::ENNParams::new(3, 1.0, 0.0).unwrap(),
            &ennbo::PosteriorFlags::default(),
        )
        .unwrap();
    assert_eq!(actual.mu.into_dyn(), expected.mu);
    let recovered = surrogate.observations_x().unwrap().unwrap();
    assert_eq!(recovered.dim(), x.dim());
    assert!((&recovered - x).iter().all(|v| v.abs() < 1e-14));
}

#[test]
fn integer_category_ids_roundtrip_without_float_rounding() {
    let x = ndarray::array![[1_i64 << 53], [(1_i64 << 53) + 1], [i64::MIN], [i64::MAX]];
    let encoder = CatEncoder::new(
        vec![true],
        None,
        CatOptions {
            encoding: CatEncoding::Rademacher,
            code_dim: Some(64),
            ..Default::default()
        },
    )
    .unwrap();
    let codes = encoder.encode_i64(&x.view()).unwrap();
    use ennbo::categorical_encoder::InputValue;
    let mixed = Array2::from_shape_fn((x.nrows(), 2), |(i, j)| {
        if j == 0 {
            InputValue::Number(0.25)
        } else {
            InputValue::Category(x[[i, 0]])
        }
    });
    let mixed_encoder = CatEncoder::new(
        vec![false, true],
        None,
        CatOptions {
            encoding: CatEncoding::Rademacher,
            code_dim: Some(64),
            ..Default::default()
        },
    )
    .unwrap();
    let mixed_codes = mixed_encoder.encode_mixed(&mixed.view()).unwrap();
    assert_eq!(mixed_codes.slice(s![.., 1..]), codes);

    assert_ne!(codes.row(0), codes.row(1));
    assert_eq!(
        encoder
            .snap(&codes.view(), &[x.column(0).to_vec()])
            .unwrap(),
        x
    );
}

fn plain_store(dir: &std::path::Path) -> EpistemicNearestNeighbors {
    EpistemicNearestNeighbors::new_configured(
        ndarray::array![[2.], [5.]],
        ndarray::array![[1.], [4.]],
        None,
        ennbo::EnnLayout::new(dir.to_path_buf(), false).unwrap(),
        None,
        Vec::new(),
    )
    .unwrap()
}

#[test]
fn a_failed_store_creation_leaves_a_directory_that_can_be_created_again() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("store_identity.json"),
        std::fs::read_to_string(unfinished_identity().path().join("store_identity.json")).unwrap(),
    )
    .unwrap();
    let err = EpistemicNearestNeighbors::open(dir.path()).err().unwrap();
    assert!(err.to_string().contains("never finished"), "{err}");
    let model = plain_store(dir.path());
    assert_eq!(model.len(), 2);
    drop(model);
    assert_eq!(
        EpistemicNearestNeighbors::open(dir.path()).unwrap().len(),
        2
    );
}

fn unfinished_identity() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    drop(plain_store(dir.path()));
    let path = dir.path().join("store_identity.json");
    let text = std::fs::read_to_string(&path).unwrap();
    let unfinished = text.replace("\"committed\":true", "\"committed\":false");
    assert_ne!(unfinished, text);
    std::fs::write(&path, unfinished).unwrap();
    dir
}
