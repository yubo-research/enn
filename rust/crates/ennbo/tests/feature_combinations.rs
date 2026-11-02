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
        for mode in 0..3 {
            let projection = (mode != 0).then(|| {
                Box::new(Array2::from_shape_fn((x.ncols(), 2), |(i, j)| {
                    ((i + 1) as f64 * (j + 1) as f64).sin()
                }))
            });
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
                        semantic_projection: projection.clone(),
                        semantic_only: mode == 2,
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
    }
    assert_eq!(sequence, 27);
}

fn compare(
    config: ENNSurrogateConfig,
    x: &Array2<f64>,
    y: &Array2<f64>,
    reference_dir: std::path::PathBuf,
) {
    config.validate(x.ncols()).unwrap();
    let z = ennbo::semantic::features(
        &x.view(),
        config.semantic_projection.as_deref(),
        config.semantic_only,
    )
    .unwrap();
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

fn semantic_store(dir: &std::path::Path) -> EpistemicNearestNeighbors {
    EpistemicNearestNeighbors::new_attached(
        (
            ndarray::array![[2.], [5.]],
            ndarray::array![[1.], [4.]],
            None,
        ),
        ennbo::EnnLayout::new(dir.to_path_buf(), false).unwrap(),
        None,
        Vec::new(),
        ennbo::input_history::StoreAttachments {
            original_inputs: Some(ndarray::array![[1., 2., 3.], [4., 5., 6.]]),
            ..Default::default()
        },
    )
    .unwrap()
}

#[test]
fn semantic_only_originals_follow_disk_model_lifecycle() {
    let dir = tempfile::tempdir().unwrap();
    let mut model = semantic_store(dir.path());
    model
        .add_with_original(
            &ndarray::array![[8.]].view(),
            Some(&ndarray::array![[7., 8., 9.]].view()),
            &ndarray::array![[7.]].view(),
            None,
        )
        .unwrap();
    model.persist_index_to_disk().unwrap();
    drop(model);
    let model = EpistemicNearestNeighbors::open(dir.path()).unwrap();
    assert_eq!(
        model.original_rows_at(&[2, 0]).unwrap(),
        ndarray::array![[7., 8., 9.], [1., 2., 3.]]
    );
    assert!(model.original_rows_at(&[3]).is_err());
    drop(model);
    std::fs::OpenOptions::new()
        .write(true)
        .open(dir.path().join("original_inputs.bin"))
        .unwrap()
        .set_len(25)
        .unwrap();
    assert!(EpistemicNearestNeighbors::open(dir.path()).is_err());
}

#[test]
fn a_store_with_original_inputs_refuses_rows_without_them() {
    let dir = tempfile::tempdir().unwrap();
    let mut model = semantic_store(dir.path());
    let err = model
        .add(
            &ndarray::array![[8.]].view(),
            &ndarray::array![[7.]].view(),
            None,
        )
        .unwrap_err();
    assert!(err.to_string().contains("requires them"), "{err}");
    let short = model.add_with_original(
        &ndarray::array![[8.], [9.]].view(),
        Some(&ndarray::array![[7., 8., 9.]].view()),
        &ndarray::array![[7.], [8.]].view(),
        None,
    );
    assert!(short.is_err());
    assert_eq!(model.len(), 2);
}

#[test]
fn a_crash_after_the_original_row_and_before_the_model_row_is_recovered() {
    let dir = tempfile::tempdir().unwrap();
    let model = semantic_store(dir.path());
    model.persist_index_to_disk().unwrap();
    drop(model);
    let path = dir.path().join("original_inputs.bin");
    let mut grown = std::fs::read(&path).unwrap();
    grown[16] = 3;
    grown.extend_from_slice(&[0; 24]);
    std::fs::write(&path, grown).unwrap();
    let reopened = EpistemicNearestNeighbors::open(dir.path()).unwrap();
    assert_eq!(reopened.len(), 2);
    assert!(reopened.original_rows_at(&[2]).is_err());
    assert_eq!(
        reopened.original_rows_at(&[1]).unwrap(),
        ndarray::array![[4., 5., 6.]]
    );
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
    let model = semantic_store(dir.path());
    assert_eq!(model.len(), 2);
    drop(model);
    assert_eq!(EpistemicNearestNeighbors::open(dir.path()).unwrap().len(), 2);
}

fn unfinished_identity() -> tempfile::TempDir {
    let dir = tempfile::tempdir().unwrap();
    drop(semantic_store(dir.path()));
    let path = dir.path().join("store_identity.json");
    let text = std::fs::read_to_string(&path).unwrap();
    let unfinished = text.replace("\"committed\":true", "\"committed\":false");
    assert_ne!(unfinished, text);
    std::fs::write(&path, unfinished).unwrap();
    dir
}

#[test]
fn output_projection_blends_optimizer_predictions() {
    let dir = tempfile::tempdir().unwrap();
    let overrides = ennbo::ConfigOverrides {
        acquisition: Some(ennbo::AcquisitionConfig::Pareto),
        enn: Some(ennbo::EnnOverrides {
            fit: Some(ennbo::enn_overrides::FitOverride::Frozen),
            semantic_outputs: Some(ennbo::SemanticOutputs {
                loadings: ndarray::Array2::from_elem((2, 1), 0.5f64.sqrt()),
                alpha: Some(0.5),
            }),
            ..Default::default()
        }),
        morbo: Some(ennbo::morbo_override::MorboOverride {
            num_metrics: 2,
            alpha: 0.05,
            rescalarize: ennbo::Rescalarize::OnRestart,
        }),
        trust_region_kind: Some(ennbo::TrustRegionKind::Morbo),
        ..Default::default()
    };
    let mut optimizer = ennbo::create_optimizer_enn_with_overrides(
        ndarray::array![[0.0, 1.0], [0.0, 1.0]],
        ennbo::EnnLayout::new(dir.path(), false).unwrap(),
        Some(3),
        Some(4),
        0,
        Some(&overrides),
    )
    .unwrap();
    for _ in 0..4 {
        let x = optimizer.ask(2).unwrap();
        let y = x.map_axis(ndarray::Axis(1), |row| row.sum());
        let y = ndarray::stack(ndarray::Axis(1), &[y.view(), y.view()]).unwrap();
        optimizer.tell(&x.view(), &y.view(), None).unwrap();
    }
    assert!(optimizer.ask(1).unwrap().iter().all(|v| v.is_finite()));
}
