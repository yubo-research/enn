//! Native Rust feature composition has the same behavior as explicit model inputs.
use ennbo::categorical_encoder::CatEncoder;
use ennbo::categorical_plan::{CatEncoding, CatOptions};
use ennbo::surrogate::{ENNSurrogate, ENNSurrogateConfig, Surrogate};
use ennbo::{EnnLayout, EpistemicNearestNeighbors, FitSamples, IndexDriver, NeighborCount};
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
                for tied in [vec![], vec![vec![0, 1]]] {
                    for driver in [IndexDriver::Flat, IndexDriver::BpAnnDisk] {
                        let dir = (driver == IndexDriver::BpAnnDisk)
                            .then(|| root.path().join(format!("surrogate-{sequence}")));
                        let layout = EnnLayout::new(driver, dir, weighting).unwrap();
                        let config = ENNSurrogateConfig {
                            layout,
                            semantic_projection: projection.clone(),
                            semantic_only: mode == 2,
                            tied_dims: tied.clone(),
                            fit_samples: FitSamples::Frozen,
                            k: NeighborCount::new(3).unwrap(),
                            ..Default::default()
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
    }
    assert_eq!(sequence, 72);
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
    let dir = (config.layout.index_driver() == IndexDriver::BpAnnDisk).then_some(reference_dir);
    let reference_layout =
        EnnLayout::new(config.layout.index_driver(), dir, config.layout.weighting()).unwrap();
    let mut reference = EpistemicNearestNeighbors::new_configured(
        z.slice(s![..12, ..]).to_owned(),
        y.slice(s![..12, ..]).to_owned(),
        None,
        reference_layout,
        None,
        config.tied_dims.clone(),
    )
    .unwrap();
    assert_eq!(
        reference.metric_learning_auto(),
        config.layout.index_driver() == IndexDriver::BpAnnDisk && config.layout.weighting()
    );
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

#[test]
fn semantic_only_originals_follow_disk_model_lifecycle() {
    use ennbo::input_history::InputHistory;
    let dir = tempfile::tempdir().unwrap();
    let x = ndarray::array![[1., 2., 3.], [4., 5., 6.]];
    let features = ndarray::array![[2.], [5.]];
    let y = ndarray::array![[1.], [4.]];
    let layout = EnnLayout::disk(dir.path().to_path_buf(), false);
    let mut model =
        EpistemicNearestNeighbors::new_with_storage(features, y, None, layout.clone(), None)
            .unwrap();
    let mut history = InputHistory::for_model(x.clone(), &model).unwrap();
    model
        .add(
            &ndarray::array![[8.]].view(),
            &ndarray::array![[7.]].view(),
            None,
        )
        .unwrap();
    history
        .append(&ndarray::array![[7., 8., 9.]].view())
        .unwrap();
    model.persist_index_to_disk().unwrap();
    drop(history);
    drop(model);
    let model = EpistemicNearestNeighbors::new_with_storage(
        Array2::zeros((0, 1)),
        Array2::zeros((0, 1)),
        None,
        layout,
        None,
    )
    .unwrap();
    let history = InputHistory::for_model(Array2::zeros((0, 3)), &model).unwrap();
    assert_eq!(
        history.gather(&[2, 0]).unwrap(),
        ndarray::array![[7., 8., 9.], [1., 2., 3.]]
    );
    assert!(history.gather(&[3]).is_err());
    drop(history);
    std::fs::OpenOptions::new()
        .write(true)
        .open(dir.path().join("original_inputs.bin"))
        .unwrap()
        .set_len(25)
        .unwrap();
    assert!(InputHistory::for_model(Array2::zeros((0, 3)), &model).is_err());
}
