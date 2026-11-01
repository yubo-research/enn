use std::sync::Arc;

use crate::model::EpistemicNearestNeighbors;
use crate::output_blend::OutputBlend;
use crate::params::{ENNNormal, ENNParams, PosteriorFlags};
use crate::traits::PosteriorComputation;
use ndarray::{array, Array2, Array3, Axis, Ix2, Ix3};

fn model() -> crate::test_helpers::TestModel {
    let x = Array2::from_shape_fn((12, 2), |(i, j)| ((i * (j + 3)) % 7) as f64 * 0.3);
    let y = Array2::from_shape_fn((12, 3), |(i, j)| {
        ((i * (j + 2) * (j + 2)) % 11) as f64 * 0.1
    });
    crate::test_helpers::test_model(x, y, None, false)
}

fn blend(alpha: f64) -> Arc<OutputBlend> {
    Arc::new(
        OutputBlend::new(
            array![alpha, alpha, 0.0],
            Array2::from_elem((3, 3), 1.0 / 3.0),
            array![0.5, 1.0, 1.5],
            array![0.3, 0.6, 0.9],
        )
        .unwrap(),
    )
}

fn base() -> ENNParams {
    ENNParams::new(3, 1.0, 0.1).unwrap()
}

fn with(alpha: f64) -> ENNParams {
    base().with_output_blend(Some(blend(alpha)))
}

fn query() -> Array2<f64> {
    array![[0.1, 0.2], [0.7, 0.3]]
}

fn hand_mean(alpha: f64, mean: &Array2<f64>) -> Array2<f64> {
    let b = blend(alpha);
    let z = (mean - &b.center) / &b.scale;
    let mut out = z.clone();
    for mut row in out.rows_mut() {
        let total: f64 = row.sum();
        row[0] += alpha * (total / 3.0 - row[0]);
        row[1] += alpha * (total / 3.0 - row[1]);
    }
    out * &b.scale + &b.center
}

fn mu2(normal: &ENNNormal) -> Array2<f64> {
    normal.mu.clone().into_dimensionality::<Ix2>().unwrap()
}

fn assert_close(a: &Array2<f64>, b: &Array2<f64>) {
    assert_eq!(a.dim(), b.dim());
    assert!(
        a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < 1e-9),
        "{a:?} vs {b:?}"
    );
}

fn assert_matches_hand_operator(path: impl Fn(&ENNParams) -> Array2<f64>) {
    let plain = path(&base());
    assert_close(&path(&with(0.0)), &plain);
    let blended = path(&with(0.3));
    assert_close(&blended, &hand_mean(0.3, &plain));
    assert!((&blended - &plain).iter().any(|d| d.abs() > 1e-3));
}

#[test]
fn light_posterior_blends() {
    let m = model();
    let flags = PosteriorFlags::new();
    assert_matches_hand_operator(|p| mu2(&m.posterior(&query().view(), p, &flags).unwrap()));
}

#[test]
fn full_posterior_blends() {
    let m = model();
    let flags = PosteriorFlags::new().with_observation_noise(true);
    assert_matches_hand_operator(|p| mu2(&m.posterior(&query().view(), p, &flags).unwrap()));
}

#[test]
fn conditional_posterior_blends() {
    let m = model();
    let flags = PosteriorFlags::new();
    let x_whatif = array![[0.15, 0.25]];
    let y_whatif = array![[1.0, 2.0, 3.0]];
    assert_matches_hand_operator(|p| {
        mu2(&m
            .conditional_posterior(
                &x_whatif.view(),
                &y_whatif.view(),
                &query().view(),
                p,
                &flags,
            )
            .unwrap())
    });
}

fn first_draw(draws: Array3<f64>) -> Array2<f64> {
    draws.index_axis(Axis(2), 0).to_owned()
}

#[test]
fn function_draws_blend() {
    let m = model();
    let flags = PosteriorFlags::new();
    assert_matches_hand_operator(|p| {
        first_draw(
            m.posterior_function_draw(&query().view(), p, &[7, 8], &flags)
                .unwrap()
                .0,
        )
    });
}

#[test]
fn conditional_function_draws_blend() {
    let m = model();
    let flags = PosteriorFlags::new();
    let x_whatif = array![[0.15, 0.25]];
    let y_whatif = array![[1.0, 2.0, 3.0]];
    assert_matches_hand_operator(|p| {
        first_draw(
            m.conditional_posterior_function_draw(
                &x_whatif.view(),
                &y_whatif.view(),
                &query().view(),
                p,
                &[7],
                &flags,
            )
            .unwrap()
            .0,
        )
    });
}

fn batch_entries(m: &EpistemicNearestNeighbors, paramss: &[ENNParams], flags: &PosteriorFlags) {
    let out = PosteriorComputation::batch_posterior(m, &query().view(), paramss, flags).unwrap();
    let mu = out.mu.into_dimensionality::<Ix3>().unwrap();
    for (i, params) in paramss.iter().enumerate() {
        let one = m.posterior(&query().view(), params, flags).unwrap();
        assert_close(&mu.index_axis(Axis(0), i).to_owned(), &mu2(&one));
    }
}

#[test]
fn batch_applies_each_entrys_own_blend_on_every_path() {
    let m = model();
    let shared_k = vec![with(0.1), base(), with(0.9)];
    batch_entries(&m, &shared_k, &PosteriorFlags::new());
    batch_entries(
        &m,
        &shared_k,
        &PosteriorFlags::new().with_observation_noise(true),
    );
    let mixed_k = vec![
        with(0.1),
        ENNParams::new(5, 1.0, 0.1)
            .unwrap()
            .with_output_blend(Some(blend(0.9))),
    ];
    batch_entries(&m, &mixed_k, &PosteriorFlags::new());
    batch_entries(
        &m,
        &mixed_k,
        &PosteriorFlags::new().with_observation_noise(true),
    );
}

#[test]
fn subsample_loglik_scores_blended_predictions() {
    let m = model();
    let (x, y, _) = m
        .rows()
        .train_rows_at(&(0..12).collect::<Vec<_>>())
        .unwrap();
    let paramss = vec![base(), with(0.9)];
    let score = |p: &[ENNParams]| {
        use rand::SeedableRng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        crate::fit::subsample_loglik(&m, &x.view(), &y.view(), p, 12, &mut rng, None).unwrap()
    };
    let together = score(&paramss);
    assert_ne!(together[0], together[1]);
    assert_eq!(together[1], score(&[paramss[1].clone()])[0]);
}

#[test]
fn blends_with_the_wrong_output_count_are_rejected() {
    let m = model();
    let wrong = base().with_output_blend(Some(Arc::new(
        OutputBlend::new(array![0.1], Array2::eye(1), array![0.0], array![1.0]).unwrap(),
    )));
    let flags = PosteriorFlags::new();
    assert!(m.posterior(&query().view(), &wrong, &flags).is_err());
    assert!(m
        .posterior_function_draw(&query().view(), &wrong, &[1], &flags)
        .is_err());
    assert!(m
        .batch_posterior(&query().view(), std::slice::from_ref(&wrong), &flags)
        .is_err());
}

#[test]
fn y_bounds_are_naturalized_after_blending() {
    let x = array![[0.0], [1.0], [2.0]];
    let y = array![[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]];
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let dir = tempfile::tempdir().unwrap();
    let m = EpistemicNearestNeighbors::new_with_storage(
        x,
        y,
        None,
        crate::layout::EnnLayout::new(dir.path(), false).unwrap(),
        Some(bounds),
    )
    .unwrap();
    let identity = Arc::new(
        OutputBlend::new(
            array![1.0, 1.0],
            Array2::eye(2),
            array![0.0, 0.0],
            array![1.0, 1.0],
        )
        .unwrap(),
    );
    let flags = PosteriorFlags::new();
    let q = array![[0.5]];
    let plain = base();
    let blended = base().with_output_blend(Some(identity));
    let a = m.posterior(&q.view(), &plain, &flags).unwrap();
    let b = m.posterior(&q.view(), &blended, &flags).unwrap();
    assert_close(&mu2(&a), &mu2(&b));
    assert!(a.mu.iter().all(|v| (0.0..=1.0).contains(v)));
}
