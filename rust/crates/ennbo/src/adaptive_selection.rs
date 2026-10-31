//! Held-out candidate selection, continuous output mixing, and epoch calibration.
use crate::adaptive_basis::{Basis, Metadata};
use crate::adaptive_geometry::invalid;
use crate::adaptive_predictor::{Choice, Epoch, Predictor};
use crate::adaptive_scales::bake;
use crate::error::ENNError;
use crate::params::{ENNParams, PosteriorFlags};
use ndarray::{s, Array1, Array2, Array3, Axis, Ix3};

/// Least-squares mixing on selection rows only; ties choose zero sharing.
pub(crate) fn mixing(mean: &Array2<f64>, target: &Array2<f64>, projector: &Array2<f64>) -> f64 {
    let delta = mean.dot(projector) - mean;
    let norm = delta.iter().map(|v| v * v).sum::<f64>();
    if norm == 0.0 {
        return 0.0;
    }
    (delta
        .iter()
        .zip((target - mean).iter())
        .map(|(d, r)| d * r)
        .sum::<f64>()
        / norm)
        .clamp(0.0, 1.0)
}

fn selection_means(
    basis: &Basis,
    x: &Array2<f64>,
    y: &Array2<f64>,
    query: &Array2<f64>,
    choice: &Choice,
    trend: bool,
) -> Result<Array3<f64>, ENNError> {
    let mut train = basis.features(x, &choice.geometry, choice.weight)?;
    let mut query_features = basis.features(query, &choice.geometry, choice.weight)?;
    if choice.geometry == "scales" {
        bake(&mut train, &basis.column_scale);
        bake(&mut query_features, &basis.column_scale);
    }
    let target = basis.target(x, y, trend);
    let neighbors = [3, 10, 30];
    let mut means = match crate::posterior::small_flat_selection_means(
        &train.view(),
        &target.view(),
        &query_features.view(),
        &neighbors,
    ) {
        Some(means) => means,
        None => {
            let model = crate::model::EpistemicNearestNeighbors::new(
                train,
                target,
                None,
                false,
                crate::index::IndexDriver::Flat,
            )?;
            let params = neighbors
                .iter()
                .map(|k| ENNParams::new((*k).min(x.nrows()) as i32, 1.0, 0.0).map_err(invalid))
                .collect::<Result<Vec<_>, _>>()?;
            model
                .batch_posterior(&query_features.view(), &params, &PosteriorFlags::default())?
                .mu
                .into_dimensionality::<Ix3>()
                .map_err(invalid)?
        }
    };
    if trend {
        let trend = basis.trend(query);
        for mut mean in means.axis_iter_mut(Axis(0)) {
            mean += &trend;
        }
    }
    for (j, constant) in basis.constant.iter().enumerate() {
        if *constant {
            means.slice_mut(s![.., .., j]).fill(0.0);
        }
    }
    Ok(means)
}

fn geometries(
    metadata: &Metadata,
    adaptation: &str,
    weight: Option<f64>,
    scaled: bool,
) -> Vec<(&'static str, f64)> {
    let mut choices = vec![("fixed", 0.0)];
    if adaptation == "outputs" || adaptation == "none" {
        return choices;
    }
    if metadata.input.is_some() {
        choices.insert(0, ("raw", 0.0));
    }
    if scaled && metadata.input.as_ref().is_some_and(|p| p.ncols() > 0) {
        choices.insert(choices.len().min(2), ("scales", 1.0));
    }
    for geometry in ["linear", "quadratic"] {
        for weight in weight.map_or_else(|| vec![3.0, 30.0], |v| vec![v]) {
            choices.push((geometry, weight));
        }
    }
    choices
}

fn select(
    x: &Array2<f64>,
    y: &Array2<f64>,
    metadata: &Metadata,
    trend: bool,
    adaptation: &str,
    weight: Option<f64>,
    column_scale: Option<&Array1<f64>>,
) -> Result<Choice, ENNError> {
    let end = x.nrows() * 3 / 5;
    let train_x = x.slice(s![..end, ..]).to_owned();
    let train_y = y.slice(s![..end, ..]).to_owned();
    let query = x.slice(s![end..x.nrows() * 4 / 5, ..]).to_owned();
    let mut basis = Basis::new(&train_x, &train_y, metadata)?;
    if let Some(scale) = column_scale {
        basis.column_scale = scale.clone();
    }
    let target =
        (y.slice(s![end..x.nrows() * 4 / 5, ..]).to_owned() - &basis.y_mean) / &basis.y_scale;
    let mut best = None;
    let mut best_loss = f64::INFINITY;
    for (geometry, weight) in geometries(metadata, adaptation, weight, column_scale.is_some()) {
        let mut choice = Choice {
            geometry: geometry.into(),
            weight,
            output: "independent".into(),
            neighbors: 3,
            alpha: 0.0,
        };
        let means = selection_means(&basis, &train_x, &train_y, &query, &choice, trend)?;
        for (i, mean) in means.axis_iter(Axis(0)).enumerate() {
            for (name, projector) in &basis.operators {
                if !allowed_output(name, adaptation, metadata.output.is_some()) {
                    continue;
                }
                let alpha = match *name {
                    "independent" => 0.0,
                    "lowrank" => 1.0,
                    _ => mixing(&mean.to_owned(), &target, projector),
                };
                let prediction = &mean + &((mean.dot(projector) - mean) * alpha);
                let loss = (&prediction - &target).mapv(|v| v * v).sum();
                if loss < best_loss {
                    best_loss = loss;
                    choice.output = (*name).into();
                    choice.neighbors = [3, 10, 30][i];
                    choice.alpha = alpha;
                    best = Some(choice.clone());
                }
            }
        }
    }
    best.ok_or_else(|| invalid("adaptive candidates produced no finite score"))
}

fn allowed_output(name: &str, adaptation: &str, has_metadata: bool) -> bool {
    match adaptation {
        "inputs" => name == "independent",
        "none" => {
            name == if has_metadata {
                "semantic"
            } else {
                "independent"
            }
        }
        _ => true,
    }
}

pub(crate) fn fit_epoch(
    x: &Array2<f64>,
    y: &Array2<f64>,
    metadata: &Metadata,
    trend: bool,
    adaptation: &str,
    weight: Option<f64>,
    column_scale: Option<&Array1<f64>>,
) -> Result<Epoch, ENNError> {
    let choice = select(x, y, metadata, trend, adaptation, weight, column_scale)?;
    let end = x.nrows() * 4 / 5;
    let predictor = Predictor::new(
        &x.slice(s![..end, ..]).to_owned(),
        &y.slice(s![..end, ..]).to_owned(),
        metadata,
        choice,
        trend,
        column_scale,
    )?;
    let (mu, se) = predictor.posterior(&x.slice(s![end.., ..]).to_owned())?;
    let floor = predictor.basis.y_scale.mapv(|v| (v * 0.03).max(1e-12));
    let errors = y.slice(s![end.., ..]).to_owned() - mu;
    let ratios = Array2::from_shape_fn(errors.dim(), |(i, j)| {
        errors[[i, j]].powi(2) / se[[i, j]].powi(2).max(floor[j].powi(2))
    });
    let calibration = ratios.mean_axis(Axis(0)).unwrap().mapv(|v| v.max(1e-12));
    Ok(Epoch {
        predictor,
        floor,
        calibration,
    })
}
