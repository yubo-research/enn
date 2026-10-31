
use ndarray::{Array3, ArrayView1, ArrayView2, Axis};

use super::light::{
    compute_posterior_light, fill_neighbor_means, fill_neighbor_weights, fuse_neighbors_to_mu_se,
};
use super::small_flat::SmallFlatNeighbors;
use crate::error::ENNError;
use crate::model::EpistemicNearestNeighbors;
use crate::params::{ENNNormal, ENNParams, PosteriorFlags};

pub(super) fn compute_mixed_posterior_light(
    model: &EpistemicNearestNeighbors,
    x: &ArrayView2<f64>,
    paramss: &[ENNParams],
    flags: &PosteriorFlags,
) -> Result<ENNNormal, ENNError> {
    let shape = (paramss.len(), x.nrows(), model.num_metrics());
    let mut mu = Array3::zeros(shape);
    let mut se = Array3::zeros(shape);
    let mut se_epi = Array3::zeros(shape);
    let mut se_ale = Array3::zeros(shape);
    let max_k = paramss
        .iter()
        .map(|p| p.k_num_neighbors as usize)
        .max()
        .unwrap_or(0);
    let neighbors = (!flags.exclude_nearest)
        .then(|| SmallFlatNeighbors::new(model, x, max_k))
        .flatten();
    for (i, params) in paramss.iter().enumerate() {
        let shared = neighbors
            .as_ref()
            .and_then(|cache| cache.neighbors(params.k_num_neighbors as usize));
        let (mean, sd, epi, ale, _) = if let Some((distances, indices)) = shared {
            fuse_neighbors_to_mu_se(model, &distances.view(), &indices.view(), params)
        } else {
            compute_posterior_light(model, x, params, flags)?
        };
        mu.index_axis_mut(Axis(0), i).assign(&mean);
        se.index_axis_mut(Axis(0), i).assign(&sd);
        se_epi.index_axis_mut(Axis(0), i).assign(&epi);
        se_ale.index_axis_mut(Axis(0), i).assign(&ale);
    }
    Ok(ENNNormal::new(
        mu.into_dyn(),
        se.into_dyn(),
        se_epi.into_dyn(),
        se_ale.into_dyn(),
        None,
    ))
}

pub fn small_flat_selection_means(
    train_x: &ArrayView2<f64>,
    train_y: &ArrayView2<f64>,
    query: &ArrayView2<f64>,
    neighbors: &[usize],
) -> Option<Array3<f64>> {
    if train_x.nrows() != train_y.nrows()
        || train_y.ncols() == 0
        || train_y.iter().any(|v| !v.is_finite())
        || neighbors.contains(&0)
    {
        return None;
    }
    let max_k = (*neighbors.iter().max()?).min(train_x.nrows());
    let cache = SmallFlatNeighbors::from_arrays(train_x, query, None, max_k)?;
    let mut means = Array3::zeros((neighbors.len(), query.nrows(), train_y.ncols()));
    let mut distances = vec![0.0; max_k];
    let mut indices = vec![0; max_k];
    let mut weights = vec![0.0; max_k];
    for (i, &count) in neighbors.iter().enumerate() {
        let k = count.min(train_x.nrows());
        let params = ENNParams {
            k_num_neighbors: k as i32,
            epistemic_variance_scale: 1.0,
            aleatoric_variance_scale: 0.0,
        };
        let mut output = means.index_axis_mut(Axis(0), i);
        for q in 0..query.nrows() {
            for (j, &(distance, index, _)) in cache.certified_row(q, k)?.iter().enumerate() {
                distances[j] = distance;
                indices[j] = index;
            }
            let inv_norm = fill_neighbor_weights(
                ArrayView1::from(&distances[..k]),
                &params,
                &mut weights[..k],
            );
            fill_neighbor_means(
                *train_y,
                ArrayView1::from(&indices[..k]),
                &weights[..k],
                inv_norm,
                output.row_mut(q),
            );
        }
    }
    Some(means)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::IndexDriver;
    use ndarray::array;

    #[test]
    fn selection_means_match_full_posterior_without_temporary_index() {
        let x = ndarray::Array2::from_shape_fn((100, 3), |(i, j)| {
            ((i * 7 + j * 13) as f64 * 0.3).sin()
        });
        let y = ndarray::Array2::from_shape_fn((100, 2), |(i, j)| (i + j) as f64);
        let query = array![[0.03, 0.3, 1.0], [0.1, 0.01, 0.3]];
        let means =
            small_flat_selection_means(&x.view(), &y.view(), &query.view(), &[3, 10, 30]).unwrap();
        let model = EpistemicNearestNeighbors::new(x, y, None, false, IndexDriver::Flat).unwrap();
        for (i, k) in [3, 10, 30].into_iter().enumerate() {
            let posterior = model
                .posterior(
                    &query.view(),
                    &ENNParams::new(k, 1.0, 0.0).unwrap(),
                    &PosteriorFlags::new(),
                )
                .unwrap();
            assert_eq!(
                means.index_axis(Axis(0), i).to_owned().into_dyn(),
                posterior.mu
            );
        }
    }

    #[test]
    fn selection_means_preserve_strided_scalar_and_multiple_targets() {
        let x =
            ndarray::Array2::from_shape_fn((30, 3), |(i, j)| ((i * 7 + j * 13) as f64 * 0.3).sin());
        let y = ndarray::Array2::from_shape_fn((30, 12), |(i, j)| (i + j) as f64);
        let query = array![[0.03, 0.3, 1.0]];
        for metrics in [1, 6] {
            let target = y.slice(ndarray::s![.., ..metrics * 2;2]);
            let means = small_flat_selection_means(&x.view(), &target, &query.view(), &[3, 10, 30])
                .unwrap();
            let model = EpistemicNearestNeighbors::new(
                x.clone(),
                target.to_owned(),
                None,
                false,
                IndexDriver::Flat,
            )
            .unwrap();
            for (i, k) in [3, 10, 30].into_iter().enumerate() {
                let expected = model
                    .posterior(
                        &query.view(),
                        &ENNParams::new(k, 1.0, 0.0).unwrap(),
                        &PosteriorFlags::new(),
                    )
                    .unwrap();
                assert_eq!(
                    means.index_axis(Axis(0), i).to_owned().into_dyn(),
                    expected.mu
                );
            }
        }
    }

    #[test]
    fn selection_means_decline_invalid_data_and_float32_collisions() {
        let x = ndarray::Array2::from_shape_fn((100, 1), |(i, _)| 1.0 + i as f64 * 1e-10);
        let y = ndarray::Array2::ones((100, 2));
        let q = array![[2.0]];
        assert!(
            small_flat_selection_means(&x.view(), &y.view(), &q.view(), &[3, 10, 30]).is_none()
        );
        assert!(small_flat_selection_means(&x.view(), &y.view(), &q.view(), &[]).is_none());
        assert!(small_flat_selection_means(&x.view(), &y.view(), &q.view(), &[0]).is_none());
        assert!(small_flat_selection_means(&q.view(), &y.view(), &q.view(), &[3]).is_none());
        assert!(small_flat_selection_means(
            &x.view(),
            &y.mapv(|_| f64::NAN).view(),
            &q.view(),
            &[3]
        )
        .is_none());
    }
}
