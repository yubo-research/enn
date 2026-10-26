"""One-hot labels must not change exact disk neighbors or calibrated predictions."""

import numpy as np

from enn import EpistemicNearestNeighbors
from enn.enn.cat_codes import CatEncoder, CatEncoding, DimKind
from enn.enn.enn_fitter import ENNStatefulFitter
from enn.enn.enn_params import PosteriorFlags
from enn.turbo.config.enn_index_driver import ENNIndexDriver
from enn.turbo.config.enn_surrogate_config import ENNStorage
from enn.turbo.config.enn_x_scaling import ENNMetricLearning


def _sparse_categories():
    rng = np.random.default_rng(4)
    categories = np.repeat(np.arange(128), 2)
    raw = np.column_stack([rng.random((256, 5)), categories])
    query = np.column_stack([rng.random((256, 5)), categories])
    y = rng.normal(size=128)[categories, None]
    y += 0.05 * rng.normal(size=y.shape)
    encoder = CatEncoder(
        [DimKind.CONTINUOUS] * 5 + [DimKind.CATEGORICAL],
        [None] * 5 + [128],
        encoding=CatEncoding.ONE_HOT,
    )
    return encoder, raw, query, y


def _model(x, y, path):
    return EpistemicNearestNeighbors(
        x,
        y,
        metric_learning=ENNMetricLearning.AUTO,
        index_driver=ENNIndexDriver.BPANN_DISK,
        enn_storage=ENNStorage.DISK,
        work_dir=path,
    )


def test_fixed_metric_onehot_neighbors_match_exact_after_relabeling(tmp_path):
    encoder, raw, query, y = _sparse_categories()
    weights = np.r_[np.full(5, 7e-5), np.full(128, 6.0)]
    expected = None
    for label in range(2):
        x, q = encoder.encode(raw), encoder.encode(query)
        model = _model(x, y, tmp_path / str(label))
        model.metric.set_weights(weights)
        exact = np.argsort(((x[None] - q[:, None]) ** 2) @ weights, axis=1)
        if expected is None:
            expected = exact[:, :10]
        np.testing.assert_array_equal(exact[:, :10], expected)
        actual = np.array([model.neighbors(row, 10) for row in q])
        np.testing.assert_array_equal(actual, expected)
        permutation = np.random.default_rng(505).permutation(128)
        raw[:, -1] = permutation[raw[:, -1].astype(int)]
        query[:, -1] = permutation[query[:, -1].astype(int)]


def test_auto_affine_onehot_predictions_survive_relabeling(tmp_path):
    encoder, raw, query, y = _sparse_categories()
    results = []
    neighbors = []
    for label in range(2):
        x, q = encoder.encode(raw), encoder.encode(query)
        model = _model(x, y, tmp_path / str(label))
        fitter = ENNStatefulFitter(k=10, rng=np.random.default_rng([404, 4]))
        fitter.tell(x, y)
        params = fitter.ask(
            model,
            num_fit_candidates=30,
            num_fit_samples=256,
            affine_calibrate=True,
        )
        assert model.metric.uses_learned_metric
        neighbors.append(np.array([model.neighbors(row, 10) for row in q]))
        posterior = fitter.affine_calibrator.apply(
            model.posterior(
                q, params=params, flags=PosteriorFlags(observation_noise=True)
            )
        )
        results.append((posterior.mu, posterior.se**2))
        permutation = np.random.default_rng(505).permutation(128)
        raw[:, -1] = permutation[raw[:, -1].astype(int)]
        query[:, -1] = permutation[query[:, -1].astype(int)]
    np.testing.assert_array_equal(neighbors[0], neighbors[1])
    np.testing.assert_allclose(results[0], results[1], rtol=0, atol=1e-10)
