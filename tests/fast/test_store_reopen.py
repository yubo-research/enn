import subprocess
import sys
import textwrap

import numpy as np
import pytest

from enn import ENNStatefulFitter, EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams
from enn.enn.mbpann import MBPANNMetric

PARAMS = ENNParams(
    k_num_neighbors=5, epistemic_variance_scale=1.0, aleatoric_variance_scale=0.1
)


def _sparse_rows(n, seed=0, num_dim=8):
    x = np.random.default_rng(seed).random((n, num_dim))
    return x, np.sin(6 * x[:, :1])


def _rmse(model, x, y):
    return float(np.sqrt(np.mean((model.posterior(x, params=PARAMS).mu - y) ** 2)))


def _plain_store(work_dir, num_dim=3, **kwargs):
    x = np.random.default_rng(0).random((30, num_dim))
    model = EpistemicNearestNeighbors(
        x, x.sum(axis=1, keepdims=True), work_dir=work_dir, **kwargs
    )
    model.persist_index_to_disk()
    del model


def test_reopen_keeps_learned_metric_and_accuracy(tmp_path):
    x, y = _sparse_rows(2000)
    x_test, y_test = _sparse_rows(200, seed=1)
    live = EpistemicNearestNeighbors(x, y, work_dir=tmp_path / "w", weighting=True)
    live_rmse = _rmse(live, x_test, y_test)
    live_scale = live._x_scale.copy()
    live_weights = live.metric.weights.copy()
    assert live_scale.max() / live_scale.min() > 10
    del live
    reopened = EpistemicNearestNeighbors.open(tmp_path / "w")
    np.testing.assert_allclose(reopened._x_scale, live_scale, rtol=1e-12)
    np.testing.assert_allclose(reopened.metric.weights, live_weights)
    assert _rmse(reopened, x_test, y_test) < 1.5 * live_rmse


def test_reopen_continues_the_metric_stream(tmp_path):
    x, y = _sparse_rows(1500)
    closed = EpistemicNearestNeighbors(
        x[:1000], y[:1000], work_dir=tmp_path / "c", weighting=True
    )
    never_closed = EpistemicNearestNeighbors(
        x[:1000], y[:1000], work_dir=tmp_path / "n", weighting=True
    )
    del closed
    closed = EpistemicNearestNeighbors.open(tmp_path / "c")
    for start in range(1000, 1500, 100):
        closed.add(x[start : start + 100], y[start : start + 100])
        never_closed.add(x[start : start + 100], y[start : start + 100])
        del closed
        closed = EpistemicNearestNeighbors.open(tmp_path / "c")
    assert closed.metric.num_seen == never_closed.metric.num_seen == 1500
    assert closed.metric.num_refits == never_closed.metric.num_refits
    assert np.allclose(closed._x_scale, never_closed._x_scale)


def test_reopen_restores_metric_policy_and_sample(tmp_path):
    x, y = _sparse_rows(2000)
    model = EpistemicNearestNeighbors(
        x[:1], y[:1], work_dir=tmp_path / "w", weighting=True
    )
    MBPANNMetric(model, seed=7, reservoir_capacity=500, rebuild_drift=0.3)
    model.add(x[1:], y[1:])
    seen = model.metric.num_seen
    gain = model.metric.heldout_gain
    del model
    reopened = EpistemicNearestNeighbors.open(tmp_path / "w")
    assert reopened.metric.num_seen == seen == 2000
    assert reopened.metric.heldout_gain == gain
    MBPANNMetric(reopened, seed=7, reservoir_capacity=500, rebuild_drift=0.3)


@pytest.mark.parametrize(
    ("num_dim", "num_outputs", "weighting"),
    [(5, 1, False), (3, 4, False), (3, 1, True), (3, 1, False)],
)
def test_constructor_rejects_any_existing_store(
    tmp_path, num_dim, num_outputs, weighting
):
    _plain_store(tmp_path / "w")
    with pytest.raises(ValueError, match=r"EpistemicNearestNeighbors\.open"):
        EpistemicNearestNeighbors(
            np.zeros((0, num_dim)),
            np.zeros((0, num_outputs)),
            work_dir=tmp_path / "w",
            weighting=weighting,
        )
    assert len(EpistemicNearestNeighbors.open(tmp_path / "w")) == 30


def test_an_empty_model_in_a_fresh_directory_is_legal(tmp_path):
    model = EpistemicNearestNeighbors(
        np.zeros((0, 3)), np.zeros((0, 1)), work_dir=tmp_path / "fresh"
    )
    assert len(model) == 0


def test_open_on_an_empty_directory_is_a_clear_error(tmp_path):
    with pytest.raises(ValueError, match=r"store_identity\.json"):
        EpistemicNearestNeighbors.open(tmp_path)


def test_store_without_an_identity_file_is_not_supported(tmp_path):
    _plain_store(tmp_path / "w")
    (tmp_path / "w" / "store_identity.json").unlink()
    with pytest.raises(ValueError, match=r"store_identity\.json"):
        EpistemicNearestNeighbors.open(tmp_path / "w")


def test_open_restores_y_bounds_without_any_sync(tmp_path):
    bounds = np.array([[0.0, 1.0], [0.0, np.inf]])
    x = np.array([[0.0], [1.0]])
    y = np.array([[0.2, 1.5], [0.8, 3.0]])
    model = EpistemicNearestNeighbors(x, y, y_bounds=bounds, work_dir=tmp_path / "w")
    del model
    reopened = EpistemicNearestNeighbors.open(tmp_path / "w")
    np.testing.assert_array_equal(reopened.rust_backend.y_bounds, bounds)


def test_open_restores_tied_dims(tmp_path):
    x = np.random.default_rng(0).random((30, 3))
    model = EpistemicNearestNeighbors(
        x,
        x.sum(axis=1, keepdims=True),
        tied_dims=[[0, 1]],
        weighting=True,
        work_dir=tmp_path / "w",
    )
    del model
    assert EpistemicNearestNeighbors.open(tmp_path / "w").tied_dims == ((0, 1),)


def test_second_open_model_is_rejected_until_the_first_is_released(tmp_path):
    _plain_store(tmp_path / "w")
    first = EpistemicNearestNeighbors.open(tmp_path / "w")
    with pytest.raises(ValueError, match="already open"):
        EpistemicNearestNeighbors.open(tmp_path / "w")
    del first
    assert len(EpistemicNearestNeighbors.open(tmp_path / "w")) == 30


def test_store_open_in_another_process_is_rejected(tmp_path):
    _plain_store(tmp_path / "w")
    first = EpistemicNearestNeighbors.open(tmp_path / "w")
    script = textwrap.dedent(
        """
        import sys
        import numpy as np
        from enn import EpistemicNearestNeighbors
        EpistemicNearestNeighbors.open(sys.argv[1])
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "w")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    assert "already open" in result.stderr
    del first


def test_reopened_metric_sample_uses_original_outputs_with_y_bounds(tmp_path):
    x, _ = _sparse_rows(1000)
    y = 0.5 + 0.4 * np.sin(6 * x[:, :1])
    bounds = np.array([[0.0, 1.0]])
    closed = EpistemicNearestNeighbors(
        x[:600], y[:600], work_dir=tmp_path / "c", weighting=True, y_bounds=bounds
    )
    never_closed = EpistemicNearestNeighbors(
        x[:600], y[:600], work_dir=tmp_path / "n", weighting=True, y_bounds=bounds
    )
    del closed
    closed = EpistemicNearestNeighbors.open(tmp_path / "c")
    closed.add(x[600:], y[600:])
    never_closed.add(x[600:], y[600:])
    np.testing.assert_allclose(closed._x_scale, never_closed._x_scale, rtol=1e-12)


def test_killed_owner_releases_the_store(tmp_path):
    script = textwrap.dedent(
        """
        import os
        import signal
        import sys
        import numpy as np
        from enn import EpistemicNearestNeighbors
        model = EpistemicNearestNeighbors(
            np.zeros((0, 3)), np.zeros((0, 1)), work_dir=sys.argv[1]
        )
        os.kill(os.getpid(), signal.SIGKILL)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "w")], check=False
    )
    assert result.returncode == -9
    assert len(EpistemicNearestNeighbors.open(tmp_path / "w")) == 0


def test_ask_reads_everything_it_needs_from_the_model(tmp_path):
    x = np.random.default_rng(0).random((300, 3))
    y = x.sum(axis=1, keepdims=True)
    model = EpistemicNearestNeighbors(x, y, work_dir=tmp_path / "w")
    first = ENNStatefulFitter(5, seed=1).ask(model, num_fit_candidates=30)
    second = ENNStatefulFitter(5, seed=1).ask(model, num_fit_candidates=30)
    assert first == second
