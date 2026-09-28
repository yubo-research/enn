from __future__ import annotations

import tempfile

import numpy as np
import pytest

from evals import metric_12d as mod
from evals import metric_stream_models as sm

TINY = mod.Metric12dConfig(
    n_grid=(10, 30, 60),
    num_test=20,
    batch=8,
    k=4,
    num_fit_candidates=6,
    num_fit_samples=4,
    metric_fit_subsample=25,
    num_seeds=1,
)


@pytest.mark.parametrize("name", sm.STREAM_METRIC_MODELS)
def test_make_source_observes_and_returns_positive_weights(name: str) -> None:
    x, y = mod.make_data(60, np.random.default_rng(0))
    source = sm.make_source(
        name, mod.load_iaml_core(), mod.NUM_DIM, 25, 4, np.random.default_rng(1)
    )
    source.observe(x[:30], y[:30, 0])
    first = source.weights()
    source.observe(x[30:], y[30:, 0])
    second = source.weights()
    assert first.shape == second.shape == (mod.NUM_DIM,)
    assert np.all(np.isfinite(second) & (second > 0))


def test_make_source_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="unknown"):
        sm.make_source("nope", mod.load_iaml_core(), 3, 5, 2, np.random.default_rng(0))


def test_reservoir_lbfgs_warm_starts_after_full(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core = mod.load_iaml_core()
    calls: list[dict] = []
    monkeypatch.setattr(
        core, "fit_exact", lambda metric, x, y, k, **kw: calls.append(kw)
    )
    source = sm.ReservoirLbfgs(core, 3, 10, 2, np.random.default_rng(0))
    rng = np.random.default_rng(1)
    for n in (6, 6, 6):
        source.observe(rng.random((n, 3)), rng.random(n))
        source.weights()
    assert calls == [{}, {}, {"outer": 1, "restart": False}]


@pytest.mark.parametrize("name", ["bpann_disk_sobol", "bpann_disk_spsa"])
def test_streamed_model_applies_source_weights(name: str) -> None:
    x, y = mod.make_data(60, np.random.default_rng(2))
    with tempfile.TemporaryDirectory() as work_dir:
        results = mod.run_model(name, work_dir, TINY)
    assert [r.num_obs for r in results] == [10, 30, 60]
    assert all(np.isfinite(r.loglik) for r in results)


def test_spsa_row_steps_once_per_row() -> None:
    x, y = mod.make_data(40, np.random.default_rng(3))
    core = mod.load_iaml_core()
    batched = sm.make_source(
        "bpann_disk_spsa", core, mod.NUM_DIM, 25, 4, np.random.default_rng(4)
    )
    per_row = sm.make_source(
        "bpann_disk_spsa_row", core, mod.NUM_DIM, 25, 4, np.random.default_rng(4)
    )
    for lo in (0, 20):
        batched.observe(x[lo : lo + 20], y[lo : lo + 20, 0])
        per_row.observe(x[lo : lo + 20], y[lo : lo + 20, 0])
    assert (batched.learner.num_updates, per_row.learner.num_updates) == (20, 38)
    assert not np.allclose(batched.weights(), per_row.weights())


def test_default_models_exclude_slow_per_row_spsa() -> None:
    assert set(sm.PER_ROW_SPSA_MODELS).isdisjoint(mod.MODELS)
    assert set(sm.FAST_STREAM_METRIC_MODELS) <= set(mod.MODELS)
    assert set(sm.STREAM_METRIC_MODELS) == set(sm.FAST_STREAM_METRIC_MODELS) | set(sm.PER_ROW_SPSA_MODELS)
