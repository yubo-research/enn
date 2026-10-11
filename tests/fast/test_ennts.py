from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from enn.ennts import ENNTS, ENNTSConfig
from enn.turbo.config import surrogate
from evals.short import eval_ennts_cv, eval_ennts_timing


def test_posterior_uses_only_an_available_target() -> None:
    model = ENNTS(
        ENNTSConfig(
            lookback=5,
            bucket_width=1.0e9,
            k=1,
            num_tables=2,
            num_projections=2,
            probe=0,
            seed=1,
            aleatoric_variance_scale=0.0,
        )
    )
    x = np.zeros((8, 2))
    x[:, 0] = np.arange(8)
    model.add(0, x, np.arange(8, dtype=float))
    mu, se, _, _ = model.posterior(x[:5], cutoff=5, k=1, observation_noise=False)
    assert abs(float(np.asarray(mu).ravel()[0]) - 5.0) < 1e-6
    assert np.isfinite(se).all()
    series, _, _, _ = model.search(x[:5], cutoff=4, k=1, require_target=True)
    assert int(series[0, 0]) == -1


def test_metric_learning_rejects_unknown_names_and_starts_at_identity() -> None:
    with pytest.raises(ValueError, match="metric_learning"):
        ENNTSConfig(lookback=2, metric_learning="full")
    model = ENNTS(
        ENNTSConfig(
            lookback=4,
            metric_learning="auto",
            bucket_width=1.0e9,
            num_tables=2,
            num_projections=2,
            k=2,
        )
    )
    x = np.column_stack([np.linspace(0.0, 1.0, 12), np.zeros(12)])
    model.add(0, x, x[:, 0])
    weights = model.metric_weights
    assert weights is not None
    assert np.allclose(weights, 1.0)
    assert model.heldout_gain is None
    assert model.metric_morph_remaining == 0
    assert model.uses_learned_metric is False
    mu, se, _, _ = model.posterior(x[:4], cutoff=8, observation_noise=False)
    assert np.isfinite(mu).all()
    assert np.isfinite(se).all()


def test_lookback_is_required_and_stays_fixed() -> None:
    with pytest.raises(TypeError):
        ENNTSConfig()
    x = np.zeros((4, 1))
    y = np.zeros(4)
    short = ENNTS(ENNTSConfig(lookback=3, bucket_width=1.0e9, num_tables=1, num_projections=1))
    long = ENNTS(ENNTSConfig(lookback=7, bucket_width=1.0e9, num_tables=1, num_projections=1))
    short.add(0, x, y)
    long.add(1, x, y)
    assert short.lookback == 3
    assert long.lookback == 7
    assert short.num_windows == 2
    assert long.num_windows == 0
    short.set_variance_scales(2.0, 0.2)
    assert short.lookback == 3


def test_optimizer_surrogate_does_not_name_ennts() -> None:
    text = Path(surrogate.__file__).read_text(encoding="utf-8")
    assert "ENNTS" not in text
    assert "EnnTs" not in text


def test_cv_eval_prints_loglik_and_nrmse(capsys) -> None:
    eval_ennts_cv.evaluate()
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("EVAL:")]
    assert len(lines) == 4
    assert "LARGER(loglik)" in lines[0]
    assert "SMALLER(nrmse)" in lines[0]
    assert "nan" not in lines[0]


def test_timing_eval_prints_add_and_query(capsys) -> None:
    eval_ennts_timing.evaluate()
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("EVAL:")]
    assert len(lines) == 8
    assert "n = " in lines[0]
    assert "SMALLER(add_s)" in lines[0]
    assert "SMALLER(query_s)" in lines[0]
