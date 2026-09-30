from __future__ import annotations

import numpy as np
import pytest

from evals import metric_categorical as mod
from evals.metric_12d import Metric12dConfig
from evals.short import eval_metric_categorical as entry

TINY = Metric12dConfig(
    n_grid=(20, 120),
    num_test=20,
    batch=60,
    k=4,
    num_fit_candidates=4,
    num_fit_samples=4,
    num_seeds=1,
)


def test_tied_groups() -> None:
    assert mod.tied_groups((1, 3, 2)) == ((0,), (1, 2, 3), (4, 5))
    assert mod.tied_groups((2,), start=4) == ((4, 5),)
    assert mod.PROBLEMS[1].tied_dims == ((4, 5, 6), (7, 8, 9, 10, 11), (12, 13))


@pytest.mark.parametrize("problem", mod.PROBLEMS, ids=lambda p: p.name)
def test_data_one_hot_blocks(problem: mod.Problem) -> None:
    x, y = problem.data(400, np.random.default_rng(0))
    num_dim = 1 + max(j for g in problem.tied_dims for j in g)
    assert x.shape == (400, num_dim) and y.shape == (400, 1)
    for g in problem.tied_dims:
        np.testing.assert_array_equal(x[:, list(g)].sum(axis=1), 1.0)
        assert set(np.unique(x[:, list(g)])) <= {0.0, 1.0}
    assert np.std(y) > 3 * mod.NOISE_STD


def test_data_scales_and_dependence() -> None:
    x, _ = mod.make_mixed(2000, np.random.default_rng(1))
    np.testing.assert_allclose(x[:, :4].max(axis=0), mod.MIXED_RANGES, rtol=0.01)
    for t, s in zip(mod.CAT_ONLY_EFFECTS[1:], mod.CAT_ONLY_STRENGTHS[1:]):
        assert t.std() == pytest.approx(s)
    assert mod.CAT_ONLY_EFFECTS[0].tolist() == [0.0]
    assert mod.CAT_ONLY_PAIR.shape == (3, 4)


def test_build_model_options(tmp_path) -> None:
    x, y = mod.make_mixed(50, np.random.default_rng(0))
    tied = mod.PROBLEMS[1].tied_dims
    tied_model = mod.build_model("bpann_disk_auto_tied", x, y, str(tmp_path), tied)
    assert tied_model.tied_dims == tied and tied_model.metric is not None
    plain = mod.build_model("bpann_disk_scale_x", x, y, str(tmp_path), tied)
    assert plain.tied_dims == () and plain.metric is None
    with pytest.raises(ValueError, match="unknown model"):
        mod.build_model("flat", x, y, str(tmp_path), tied)


def test_run_eval_tiny(capsys: pytest.CaptureFixture[str]) -> None:
    out = mod.run_eval(TINY)
    printed = capsys.readouterr().out
    assert list(out) == ["cat_only", "mixed"]
    for summaries in out.values():
        assert [(s.model, s.num_obs) for s in summaries] == [
            (m, n) for m in mod.MODELS for n in TINY.n_grid
        ]
        assert all(np.isfinite(s.loglik.mean) and np.isfinite(s.nrmse.mean) for s in summaries)
    assert printed.count("EVAL: problem = cat_only model = ") == len(mod.MODELS) * 2
    assert printed.count("EVAL: problem = mixed model = ") == len(mod.MODELS) * 2


def test_entry_calls_run_eval(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(entry, "run_eval", lambda: calls.append(1))
    entry.evaluate()
    assert calls == [1]
