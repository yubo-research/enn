"""Synthetic experiment and validation/test separation contracts."""

import inspect

import numpy as np
import pytest

from enn import SemanticOutputProjection
from enn.semantic_multi_output import OutputPosterior
from evals.long.eval_semantic_outputs import evaluate, summarize
from evals.long.semantic_output_problem import method_embeddings, synthetic_problem
from evals.long.semantic_output_scoring import benchmark_method, scores


def definition():
    return SemanticOutputProjection.from_embeddings(
        ["a", "b", "c"], [[1, 0], [1, 0], [0, 1]]
    )


@pytest.mark.parametrize(
    "scenario", ["A_aligned", "B_independent", "C_opposing", "D_misleading"]
)
def test_synthetic_reproducibility(scenario):
    x, y = synthetic_problem(3, 20, definition(), scenario, 12)
    repeated = synthetic_problem(3, 20, definition(), scenario, 12)
    np.testing.assert_array_equal(x, repeated[0])
    np.testing.assert_array_equal(y, repeated[1])
    assert x.shape == (12, 20) and y.shape == (12, 3)
    assert set(method_embeddings(definition(), 3)) == {
        "independent",
        "semantic",
        "shuffled",
        "random",
    }


def test_selection_ignores_test_labels(monkeypatch):
    monkeypatch.setattr(
        "evals.long.semantic_output_scoring._memory_peak", lambda *args: 0
    )
    p = definition()
    x, y = synthetic_problem(3, 2, p, "A_aligned", 16)
    arguments = ((8, 12, 16), (p.names, p.embeddings), ([0.3, 0.9, 1], [0, 0.3, 1]))
    original, _ = benchmark_method(x, y, *arguments)
    y[12:] += 100
    changed, _ = benchmark_method(x, y, *arguments)
    assert original["alpha"] == changed["alpha"]
    assert original["rank"] == changed["rank"]
    assert original["validation_nrmse"] == changed["validation_nrmse"]


def test_scores_and_summary_uncertainty():
    actual = scores(
        np.zeros((2, 3)), OutputPosterior(np.ones((2, 3)), np.ones((2, 3))), np.ones(3)
    )
    np.testing.assert_array_equal(actual["nrmse"], 1)
    np.testing.assert_allclose(actual["loglik"], -0.5 * (np.log(2 * np.pi) + 1))
    row = {
        "dimensions": 20,
        "outputs": 3,
        "n": 50,
        "scenario": "A",
        "method": "semantic",
    }
    row.update(
        {
            metric: 1
            for metric in (
                "nrmse",
                "loglik",
                "coverage95",
                "selection_seconds",
                "fit_seconds",
                "prediction_seconds",
                "python_peak_bytes",
            )
        }
    )
    result = summarize([row, row])[0]
    assert result["repetitions"] == 2 and result["nrmse_se"] == 0
    inspect.signature(evaluate).bind()
    from ops.evaluate import load_evaluate

    inspect.signature(load_evaluate("long/semantic_outputs")).bind()


def test_opposing_effects_within_identical_descriptions():
    p = SemanticOutputProjection.from_embeddings(
        ["a", "b", "c", "d", "e"],
        np.eye(4)[np.arange(5) % 4],
        variance_cutoff=0.9,
    )
    _, aligned = synthetic_problem(2, 20, p, "A_aligned", 100)
    _, opposing = synthetic_problem(2, 20, p, "C_opposing", 100)
    assert np.corrcoef(aligned.T)[0, 4] > 0.8
    assert np.corrcoef(opposing.T)[0, 4] < -0.8


def test_yy_controls_use_cutoff_including_training_pca():
    from evals.long.yy_problem import NAMES, projections

    projection = SemanticOutputProjection.from_embeddings(
        NAMES, np.eye(4)[np.arange(16) % 4]
    )
    y = np.random.default_rng(3).normal(size=(30, 16))
    widths = []
    for cutoff in (0.3, 0.9, 1):
        variants = projections(projection, projection, y, 0, "aligned_noisy", cutoff)
        assert all(p.loadings.shape[1] < 16 for p in variants.values())
        widths.append(variants["pca"].loadings.shape[1])
    assert widths[0] < widths[1] <= widths[2]
