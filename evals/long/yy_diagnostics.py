"""Independent numerical checks of the output-sharing experiment."""

import json
from pathlib import Path

import numpy as np

from enn import SemanticMultiOutputENN, SemanticOutputProjection
from enn.enn.enn_params import ENNParams
from evals.long.yy_problem import problem


def check_projection(directory):
    projection = SemanticOutputProjection.load(directory / "paraphrase_projection.npz")
    x, y, _ = problem(0, 100, "aligned_noisy")
    params = ENNParams(8, 1, 0)
    independent = SemanticMultiOutputENN(projection, alpha=0, params=params, folds=3)
    shared = SemanticMultiOutputENN(projection, alpha=1, params=params, folds=3)
    baseline = independent.fit(x[:100], y[:100]).posterior(x[100:]).mu
    actual = shared.fit(x[:100], y[:100]).posterior(x[100:]).mu
    mean, scale = y[:100].mean(axis=0), y[:100].std(axis=0)
    p = projection.loadings @ projection.loadings.T
    expected = (((baseline - mean) / scale) @ p) * scale + mean
    error = float(np.max(np.abs(actual - expected)))
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)
    identity_error = float(np.max(np.abs(p @ p - p)))
    assert identity_error < 1e-12
    record = {
        "projection_mean_max_abs_error": error,
        "projector_idempotence_max_abs_error": identity_error,
        "seed": 0,
        "training_rows": 100,
        "query_rows": len(x) - 100,
    }
    (directory / "numerical_checks.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(json.dumps(record))


if __name__ == "__main__":
    check_projection(Path("_reports/yy"))
