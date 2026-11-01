import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from enn import SemanticOutputProjection
from evals.long.semantic_output_scoring import PinnedAlphaENN
from evals.long.yy_problem import problem


def check_projection(directory):
    projection = SemanticOutputProjection.load(directory / "paraphrase_projection.npz")
    x, y, _ = problem(0, 100, "aligned_noisy")
    shared = PinnedAlphaENN(projection, 1).fit(x[:100], y[:100])
    actual = shared.posterior(x[100:]).mu
    blend = shared.params.output_blend
    unblended = replace(shared.params, output_blend=None)
    baseline = shared.model.posterior(x[100:], params=unblended).mu
    p = projection.loadings @ projection.loadings.T
    expected = (((baseline - blend.center) / blend.scale) @ p) * blend.scale
    expected = expected + blend.center
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
