from __future__ import annotations

import numpy as np

from enn.enn.enn_conditional_helpers import (
    build_candidates,
    compute_draw_internals,
    compute_mu_se,
    compute_search_k,
    compute_total_n,
    get_train_candidates,
    get_whatif_candidates,
    make_empty_normal,
    merge_candidates,
    pairwise_sq_l2,
    scale_x_if_needed,
    select_effective_neighbors,
    select_sorted_candidates,
    take_along_axis_3d,
    validate_whatif,
    validate_x,
)
from enn.enn.enn_params import ENNParams, PosteriorFlags


def test_conditional_helpers_exports_are_callable():
    a = np.array([[0.0, 0.0], [1.0, 1.0]])
    b = np.array([[0.5, 0.5]])
    d = pairwise_sq_l2(a, b)
    assert d.shape == (2, 1)
    p = ENNParams(
        k_num_neighbors=2, epistemic_variance_scale=1.0, aleatoric_variance_scale=1.0
    )
    f = PosteriorFlags()
    assert compute_search_k(p, f, 5) >= 1
    assert compute_total_n is not None
    assert validate_x is not None
    assert validate_whatif is not None
    assert scale_x_if_needed is not None
    assert get_train_candidates is not None
    assert get_whatif_candidates is not None
    assert merge_candidates is not None
    assert select_sorted_candidates is not None
    assert take_along_axis_3d is not None
    assert make_empty_normal is not None
    assert build_candidates is not None
    assert select_effective_neighbors is not None
    assert compute_mu_se is not None
    assert compute_draw_internals is not None
