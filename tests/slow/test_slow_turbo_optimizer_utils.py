from __future__ import annotations

import pytest

from enn.turbo.python_fallback.turbo_utils import get_gp_posterior_suppress_warning


@pytest.mark.slow
def test_get_gp_posterior_suppress_warning_basic():
    import torch

    from enn.turbo.python_fallback.turbo_gp_fit import fit_gp

    x = [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]]
    y = [1.0, 2.0, 3.0, 4.0]
    gp_result = fit_gp(x, y, num_dim=2, num_steps=2)
    if gp_result.model is not None:
        x_torch = torch.tensor([[0.2, 0.3]], dtype=torch.float64)
        result = get_gp_posterior_suppress_warning(gp_result.model, x_torch)
        assert result is not None
