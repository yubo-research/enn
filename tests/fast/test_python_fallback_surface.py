from __future__ import annotations

import pytest

from enn.turbo.python_fallback.components.builder import build_surrogate
from enn.turbo.config import turbo_enn_config, turbo_one_config, turbo_zero_config


def test_build_surrogate_accepts_gp_only():
    assert isinstance(build_surrogate(turbo_one_config()), object)


def test_build_surrogate_rejects_rust_owned_configs():
    for cfg in (turbo_enn_config(), turbo_zero_config()):
        with pytest.raises(ValueError, match="Rust optimizer"):
            build_surrogate(cfg)
