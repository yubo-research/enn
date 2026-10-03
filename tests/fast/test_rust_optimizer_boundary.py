from __future__ import annotations

import numpy as np
import pytest

from enn import _rust
from enn.turbo.config import ENNSurrogateConfig, turbo_enn_config
from enn.turbo.config.acquisition import UCBAcquisitionConfig
from enn.turbo.config.optimizer_config import OptimizerConfig
from enn.turbo.rust_optimizer import _config_to_rust_overrides

_BOUNDS = np.array([[0.0, 1.0], [0.0, 1.0]])


def test_unknown_override_key_is_rejected():
    with pytest.raises(ValueError, match="lenght_init"):
        _rust.create_optimizer_enn(
            _BOUNDS, None, 4, 0, config_overrides={"lenght_init": 0.01}
        )
    opt = _rust.create_optimizer_enn(
        _BOUNDS, None, 4, 0, config_overrides={"length_init": 0.01}
    )
    assert opt.tr_length() == pytest.approx(0.01)


def test_every_python_override_key_is_accepted_by_rust():
    config = turbo_enn_config(enn=ENNSurrogateConfig(k=4), num_init=4)
    overrides = _config_to_rust_overrides(config)
    assert overrides is not None
    _rust.create_optimizer_enn(_BOUNDS, 4, 4, 0, config_overrides=overrides)


def test_has_surrogate_reports_the_optimizer_kind():
    assert _rust.create_optimizer_enn(_BOUNDS, None, 4, 0).has_surrogate()
    assert not _rust.create_optimizer_zero(_BOUNDS, 4, 0).has_surrogate()
    assert not _rust.create_optimizer_lhd(_BOUNDS, 4, 0).has_surrogate()


def test_ucb_override_leaves_beta_to_rust():
    config = OptimizerConfig(
        surrogate=ENNSurrogateConfig(k=4), acquisition=UCBAcquisitionConfig()
    )
    overrides = _config_to_rust_overrides(config)
    assert overrides is not None
    assert overrides["acquisition"] == "ucb"
    assert "acquisition_beta" not in overrides
