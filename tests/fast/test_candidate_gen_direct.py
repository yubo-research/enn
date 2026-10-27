from __future__ import annotations

import pytest

from enn.turbo.config.candidate_gen_config import CandidateGenConfig
from enn.turbo.config.candidate_rv import CandidateRV
from enn.turbo.config.enn_index_driver import ENNIndexDriver
from enn.turbo.config.enn_surrogate_config import ENNSurrogateConfig
from enn.turbo.config.factory import turbo_enn_config, turbo_zero_config
from enn.turbo.config.tr_length_config import TRLengthConfig
from enn.turbo.config.turbo_tr_config import TurboTRConfig
from enn.turbo.rust_optimizer_helpers import _config_to_rust_overrides


def test_turbo_factories_leave_the_sampler_to_rust():
    for config in (turbo_enn_config(), turbo_zero_config()):
        assert config.candidate_rv is None
        overrides = _config_to_rust_overrides(config) or {}
        assert "candidate_rv" not in overrides


def test_candidate_defaults_are_the_four_parameters():
    cfg = CandidateGenConfig()
    assert cfg.min_candidates == 10
    assert cfg.max_candidates == 5000
    assert cfg.num_candidates_per_dim == 100
    assert cfg.num_candidates_per_arm == 0


def test_candidate_params_reject_negatives():
    with pytest.raises(ValueError, match="min_candidates must be >= 0"):
        CandidateGenConfig(min_candidates=-1)
    with pytest.raises(ValueError, match="num_candidates_per_arm must be >= 0"):
        CandidateGenConfig(num_candidates_per_arm=-1)


def test_candidate_params_reach_rust_overrides():
    cfg = turbo_enn_config(
        candidates=CandidateGenConfig(
            candidate_rv=CandidateRV.UNIFORM,
            min_candidates=10,
            max_candidates=40,
            num_candidates_per_dim=0,
            num_candidates_per_arm=3,
        )
    )
    overrides = _config_to_rust_overrides(cfg)
    assert overrides["min_candidates"] == 10
    assert overrides["max_candidates"] == 40
    assert overrides["num_candidates_per_dim"] == 0
    assert overrides["num_candidates_per_arm"] == 3


def test_unset_lengths_are_not_sent():
    overrides = _config_to_rust_overrides(turbo_enn_config())
    assert overrides is None or "length_init" not in overrides
    explicit = turbo_enn_config(
        trust_region=TurboTRConfig(length=TRLengthConfig(length_init=0.8))
    )
    sent = _config_to_rust_overrides(explicit)
    assert sent["length_init"] == 0.8


def test_length_order_is_checked_when_both_ends_are_set():
    with pytest.raises(ValueError, match="length_min must be < length_max"):
        TRLengthConfig(length_min=2.0, length_max=1.0)


def test_backend_selects_storage_and_requires_disk_directory():
    with pytest.raises(ValueError, match="FLAT"):
        ENNSurrogateConfig(index_driver=ENNIndexDriver.FLAT, work_dir="/tmp/enn")
    with pytest.raises(ValueError, match="work_dir"):
        ENNSurrogateConfig(index_driver=ENNIndexDriver.BPANN_DISK)
    ENNSurrogateConfig(
        index_driver=ENNIndexDriver.BPANN_DISK, work_dir="/tmp/enn", weighting=True
    )
    ENNSurrogateConfig(weighting=True, tied_dims=((0, 1),))
