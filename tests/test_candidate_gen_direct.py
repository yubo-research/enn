from __future__ import annotations

import pytest

from enn.turbo.config import CandidateGenConfig
from enn.turbo.config.num_candidates_fn import (
    const_num_candidates,
    default_num_candidates,
)


def test_default_num_candidates():
    cfg = CandidateGenConfig()
    assert cfg.resolve_num_candidates(num_dim=1, num_arms=1) == 100
    assert cfg.resolve_num_candidates(num_dim=100, num_arms=1) == 5000


def test_default_num_candidates_fn():
    assert default_num_candidates(num_dim=3, num_arms=2) == 300
    assert default_num_candidates(num_dim=100, num_arms=1) == 5000


def test_const_num_candidates_fn():
    assert const_num_candidates(7)(num_dim=1, num_arms=1) == 7


def test_const_num_candidates():
    cfg = CandidateGenConfig(num_candidates=123)
    assert cfg.resolve_num_candidates(num_dim=3, num_arms=7) == 123
    with pytest.raises(ValueError, match="num_candidates must be > 0"):
        CandidateGenConfig(num_candidates=0)
