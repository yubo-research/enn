from __future__ import annotations
import pytest
from enn.turbo.config.candidate_gen_config import (
    default_num_candidates,
    const_num_candidates,
    CandidateGenConfig,
)
from enn.turbo.config.candidate_rv import CandidateRV


def test_default_num_candidates():
    assert default_num_candidates(num_dim=10, num_arms=4) == 1000
    assert default_num_candidates(num_dim=100, num_arms=4) == 5000


def test_const_num_candidates():
    fn = const_num_candidates(123)
    assert fn(num_dim=10, num_arms=4) == 123
    with pytest.raises(ValueError, match="must be > 0"):
        const_num_candidates(0)


def test_candidate_gen_config_validation():
    config = CandidateGenConfig()
    assert config.candidate_rv == CandidateRV.SOBOL
    with pytest.raises(ValueError, match="CandidateRV enum"):
        CandidateGenConfig(candidate_rv="SOBOL")
    with pytest.raises(ValueError, match="must be callable"):
        CandidateGenConfig(num_candidates=100)
