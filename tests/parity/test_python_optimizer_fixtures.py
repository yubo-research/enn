from __future__ import annotations

import pytest

from .fixture_replay import assert_fixture_json_invariants, load_fixture
from .optimizer_fixture_catalog import EXPECTED_OPTIMIZER_FIXTURE_NAMES


@pytest.mark.parametrize("name", EXPECTED_OPTIMIZER_FIXTURE_NAMES)
def test_optimizer_fixture_invariants(name: str):
    data = load_fixture(name)
    assert_fixture_json_invariants(data)
