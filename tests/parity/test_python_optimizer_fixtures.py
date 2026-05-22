from __future__ import annotations

import pytest

from enn.turbo.optimizer_fixtures import (
    EXPECTED_OPTIMIZER_FIXTURE_NAMES,
    assert_fixture_json_invariants,
    load_fixture,
)

try:
    from enn._rust import Optimizer  # noqa: F401

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    RUST_AVAILABLE,
    reason="JSON invariants are exercised by Rust replay tests",
)


@pytest.mark.parametrize("name", EXPECTED_OPTIMIZER_FIXTURE_NAMES)
def test_optimizer_fixture_invariants(name: str):
    data = load_fixture(name)
    assert_fixture_json_invariants(data)
