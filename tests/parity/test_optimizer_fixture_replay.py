from __future__ import annotations

import pytest

from enn.turbo.optimizer_fixtures import (
    EXPECTED_OPTIMIZER_FIXTURE_NAMES,
    assert_fixture_contracts,
    load_fixture,
)
from enn.turbo.optimizer_fixtures.replay import _config_for_fixture

try:
    from enn._rust import Optimizer  # noqa: F401

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")


@pytest.mark.parametrize("name", EXPECTED_OPTIMIZER_FIXTURE_NAMES)
def test_rust_optimizer_replays_fixture_contracts(name: str):
    data = load_fixture(name)
    config = _config_for_fixture(name)
    assert_fixture_contracts(data, config)
