from __future__ import annotations

import pytest

from .optimizer_fixture_catalog import EXPECTED_OPTIMIZER_FIXTURE_NAMES

try:
    from enn._rust import Optimizer  # noqa: F401

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")


@pytest.mark.parametrize("name", EXPECTED_OPTIMIZER_FIXTURE_NAMES)
def test_rust_optimizer_replays_fixture_contracts(name: str):
    from .fixture_replay import (
        _config_for_fixture,
        assert_fixture_contracts,
        load_fixture,
    )

    data = load_fixture(name)
    config = _config_for_fixture(name)
    assert_fixture_contracts(data, config)
