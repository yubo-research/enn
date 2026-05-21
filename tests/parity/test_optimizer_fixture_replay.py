from __future__ import annotations

import pytest

from .fixture_replay import list_fixture_names

try:
    from enn._rust import Optimizer  # noqa: F401

    RUST_AVAILABLE = True
except ImportError:
    RUST_AVAILABLE = False

pytestmark = pytest.mark.skipif(not RUST_AVAILABLE, reason="Rust not available")


@pytest.mark.parametrize("name", list_fixture_names())
def test_rust_optimizer_replays_fixture_contracts(name: str):
    from .fixture_replay import (
        _config_for_fixture,
        assert_fixture_contracts,
        load_fixture,
    )

    data = load_fixture(name)
    config = _config_for_fixture(name)
    assert_fixture_contracts(data, config)
