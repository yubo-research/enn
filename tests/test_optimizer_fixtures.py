from __future__ import annotations

from enn.turbo.optimizer_fixtures import (
    assert_fixture_contracts,
    assert_fixture_json_invariants,
    build_fixture,
    capture_optimizer_fixture,
    catalog_fixture_names,
    entry_for_fixture_name,
    fixture_name_prefix,
    fixture_output_path,
    fixture_subdir_for_entry,
    load_fixture,
    separable_unimodal_objective,
    sphere_centered_objective,
)
from enn.turbo.optimizer_fixtures.catalog import FixtureGeneratorEntry, FixtureRunSpec


def test_optimizer_fixtures_kiss_symbols_referenced():
    assert callable(capture_optimizer_fixture)
    assert callable(build_fixture)
    assert callable(fixture_output_path)
    assert callable(sphere_centered_objective)
    assert callable(separable_unimodal_objective)
    assert FixtureRunSpec is not None
    assert FixtureGeneratorEntry is not None
    assert callable(fixture_name_prefix)
    assert callable(entry_for_fixture_name)
    assert callable(fixture_subdir_for_entry)
    assert callable(catalog_fixture_names)
    assert callable(load_fixture)
    assert callable(assert_fixture_json_invariants)
    assert callable(assert_fixture_contracts)
