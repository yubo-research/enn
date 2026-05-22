from __future__ import annotations

from pathlib import Path

from enn.turbo.optimizer_fixtures import (
    EXPECTED_OPTIMIZER_FIXTURE_NAMES,
    FIXTURE_GENERATOR_ENTRIES,
    PREFIX_CONFIG,
    catalog_fixture_names,
    entry_for_fixture_name,
    fixture_name_prefix,
    fixture_output_path,
    fixture_subdir_for_entry,
    load_fixture,
)


def test_catalog_fixture_names():
    assert EXPECTED_OPTIMIZER_FIXTURE_NAMES == catalog_fixture_names()
    assert len(EXPECTED_OPTIMIZER_FIXTURE_NAMES) == 21


def test_each_entry_config_paths_and_morbo_routing():
    root = Path(__file__).resolve().parents[2]
    for entry in FIXTURE_GENERATOR_ENTRIES:
        assert entry.config_key in PREFIX_CONFIG
        subdir = fixture_subdir_for_entry(entry)
        assert (subdir == "morbo") == entry.morbo
        for seed in (0, 1, 2):
            name = f"{entry.prefix}{seed}"
            assert entry_for_fixture_name(name) == entry
            assert (fixture_name_prefix(name).startswith("morbo_")) == entry.morbo
            gen_path = fixture_output_path(entry, seed, root)
            assert gen_path == root / "tests" / "fixtures" / subdir / f"{name}.json"
            assert load_fixture(name) is not None
