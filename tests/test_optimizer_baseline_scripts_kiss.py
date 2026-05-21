from __future__ import annotations


def test_optimizer_baseline_generator_symbols():
    from scripts.generate_optimizer_quality_baseline import main as quality_main
    from scripts.generate_python_optimizer_fixtures import main as fixtures_main

    assert callable(quality_main)
    assert callable(fixtures_main)
