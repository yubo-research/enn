from __future__ import annotations

import sys
from pathlib import Path


def test_optimizer_baseline_generator_symbols():
    scripts = Path(__file__).resolve().parent / "scripts"
    sys.path.insert(0, str(scripts))
    from generate_optimizer_quality_baseline import main as quality_main
    from generate_python_optimizer_fixtures import main as fixtures_main

    assert callable(quality_main)
    assert callable(fixtures_main)
