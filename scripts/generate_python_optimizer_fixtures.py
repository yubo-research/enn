from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from tests.parity.optimizer_fixture_catalog import FIXTURE_GENERATOR_ENTRIES  # noqa: E402
from tests.parity.optimizer_fixture_capture import (  # noqa: E402
    build_fixture,
    fixture_output_path,
)


def main() -> None:
    for entry in FIXTURE_GENERATOR_ENTRIES:
        for seed in (0, 1, 2):
            payload = build_fixture(entry, seed)
            path = fixture_output_path(entry, seed, ROOT)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
            print("wrote", path)


if __name__ == "__main__":
    main()
