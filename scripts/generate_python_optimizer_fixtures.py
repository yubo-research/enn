from __future__ import annotations

import json
from pathlib import Path

from enn.turbo.optimizer_fixtures import (
    FIXTURE_GENERATOR_ENTRIES,
    build_fixture,
    fixture_output_path,
)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for entry in FIXTURE_GENERATOR_ENTRIES:
        for seed in (0, 1, 2):
            payload = build_fixture(entry, seed)
            path = fixture_output_path(entry, seed, root)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
            print("wrote", path)


if __name__ == "__main__":
    main()
