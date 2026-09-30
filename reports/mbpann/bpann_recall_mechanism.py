"""Test two explanations of BPANN_DISK recall figures through tuning alone.

1. Background flush (flush_recall_repro.py, recall 0.10): flushes of ~1,500 rows build
   k-means fragments, whose root is a leaf. Raising structured_build_row_limit to 4096
   makes those flushes build single row-id leaves instead; recall should jump.
2. No sync after construction (bpann_recall_repro.py, recall 0.307): a flat-forest
   fragment forces all fragments to be searched, and the 2,000-row fragments are then
   scanned exhaustively. Setting exhaustive_search_row_limit = 1 disables that scan;
   recall should fall.

Usage: PYTHONPATH=src python reports/mbpann/bpann_recall_mechanism.py
"""

import tempfile
from pathlib import Path

from bpann_recall_repro import recall as repro_recall
from flush_recall_repro import recall as flush_recall

from enn._rust import set_config_path
from enn.turbo.config.enn_index_driver import ENNIndexDriver


def with_config(text: str) -> None:
    cfg = Path(tempfile.mkdtemp()) / "config.toml"
    cfg.write_text("[bpann]\n" + text)
    set_config_path(str(cfg))


CASES = [
    ("flush, default", "", lambda: flush_recall(ENNIndexDriver.BPANN_DISK)),
    ("flush, structured_build_row_limit=4096", "structured_build_row_limit = 4096\n",
     lambda: flush_recall(ENNIndexDriver.BPANN_DISK)),
    ("no first sync, default", "", lambda: repro_recall(False)),
    ("no first sync, exhaustive_search_row_limit=1",
     "exhaustive_search_row_limit = 1\nskip_refinement_row_limit = 150000\n", lambda: repro_recall(False)),
    ("first sync, default", "", lambda: repro_recall(True)),
]

if __name__ == "__main__":
    for name, cfg, fn in CASES:
        with_config(cfg)
        print(f"{name}: recall={fn():.3f}", flush=True)
    set_config_path(None)
