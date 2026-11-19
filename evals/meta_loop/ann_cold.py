import os
import runpy
import sys
from pathlib import Path


def main():
    import enn._rust as R

    setters = {
        "leaf_coef": R.set_leaf_coefficient,
        "code_extra": lambda v: R.set_code_extra(int(v)),
        "code_prefetch": lambda v: R.set_code_prefetch(int(v)),
        "meta_prefetch": lambda v: R.set_meta_prefetch(int(v)),
        "icache": lambda v: R.set_internal_cache_bytes(int(v)),
        "leaf_codes": lambda v: R.set_leaf_codes(bool(v)),
    }
    for kv in os.environ.get("ABL", "").split(","):
        if "=" in kv:
            name, v = kv.split("=")
            setters[name](float(v))
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    runpy.run_path(str(here / "cold_cost.py"), run_name="cold_cost")


if __name__ == "__main__":
    main()
