import os
import runpy
import sys
from pathlib import Path


def main():
    import enn._rust as R

    setters = {
        "leaf_coef": R.set_leaf_coefficient,
        "internal_coef": R.set_internal_coefficient,
        "leaf_codes": lambda v: R.set_leaf_codes(bool(v)),
        "code_extra": lambda v: R.set_code_extra(int(v)),
        "icache": lambda v: R.set_internal_cache_bytes(int(v)),
        "code_prefetch": lambda v: R.set_code_prefetch(int(v)),
        "meta_prefetch": lambda v: R.set_meta_prefetch(int(v)),
        "stop_eps": R.set_stop_epsilon,
        "radius_w": R.set_radius_weight,
    }
    for kv in os.environ.get("ABL", "").split(","):
        if "=" in kv:
            name, v = kv.split("=")
            setters[name](float(v))
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    runpy.run_path(str(here / "capq_0250.py"), run_name="capq_0250")


if __name__ == "__main__":
    main()
