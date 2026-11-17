import os
import runpy
import sys
from pathlib import Path


def main():
    coef = float(os.environ.get("LEAF_COEF", "0") or 0)
    if coef > 0:
        from enn._rust import set_leaf_coefficient

        set_leaf_coefficient(coef)
    here = Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    runpy.run_path(str(here / "capq_0250.py"), run_name="capq_0250")


if __name__ == "__main__":
    main()
