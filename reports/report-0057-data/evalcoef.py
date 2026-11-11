import sys

sys.path.insert(0, "/home/dsweet/projects/enn/src")
sys.path.insert(0, "/home/dsweet/projects/enn")
from enn import _rust

_rust.set_leaf_coefficient(float(sys.argv[1]))
sys.argv = ["evaluate.py", "run", sys.argv[2]]
import runpy

runpy.run_path("/home/dsweet/projects/enn/ops/evaluate.py", run_name="__main__")
