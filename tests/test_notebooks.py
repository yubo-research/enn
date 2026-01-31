import os
import subprocess
import sys
import importlib.util
from pathlib import Path
import pytest


@pytest.fixture(autouse=True)
def set_fast_test():
    os.environ["FAST_TEST"] = "1"
    yield
    os.environ.pop("FAST_TEST", None)


def run_nbmake(notebook_path: str) -> None:
    if importlib.util.find_spec("nbmake") is None:
        pytest.skip("nbmake is not installed")
    repo_root = Path(__file__).resolve().parent.parent
    shim_dir = Path(__file__).resolve().parent / "_nbmake_sitecustomize"
    src_dir = repo_root / "src"
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    pythonpath_parts = [str(shim_dir), str(src_dir)]
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--nbmake", notebook_path, "-v"],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "FAST_TEST": "1",
            "PYTHONPATH": os.pathsep.join(pythonpath_parts),
        },
    )
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise AssertionError(f"Notebook {notebook_path} failed:\n{result.stderr}")


def test_demo_enn_notebook():
    run_nbmake("examples/demo_enn.ipynb")


def test_demo_turbo_enn_notebook():
    run_nbmake("examples/demo_turbo_enn.ipynb")


def test_demo_morbo_enn_notebook():
    run_nbmake("examples/demo_morbo_enn.ipynb")
