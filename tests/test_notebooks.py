"""Tests that notebooks run without errors using nbmake."""

import os
import subprocess
import sys

import pytest


@pytest.fixture(autouse=True)
def set_fast_test():
    os.environ["FAST_TEST"] = "1"
    yield
    os.environ.pop("FAST_TEST", None)


def run_nbmake(notebook_path: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--nbmake", notebook_path, "-v"],
        capture_output=True,
        text=True,
        env={**os.environ, "FAST_TEST": "1"},
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
