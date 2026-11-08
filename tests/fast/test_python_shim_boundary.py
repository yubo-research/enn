"""The Python package marshals arguments; every calculation belongs to Rust."""

import ast
from pathlib import Path

import pytest

import enn

PACKAGE = Path(enn.__file__).parent
MODULES = sorted(PACKAGE.rglob("*.py"))
NUMPY_MARSHALLING = {
    "array",
    "array_equal",
    "asarray",
    "atleast_2d",
    "empty",
    "float64",
    "floating",
    "int64",
    "integer",
    "issubdtype",
    "load",
    "ndarray",
    "newaxis",
    "ones",
    "pi",
    "random",
    "savez",
    "zeros",
}
FORBIDDEN_IMPORTS = {"math", "ops", "scipy", "statistics"}
ARITHMETIC = (
    ast.Add,
    ast.Div,
    ast.FloorDiv,
    ast.MatMult,
    ast.Mod,
    ast.Mult,
    ast.Pow,
    ast.Sub,
)
NUMERIC_BUILTINS = {"abs", "divmod", "max", "min", "pow", "round", "sorted", "sum"}
ARRAY_COMPUTATION_METHODS = {
    "argmax",
    "argmin",
    "argsort",
    "clip",
    "cumsum",
    "dot",
    "max",
    "mean",
    "min",
    "nonzero",
    "prod",
    "std",
    "sum",
    "var",
}


def module_id(path):
    return str(path.relative_to(PACKAGE))


def numpy_attributes(tree):
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "np"
        ):
            yield node.attr


def imported_roots(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            yield node.module.split(".")[0]


def is_literal(node):
    if isinstance(node, ast.BinOp):
        return is_literal(node.left) and is_literal(node.right)
    named_constant = isinstance(node, ast.Attribute) and node.attr == "pi"
    return named_constant or isinstance(node, ast.Constant)


def computed_arithmetic(tree):
    for node in ast.walk(tree):
        if (
            (
                isinstance(node, ast.BinOp)
                and isinstance(node.op, ARITHMETIC)
                and not is_literal(node)
            )
            or isinstance(node, ast.AugAssign)
            and isinstance(node.op, ARITHMETIC)
        ):
            yield node.lineno


def computation_calls(tree):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Name)
            and func.id in NUMERIC_BUILTINS
            or isinstance(func, ast.Attribute)
            and func.attr in ARRAY_COMPUTATION_METHODS
        ):
            yield node.lineno


@pytest.mark.parametrize("path", MODULES, ids=module_id)
def test_module_only_marshals_for_rust(path):
    tree = ast.parse(path.read_text())
    assert set(numpy_attributes(tree)) <= NUMPY_MARSHALLING
    assert not set(imported_roots(tree)) & FORBIDDEN_IMPORTS
    assert list(computed_arithmetic(tree)) == []
    assert list(computation_calls(tree)) == []


def test_arithmetic_detector_flags_every_operator():
    tree = ast.parse("a = b - c\nd = e + 1\nf /= g\nh = 2**63 - 1\ni = 'j'")
    assert sorted(computed_arithmetic(tree)) == [1, 2, 3]


def test_computation_detector_flags_builtins_and_array_methods():
    tree = ast.parse("a = max(b, 0)\nc = d.sum()\ne = f.reshape(-1)")
    assert list(computation_calls(tree)) == [1, 2]


def test_arithmetic_detector_flags_array_algebra():
    tree = ast.parse(
        "y = a @ b\nz = x**2\nlimit = 2**63 - 1\nw = np.linalg.norm(2 * np.pi * y)"
    )
    assert list(computed_arithmetic(tree)) == [1, 2, 4]
    assert set(numpy_attributes(tree)) == {"linalg", "pi"}


def test_package_ships_no_evaluation_harness():
    assert not (PACKAGE / "evals").exists()
