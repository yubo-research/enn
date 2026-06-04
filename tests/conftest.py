from __future__ import annotations

import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def enn_all_train_rows(model):
    """Return (x, y, yvar?) for all rows via index-based gather."""
    n = len(model)
    return model.train_rows_at(list(range(n)))


_ROOT = Path(__file__).parent.parent
_NATIVE_FP_CACHE = _ROOT / ".pytest_cache" / "enn_native_extension_fingerprint"
_TESTMON_DB = (
    _ROOT / ".testmondata",
    _ROOT / ".testmondata-shm",
    _ROOT / ".testmondata-wal",
)


def _testmon_invalidation_key() -> str:
    import enn.enn_rust as native

    native_path = Path(native.__file__)
    native_stat = native_path.stat()
    return f"{native_path}:{native_stat.st_mtime_ns}:{native_stat.st_size}"


def _wipe_testmon_data() -> None:
    for db_path in _TESTMON_DB:
        if db_path.exists():
            db_path.unlink()


def pytest_configure(config) -> None:
    if not config.pluginmanager.hasplugin("testmon") or config.getoption("no-testmon"):
        return
    fingerprint = _testmon_invalidation_key()
    previous = _NATIVE_FP_CACHE.read_text() if _NATIVE_FP_CACHE.exists() else None
    if previous == fingerprint:
        return
    _wipe_testmon_data()
    _NATIVE_FP_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _NATIVE_FP_CACHE.write_text(fingerprint)


def sphere_objective(x):
    import numpy as np

    return -np.sum(x**2, axis=1)


def make_from_unit_fn(bounds):
    from enn.turbo.python_fallback.turbo_utils import from_unit

    def from_unit_fn(x):
        return from_unit(x, bounds)

    return from_unit_fn


def make_select_sobol_fn(bounds, rng):
    from enn.turbo.python_fallback.turbo_utils import from_unit

    def select_sobol_fn(x, n):
        idx = rng.choice(x.shape[0], size=n, replace=False)
        return from_unit(x[idx], bounds)

    return select_sobol_fn


def make_enn_model(n=20, d=3, seed=0, yvar_scale=0.1):
    import numpy as np

    from enn.enn.enn_class import EpistemicNearestNeighbors

    rng = np.random.default_rng(seed)
    train_x = rng.standard_normal((n, d))
    train_y = (train_x.sum(axis=1, keepdims=True)).astype(float)
    train_yvar = yvar_scale * np.ones_like(train_y)
    model = EpistemicNearestNeighbors(train_x, train_y, train_yvar)
    return model, train_x, train_y, train_yvar, rng
