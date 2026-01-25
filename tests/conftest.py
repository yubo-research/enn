from __future__ import annotations
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))


def sphere_objective(x):
    import numpy as np

    return -np.sum(x**2, axis=1)


def make_from_unit_fn(bounds):
    from enn.turbo.turbo_utils import from_unit

    def from_unit_fn(x):
        return from_unit(x, bounds)

    return from_unit_fn


def make_select_sobol_fn(bounds, rng):
    from enn.turbo.turbo_utils import from_unit

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
