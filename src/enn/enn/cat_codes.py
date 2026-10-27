"""Optional numeric codes for categorical columns.

A category id is a label, not a magnitude. Leave every column continuous and
ENN behaves as before. To encode, mark each column with ``DimKind`` and choose
``CatEncoding``:

- ``ONE_HOT``: one column per category, ids ``0 .. C-1``.
- ``RADEMACHER``: a fixed-width sign vector. Same category, same vector.
  Different categories land nearly the same distance apart.
- ``AUTO``: one-hot when that block is no longer than the Rademacher block,
  and Rademacher otherwise. With the default cap that change happens at 16
  categories.

``DimKind`` is the place an ordinal kind would be added later. It has no
ordinal member until there is an encoding for ordered levels.

Per-variable blocks are multiplied by ``separation`` (default 10) so a
mismatch is long compared with ENN's aleatoric floor. ``encode`` records
the columns of each categorical variable. Metric learning gives those
columns one shared weight. An explicit ``tied_dims`` must list that same
grouping.

``code_dim`` forces every categorical column to one Rademacher width.
``hash_dim`` instead folds every categorical column into one shared vector.
Those two options are alternatives to a per-column ``cards`` plan.
"""

from __future__ import annotations

from enum import Enum, auto

import numpy as np

from enn._rust import CatEncoder as _RustCatEncoder
from enn._rust import cat_column_plan
from enn.enn.cat_block import log_code_width
from enn.enn.cat_ties import attach_tied_dims

__all__ = ["CatEncoder", "CatEncoding", "DimKind", "column_plan", "log_code_width"]


class DimKind(Enum):
    CONTINUOUS = auto()
    CATEGORICAL = auto()


class CatEncoding(Enum):
    ONE_HOT = auto()
    RADEMACHER = auto()
    AUTO = auto()


def _require(value, kind, label):
    if not isinstance(value, kind):
        raise TypeError(f"{label} must be a {kind.__name__}, got {value!r}")
    return value


def column_plan(card, encoding=CatEncoding.AUTO, dim_cap=48):
    return cat_column_plan(
        int(card), _require(encoding, CatEncoding, "encoding").name, dim_cap
    )


class CatEncoder:
    """Mixed-column categorical representation, implemented entirely in Rust.

    Continuous columns pass through. Categorical columns become one-hot or
    Rademacher blocks, or a shared hash. Encoding attaches each block's tie group.
    """

    def __init__(
        self,
        kinds,
        cards=None,
        *,
        encoding=CatEncoding.AUTO,
        code_dim=None,
        hash_dim=None,
        seed=0,
        dim_cap=48,
        separation=10.0,
    ):
        self.kinds = tuple(kinds)
        for kind in self.kinds:
            _require(kind, DimKind, "kind")
        self.encoding = _require(encoding, CatEncoding, "encoding")
        self.seed = int(seed)
        self.separation = float(separation)
        self.hash_dim = 0 if hash_dim is None else int(hash_dim)
        self._rust = _RustCatEncoder(
            [k is DimKind.CATEGORICAL for k in self.kinds],
            None if cards is None else list(cards),
            self.encoding.name,
            code_dim,
            hash_dim,
            self.seed,
            dim_cap,
            self.separation,
        )

    @property
    def num_dim(self):
        return self._rust.num_dim

    @property
    def styles(self):
        return tuple(self._rust.styles)

    @property
    def code_dims(self):
        return tuple(self._rust.code_dims)

    def tied_dims(self):
        return tuple(tuple(g) for g in self._rust.tied_dims())

    def encode(self, x):
        values = np.asarray(x)
        if values.dtype.kind in "iu":
            values = np.asarray(values, dtype=np.int64)
        elif values.dtype.kind != "O":
            values = np.asarray(values, dtype=np.float64)
        return attach_tied_dims(self._rust.encode(values), self.tied_dims())

    def snap(self, encoded, domains):
        return self._rust.snap(
            np.asarray(encoded, dtype=float),
            [np.asarray(d, dtype=np.int64).ravel().tolist() for d in domains],
        )
