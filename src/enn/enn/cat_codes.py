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
mismatch is long compared with ENN's aleatoric floor. Pass the result of
``encode`` to ``EpistemicNearestNeighbors`` with ``tied_dims``.

``code_dim`` forces every categorical column to one Rademacher width.
``hash_dim`` instead folds every categorical column into one shared vector.
Those two options are alternatives to a per-column ``cards`` plan.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum, auto

import numpy as np

from enn.enn.cat_block import (
    CODE_BIT,
    hash_columns,
    log_code_width,
    rademacher,
    snap_code,
    snap_onehot,
    width_in_range,
    write_onehot,
)

__all__ = ["CatEncoder", "CatEncoding", "DimKind", "column_plan", "log_code_width"]


class DimKind(Enum):
    """Role of one input column."""

    CONTINUOUS = auto()
    CATEGORICAL = auto()


class CatEncoding(Enum):
    """How categorical columns become numeric blocks."""

    ONE_HOT = auto()
    RADEMACHER = auto()
    AUTO = auto()


def _require(value, kind: type[Enum], label: str):
    if not isinstance(value, kind):
        raise TypeError(f"{label} must be a {kind.__name__}, got {value!r}")
    return value


def _onehot_width(card: int) -> int:
    count = int(card)
    if count < 1:
        raise ValueError(f"card must be >= 1, got {count}")
    return count


def column_plan(
    card: int,
    encoding: CatEncoding = CatEncoding.AUTO,
    dim_cap: int = 48,
) -> tuple[int, str]:
    """Width and style for one domain.

    ``AUTO`` uses one-hot when ``card`` is at most the Rademacher width.
    """
    _require(encoding, CatEncoding, "encoding")
    if encoding is CatEncoding.ONE_HOT:
        return _onehot_width(card), "onehot"
    coded = log_code_width(card, dim_cap)
    if encoding is CatEncoding.RADEMACHER or card > coded:
        return coded, "code"
    return int(card), "onehot"


def _parse_kinds(kinds: Sequence[DimKind]) -> tuple[DimKind, ...]:
    parsed = tuple(kinds)
    if not parsed:
        raise ValueError("kinds must list at least one column")
    for kind in parsed:
        _require(kind, DimKind, "kind")
    return parsed


def _mode(encoding: CatEncoding, code_dim: int | None, hash_dim: int | None) -> str:
    if hash_dim is not None and code_dim is not None:
        raise ValueError("pass only one of code_dim, hash_dim")
    if hash_dim is not None:
        if encoding is not CatEncoding.AUTO:
            raise ValueError("hash_dim already chooses the categorical encoding")
        return "hash"
    if code_dim is not None:
        if encoding is not CatEncoding.RADEMACHER:
            raise ValueError("code_dim is a Rademacher width")
        return "code_dim"
    return "cards"


def _hash_width(hash_dim: int) -> int:
    width = int(hash_dim)
    if width < 1:
        raise ValueError(f"hash_dim must be >= 1, got {width}")
    return width


def _card_at(kind: DimKind, card, need: bool) -> int | None:
    if kind is DimKind.CONTINUOUS:
        if card is not None:
            raise ValueError("a continuous column has no card")
        return None
    if card is None:
        if need:
            raise ValueError("a categorical column needs a card")
        return None
    return _onehot_width(card)


def _align_cards(kinds, cards, need: bool) -> tuple[int | None, ...]:
    if cards is None:
        if need:
            raise ValueError("categorical columns need cards")
        return tuple(None for _ in kinds)
    if len(cards) != len(kinds):
        raise ValueError(f"cards has length {len(cards)}, expected {len(kinds)}")
    return tuple(_card_at(kind, card, need) for kind, card in zip(kinds, cards))


def _count_categorical(kinds: Sequence[DimKind]) -> int:
    return sum(kind is DimKind.CATEGORICAL for kind in kinds)


def _blocks_from_cards(kinds, cards, encoding, dim_cap) -> tuple[tuple, ...]:
    blocks = []
    var = 0
    pairs = zip(kinds, cards)
    for src, (kind, card) in enumerate(pairs):
        if kind is DimKind.CONTINUOUS:
            blocks.append((src, 1, "continuous", -1))
            continue
        width, style = column_plan(card, encoding, dim_cap)
        blocks.append((src, width, style, var))
        var += 1
    return tuple(blocks)


def _blocks_fixed(kinds, width: int, style: str) -> tuple[tuple, ...]:
    if _count_categorical(kinds) == 0:
        raise ValueError("no categorical columns to encode")
    blocks = []
    var = 0
    for src, kind in enumerate(kinds):
        if kind is DimKind.CONTINUOUS:
            blocks.append((src, 1, "continuous", -1))
            continue
        blocks.append((src, width, style, var))
        var += 1
    return tuple(blocks)


def _continuous_blocks(kinds) -> tuple[tuple, ...]:
    blocks = [
        (src, 1, "continuous", -1)
        for src, kind in enumerate(kinds)
        if kind is DimKind.CONTINUOUS
    ]
    return tuple(blocks)


def _check_separation(separation: float) -> float:
    value = float(separation)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"separation must be positive and finite, got {separation}")
    return value


def _as_matrix(values, n_cols: int) -> np.ndarray:
    rows = np.asarray(values)
    if rows.ndim != 2 or rows.shape[1] != n_cols:
        raise ValueError(f"x shape must be (n, {n_cols}), got {rows.shape}")
    return rows


def _category_column(column) -> np.ndarray:
    values = np.asarray(column)
    if values.dtype.kind == "f" and np.any(values != np.floor(values)):
        raise ValueError("categorical values must be integers")
    return np.ascontiguousarray(values, dtype=np.int64)


def _write_block(out, rows, block, cursor: int, seed: int, separation: float) -> int:
    src, width, style, var = block
    nxt = cursor + width
    if style == "continuous":
        out[:, cursor] = np.asarray(rows[:, src], dtype=np.float64)
    elif style == "onehot":
        write_onehot(out, _category_column(rows[:, src]), cursor, width)
        out[:, cursor:nxt] *= separation
    else:
        coded = rademacher(_category_column(rows[:, src]), var, width, seed, separation)
        out[:, cursor:nxt] = coded
    return nxt


def _categorical_matrix(rows, kinds) -> np.ndarray:
    idx = [i for i, kind in enumerate(kinds) if kind is DimKind.CATEGORICAL]
    if rows.shape[0] == 0:
        return np.zeros((0, len(idx)), dtype=np.int64)
    columns = [_category_column(rows[:, i]) for i in idx]
    return np.column_stack(columns)


class CatEncoder:
    """Expand a mixed row into continuous columns and categorical blocks.

    ``kinds`` follows the columns of ``x``. ``cards`` is the same length:
    ``None`` for a continuous column and the domain size for a categorical
    one. ``encoding`` selects one-hot, Rademacher, or Auto. ``seed`` fixes
    the Rademacher and hash vectors.
    """

    def __init__(
        self,
        kinds: Sequence[DimKind],
        cards: Sequence[int | None] | None = None,
        *,
        encoding: CatEncoding = CatEncoding.AUTO,
        code_dim: int | None = None,
        hash_dim: int | None = None,
        seed: int = 0,
        dim_cap: int = 48,
        separation: float = 10.0,
    ) -> None:
        if int(seed) < 0:
            raise ValueError(f"seed must be >= 0, got {seed}")
        chosen = _require(encoding, CatEncoding, "encoding")
        parsed = _parse_kinds(kinds)
        layout, hashed = _layout(parsed, cards, chosen, code_dim, hash_dim, dim_cap)
        self.kinds = parsed
        self.encoding = chosen
        self._blocks = layout
        self.hash_dim = hashed
        self.seed = int(seed)
        self.separation = _check_separation(separation)

    @property
    def num_dim(self) -> int:
        return self.hash_dim + sum(block[1] for block in self._blocks)

    @property
    def styles(self) -> tuple[str, ...]:
        """Per categorical column: ``onehot`` or ``code``. Empty when hashed."""
        return tuple(block[2] for block in self._blocks if block[2] != "continuous")

    @property
    def code_dims(self) -> tuple[int, ...]:
        """Width of each categorical block. Empty when the columns are hashed."""
        return tuple(block[1] for block in self._blocks if block[2] != "continuous")

    def tied_dims(self) -> tuple[tuple[int, ...], ...]:
        """One group per categorical variable, or one group for a hash.

        Tied columns stay at scale 1 under ``scale_x`` and share one metric weight.
        """
        if self.hash_dim:
            start = sum(block[1] for block in self._blocks)
            return (tuple(range(start, start + self.hash_dim)),)
        groups = []
        cursor = 0
        for block in self._blocks:
            if block[2] != "continuous":
                groups.append(tuple(range(cursor, cursor + block[1])))
            cursor += block[1]
        return tuple(groups)

    def encode(self, x) -> np.ndarray:
        """Return ``(n, num_dim)`` float64 rows."""
        rows = _as_matrix(x, len(self.kinds))
        out = np.empty((rows.shape[0], self.num_dim), dtype=np.float64)
        if self.hash_dim:
            _write_hash(out, rows, self)
            return out
        cursor = 0
        for block in self._blocks:
            cursor = _write_block(out, rows, block, cursor, self.seed, self.separation)
        return out

    def snap(self, encoded, domains: Sequence[np.ndarray]) -> np.ndarray:
        """Nearest domain id for each categorical column of ``encoded``.

        ``domains[j]`` lists the ids allowed for the ``j``-th categorical
        column, skipping continuous columns. A hash has no per-variable block.
        """
        if self.hash_dim:
            raise ValueError("hashed codes are not separated by variable")
        cats = [block for block in self._blocks if block[2] != "continuous"]
        if len(domains) != len(cats):
            raise ValueError(f"domains has length {len(domains)}, expected {len(cats)}")
        rows = np.asarray(encoded, dtype=np.float64)
        if rows.ndim != 2 or rows.shape[1] != self.num_dim:
            raise ValueError(f"encoded shape must be (n, {self.num_dim})")
        chosen = np.empty((rows.shape[0], len(cats)), dtype=np.int64)
        _fill_snap(chosen, rows, self._blocks, domains, self.seed, self.separation)
        return chosen


def _reject_cards(cards, label: str) -> None:
    if cards is not None:
        raise ValueError(f"{label} does not use cards")


def _layout(kinds, cards, encoding, code_dim, hash_dim, dim_cap):
    mode = _mode(encoding, code_dim, hash_dim)
    if mode == "hash":
        _reject_cards(cards, "hash_dim")
        if _count_categorical(kinds) == 0:
            raise ValueError("no categorical columns to encode")
        return _continuous_blocks(kinds), _hash_width(hash_dim)
    if mode == "code_dim":
        _reject_cards(cards, "code_dim")
        return _blocks_fixed(
            kinds, width_in_range(code_dim, "code_dim", CODE_BIT), "code"
        ), 0
    cap = width_in_range(dim_cap, "dim_cap", CODE_BIT)
    aligned = _align_cards(kinds, cards, need=_count_categorical(kinds) > 0)
    return _blocks_from_cards(kinds, aligned, encoding, cap), 0


def _write_hash(out, rows, encoder: CatEncoder) -> None:
    cursor = 0
    for block in encoder._blocks:
        out[:, cursor] = np.asarray(rows[:, block[0]], dtype=np.float64)
        cursor += 1
    cats = _categorical_matrix(rows, encoder.kinds)
    out[:, cursor:] = hash_columns(cats, encoder.hash_dim, encoder.seed)


def _fill_snap(chosen, rows, blocks, domains, seed: int, separation: float) -> None:
    cursor = 0
    cat = 0
    for block in blocks:
        width, style, var = block[1], block[2], block[3]
        if style != "continuous":
            piece = rows[:, cursor : cursor + width]
            chosen[:, cat] = _nearest(
                piece, domains[cat], style, var, width, seed, separation
            )
            cat += 1
        cursor += width


def _nearest(
    piece, domain, style: str, var: int, width: int, seed: int, separation: float
):
    if style == "onehot":
        return snap_onehot(piece, domain, width)
    return snap_code(piece, domain, var, width, seed, separation)
