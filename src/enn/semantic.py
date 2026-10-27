"""Fixed, objective-independent semantic features for bounded numerical inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

import numpy as np


def _description(name, description):
    if description and description.startswith("file://"):
        uri = urlparse(description)
        if uri.netloc not in ("", "localhost"):
            raise ValueError("description files must be local")
        description = Path(unquote(uri.path)).read_text(encoding="utf-8")
    return description.strip() if description and description.strip() else name


def _projection(embeddings, dimensions):
    vectors = np.asarray(embeddings, dtype=float)
    if vectors.ndim != 2 or vectors.shape[1] == 0 or not np.isfinite(vectors).all():
        raise ValueError("embeddings must be a finite nonempty matrix")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    vectors = vectors / np.where(norms > 0, norms, 1)
    vectors -= vectors.mean(axis=0)
    u, singular, _ = np.linalg.svd(vectors, full_matrices=False)
    rank = np.count_nonzero(singular > max(vectors.shape) * np.finfo(float).eps)
    retained = min(dimensions, rank)
    projection = u[:, :retained] * singular[:retained]
    for column in projection.T:
        column *= 1 if column[np.argmax(np.abs(column))] >= 0 else -1
    norm = np.linalg.norm(projection)
    return projection * np.sqrt(retained) / norm if norm else projection


@dataclass(frozen=True, eq=False, init=False)
class SemanticInputTransform:
    """Construct once per ordered problem definition; reuse or save the projection.

    Bounds have shape (D, 2). ``relative_weight`` is the expected semantic/original
    squared-distance ratio for independent inputs with equal normalized variance.
    An optional encoder implements ``encode(texts)``; default construction alone
    imports sentence-transformers. Inference and loading require only NumPy.
    """

    names: tuple[str, ...]
    bounds: np.ndarray
    model: str
    relative_weight: float
    projection: np.ndarray

    def __init__(
        self,
        names,
        bounds,
        descriptions=None,
        *,
        model="sentence-transformers/all-MiniLM-L6-v2",
        dimensions=8,
        relative_weight=0.25,
        encoder=None,
    ):
        object.__setattr__(self, "names", tuple(names))
        object.__setattr__(self, "bounds", np.array(bounds, dtype=float, copy=True))
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "relative_weight", float(relative_weight))
        self._validate_definition(dimensions)
        descriptions = (
            [None] * len(self.names) if descriptions is None else descriptions
        )
        if len(descriptions) != len(self.names):
            raise ValueError("one description is required per variable")
        texts = [_description(n, d) for n, d in zip(self.names, descriptions)]
        if encoder is None:
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(model, device="cpu")
        unique_texts = list(dict.fromkeys(texts))
        embeddings = np.asarray(encoder.encode(unique_texts), dtype=float)
        if embeddings.ndim != 2 or embeddings.shape[0] != len(unique_texts):
            raise ValueError("encoder returned the wrong number of variables")
        rows = [unique_texts.index(text) for text in texts]
        object.__setattr__(
            self, "projection", _projection(embeddings[rows], dimensions)
        )
        self._freeze()

    def _validate_definition(self, dimensions):
        if (
            not self.names
            or any(not isinstance(n, str) or not n.strip() for n in self.names)
            or len(set(self.names)) != len(self.names)
        ):
            raise ValueError("variable names must be nonempty and unique")
        if (
            self.bounds.shape != (len(self.names), 2)
            or not np.isfinite(self.bounds).all()
            or np.any(self.bounds[:, 1] <= self.bounds[:, 0])
        ):
            raise ValueError("bounds must be finite increasing pairs, one per variable")
        if (
            not isinstance(dimensions, int)
            or isinstance(dimensions, bool)
            or dimensions < 1
        ):
            raise ValueError("dimensions must be a positive integer")
        if not np.isfinite(self.relative_weight) or self.relative_weight < 0:
            raise ValueError("relative_weight must be finite and nonnegative")

    def _freeze(self):
        self.bounds.setflags(write=False)
        self.projection.setflags(write=False)

    @property
    def weighted_projection(self):
        rank = self.projection.shape[1]
        return self.projection * np.sqrt(
            self.relative_weight * len(self.names) / max(rank, 1)
        )

    def transform(self, x, *, names=None):
        """Transform rows in definition order, or explicitly reorder named columns."""
        values = np.asarray(x, dtype=float)
        if (
            values.ndim not in (1, 2)
            or values.shape[-1] != len(self.names)
            or not np.isfinite(values).all()
        ):
            raise ValueError("inputs must be finite rows with one column per variable")
        if names is not None:
            if len(names) != len(self.names) or set(names) != set(self.names):
                raise ValueError("input names must match the problem definition")
            values = values[..., [list(names).index(n) for n in self.names]]
        normalized = (values - self.bounds[:, 0]) / np.diff(
            self.bounds, axis=1
        ).ravel() - 0.5
        return np.concatenate(
            (normalized, normalized @ self.weighted_projection), axis=-1
        )

    def inverse(self, z):
        """Recover original numerical columns from augmented rows."""
        return (np.asarray(z)[..., : len(self.names)] + 0.5) * np.diff(
            self.bounds, axis=1
        ).ravel() + self.bounds[:, 0]

    def save(self, path):
        metadata = json.dumps(
            {
                "names": self.names,
                "model": self.model,
                "relative_weight": self.relative_weight,
                "version": 1,
            }
        )
        with Path(path).open("wb") as stream:
            np.savez(
                stream,
                bounds=self.bounds,
                projection=self.projection,
                metadata=metadata,
            )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            if meta.pop("version") != 1:
                raise ValueError("unsupported semantic transform version")
            result = cls.__new__(cls)
            object.__setattr__(result, "names", tuple(meta["names"]))
            object.__setattr__(result, "model", meta["model"])
            object.__setattr__(
                result, "relative_weight", float(meta["relative_weight"])
            )
            object.__setattr__(result, "bounds", data["bounds"].copy())
            object.__setattr__(result, "projection", data["projection"].copy())
        if (
            result.projection.ndim != 2
            or result.projection.shape[0] != len(result.names)
            or not np.isfinite(result.projection).all()
        ):
            raise ValueError("invalid cached projection")
        result._validate_definition(max(1, result.projection.shape[1]))
        result._freeze()
        return result
