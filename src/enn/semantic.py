"""Fixed, objective-independent semantic features for bounded numerical inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True, eq=False, init=False)
class SemanticInputTransform:
    """Construct once per ordered problem definition; reuse or save the projection.

    Bounds have shape (D, 2). ``relative_weight`` is the expected semantic/original
    squared-distance ratio for independent inputs with equal normalized variance.
    ``variance_cutoff`` retains the shortest SVD prefix explaining this fraction
    of the variance above the numerical-rank tolerance.
    An optional encoder implements ``encode(texts)``. Default construction uses
    sentence-transformers-rs on the CPU and caches downloaded Hugging Face model
    files. Inference and loading never load the sentence transformer.
    """

    names: tuple[str, ...]
    bounds: np.ndarray
    model: str
    relative_weight: float
    variance_cutoff: float
    projection: np.ndarray

    def __init__(
        self,
        names,
        bounds,
        descriptions=None,
        *,
        model="sentence-transformers/all-MiniLM-L6-v2",
        variance_cutoff=0.9,
        relative_weight=0.25,
        encoder=None,
    ):
        self._set_definition(names, bounds, model, relative_weight, variance_cutoff)
        from enn._rust import semantic_text_projection, semantic_texts

        texts, rows = semantic_texts(self.names, descriptions)
        embeddings = self._encode(texts, encoder)
        object.__setattr__(
            self,
            "projection",
            semantic_text_projection(embeddings, rows, self.variance_cutoff),
        )
        self._freeze()

    @classmethod
    def from_input_output(
        cls,
        names,
        bounds,
        output_names,
        descriptions=None,
        output_descriptions=None,
        *,
        model="sentence-transformers/all-MiniLM-L6-v2",
        variance_cutoff=0.9,
        relative_weight=0.25,
        encoder=None,
    ):
        """Append shared x-y features, centered cosine SVD factors.

        This is an alternative to the x-x prior built by the normal constructor.
        All output features are shared by all surrogate outputs. Descriptions
        follow the same fallback and file URI rules as input descriptions.
        Construction uses only metadata; observed targets are never required.
        The returned fixed transform supports the same models, optimizers, and
        save/load operations. Keep the default ``semantic_only=False`` to retain
        original inputs alongside this prior. The variance cutoff selects the retained factor count.
        """
        from enn._rust import semantic_input_output_projection
        from enn.semantic_relationships import relationship_texts

        result = cls.__new__(cls)
        result._set_definition(names, bounds, model, relative_weight, variance_cutoff)
        texts, rows = relationship_texts(
            result.names, output_names, descriptions, output_descriptions
        )
        embeddings = result._encode(texts, encoder)
        object.__setattr__(
            result,
            "projection",
            semantic_input_output_projection(
                embeddings, rows, len(result.names), result.variance_cutoff
            ),
        )
        result._freeze()
        return result

    def _set_definition(self, names, bounds, model, relative_weight, variance_cutoff):
        object.__setattr__(self, "names", tuple(names))
        object.__setattr__(self, "bounds", np.array(bounds, dtype=float, copy=True))
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "relative_weight", float(relative_weight))
        object.__setattr__(self, "variance_cutoff", float(variance_cutoff))
        self._validate_definition()

    def _encode(self, texts, encoder):
        from enn._rust import semantic_encode

        embeddings = np.asarray(
            semantic_encode(texts, self.model)
            if encoder is None
            else encoder.encode(texts),
            dtype=float,
        )
        if embeddings.ndim != 2:
            raise ValueError("encoder must return a two-dimensional embedding matrix")
        return embeddings

    def _validate_definition(self):
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
        if not np.isfinite(self.variance_cutoff) or not 0 < self.variance_cutoff <= 1:
            raise ValueError("variance_cutoff must be finite and in (0, 1]")
        if not np.isfinite(self.relative_weight) or self.relative_weight < 0:
            raise ValueError("relative_weight must be finite and nonnegative")

    def _freeze(self):
        self.bounds.setflags(write=False)
        self.projection.setflags(write=False)

    @property
    def weighted_projection(self):
        from enn._rust import semantic_weighted

        return semantic_weighted(self.projection, self.relative_weight)

    def transform(self, x, *, names=None, semantic_only=False):
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
        from enn._rust import semantic_transform

        result = semantic_transform(
            np.atleast_2d(values), self.bounds, self.weighted_projection, semantic_only
        )
        return result[0] if values.ndim == 1 else result

    def save(self, path):
        metadata = json.dumps(
            {
                "names": self.names,
                "model": self.model,
                "relative_weight": self.relative_weight,
                "variance_cutoff": self.variance_cutoff,
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
            object.__setattr__(
                result, "variance_cutoff", float(meta.get("variance_cutoff", 0.9))
            )
            object.__setattr__(result, "bounds", data["bounds"].copy())
            object.__setattr__(result, "projection", data["projection"].copy())
        if (
            result.projection.ndim != 2
            or result.projection.shape[0] != len(result.names)
            or not np.isfinite(result.projection).all()
        ):
            raise ValueError("invalid cached projection")
        result._validate_definition()
        result._freeze()
        return result
