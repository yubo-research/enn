from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from enn.semantic_defaults import (
    DEFAULT_SEMANTIC_MODEL,
    DEFAULT_VARIANCE_CUTOFF,
    SEMANTIC_FORMAT_VERSION,
)


@dataclass(frozen=True, eq=False, init=False)
class SemanticInputTransform:
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
        model=DEFAULT_SEMANTIC_MODEL,
        variance_cutoff=DEFAULT_VARIANCE_CUTOFF,
        relative_weight=0.25,
        encoder=None,
    ):
        definition = _definition(names, bounds, model, relative_weight, variance_cutoff)
        from enn._rust import semantic_text_projection, semantic_texts

        texts, rows = semantic_texts(definition[0], descriptions)
        embeddings = _encode(texts, definition[2], encoder)
        self._assign(
            definition, semantic_text_projection(embeddings, rows, definition[4])
        )

    @classmethod
    def from_input_output(
        cls,
        names,
        bounds,
        output_names,
        descriptions=None,
        output_descriptions=None,
        *,
        model=DEFAULT_SEMANTIC_MODEL,
        variance_cutoff=DEFAULT_VARIANCE_CUTOFF,
        relative_weight=0.25,
        encoder=None,
    ):
        from enn._rust import (
            semantic_input_output_projection,
            semantic_relationship_texts,
        )

        definition = _definition(names, bounds, model, relative_weight, variance_cutoff)
        texts, rows = semantic_relationship_texts(
            definition[0], output_names, descriptions, output_descriptions
        )
        embeddings = _encode(texts, definition[2], encoder)
        projection = semantic_input_output_projection(
            embeddings, rows, len(definition[0]), definition[4]
        )
        result = cls.__new__(cls)
        result._assign(definition, projection)
        return result

    def _assign(self, definition, projection):
        names, bounds, model, relative_weight, variance_cutoff = definition
        projection = np.array(projection, dtype=float, copy=True)
        bounds.setflags(write=False)
        projection.setflags(write=False)
        for key, value in (
            ("names", names),
            ("bounds", bounds),
            ("model", model),
            ("relative_weight", relative_weight),
            ("variance_cutoff", variance_cutoff),
            ("projection", projection),
        ):
            object.__setattr__(self, key, value)

    @property
    def weighted_projection(self):
        from enn._rust import semantic_weighted

        return semantic_weighted(self.projection, self.relative_weight)

    def transform(self, x, *, names=None, semantic_only=False):
        from enn._rust import semantic_transform

        return semantic_transform(
            np.asarray(x, dtype=float),
            self.names,
            names,
            self.bounds,
            self.weighted_projection,
            semantic_only,
        )

    def save(self, path):
        metadata = json.dumps(
            {
                "names": self.names,
                "model": self.model,
                "relative_weight": self.relative_weight,
                "variance_cutoff": self.variance_cutoff,
                "version": SEMANTIC_FORMAT_VERSION,
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
            if meta.pop("version") != SEMANTIC_FORMAT_VERSION:
                raise ValueError("unsupported semantic transform version")
            definition = _definition(
                meta["names"],
                data["bounds"],
                meta["model"],
                meta["relative_weight"],
                meta["variance_cutoff"],
            )
            projection = data["projection"].copy()
        from enn._rust import validate_semantic_cache

        validate_semantic_cache(projection, len(definition[0]))
        result = cls.__new__(cls)
        result._assign(definition, projection)
        return result


def _definition(names, bounds, model, relative_weight, variance_cutoff):
    from enn._rust import validate_semantic_definition

    names = tuple(names)
    bounds = np.array(bounds, dtype=float, copy=True)
    relative_weight = float(relative_weight)
    variance_cutoff = float(variance_cutoff)
    validate_semantic_definition(names, bounds, variance_cutoff, relative_weight)
    return names, bounds, model, relative_weight, variance_cutoff


def _encode(texts, model, encoder):
    from enn._rust import semantic_encode

    return np.asarray(
        semantic_encode(texts, model) if encoder is None else encoder.encode(texts),
        dtype=float,
    )
