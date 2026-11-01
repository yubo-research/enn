from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
        from enn._rust import (
            semantic_input_output_projection,
            semantic_relationship_texts,
        )

        result = cls.__new__(cls)
        result._set_definition(names, bounds, model, relative_weight, variance_cutoff)
        texts, rows = semantic_relationship_texts(
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

        return np.asarray(
            semantic_encode(texts, self.model)
            if encoder is None
            else encoder.encode(texts),
            dtype=float,
        )

    def _validate_definition(self):
        from enn._rust import validate_semantic_definition

        validate_semantic_definition(
            self.names, self.bounds, self.variance_cutoff, self.relative_weight
        )

    def _freeze(self):
        self.bounds.setflags(write=False)
        self.projection.setflags(write=False)

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
        from enn._rust import validate_semantic_cache

        validate_semantic_cache(result.projection, len(result.names))
        result._validate_definition()
        result._freeze()
        return result
