import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

from enn._rust import (
    semantic_encode,
    semantic_output_definition,
    semantic_output_embeddings,
    semantic_output_projection,
    semantic_texts,
    validate_semantic_outputs,
)


@dataclass(frozen=True, eq=False)
class SemanticOutputProjection:
    names: tuple[str, ...]
    embeddings: np.ndarray
    loadings: np.ndarray

    def __post_init__(self):
        names, embeddings, loadings = semantic_output_definition(
            self.names,
            np.asarray(self.embeddings, dtype=float),
            np.asarray(self.loadings, dtype=float),
        )
        embeddings.setflags(write=False)
        loadings.setflags(write=False)
        object.__setattr__(self, "names", tuple(names))
        object.__setattr__(self, "embeddings", embeddings)
        object.__setattr__(self, "loadings", loadings)

    @classmethod
    def from_embeddings(cls, names, embeddings, *, variance_cutoff=0.9):
        names = tuple(names)
        values = np.asarray(embeddings, dtype=float)
        loadings = semantic_output_projection(names, values, float(variance_cutoff))
        return cls(names, values, loadings)

    @classmethod
    def from_metadata(
        cls,
        names,
        descriptions=None,
        *,
        variance_cutoff=0.9,
        model="sentence-transformers/all-MiniLM-L6-v2",
        encoder=None,
        cache_dir=None,
    ):
        names = tuple(names)
        variance_cutoff = float(variance_cutoff)
        validate_semantic_outputs(names, variance_cutoff)
        texts, rows = semantic_texts(names, descriptions)
        path = _cache_path(
            cache_dir, names, texts, rows, model, variance_cutoff, encoder
        )
        if path is not None and path.exists():
            return cls.load(path)
        encoded = (
            semantic_encode(texts, model) if encoder is None else encoder.encode(texts)
        )
        result = cls.from_embeddings(
            names,
            semantic_output_embeddings(
                np.asarray(encoded, dtype=float), len(texts), rows
            ),
            variance_cutoff=variance_cutoff,
        )
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            result.save(path)
        return result

    def save(self, path):
        with Path(path).open("wb") as stream:
            np.savez(
                stream,
                names=self.names,
                embeddings=self.embeddings,
                loadings=self.loadings,
                version=1,
            )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            if int(data["version"]) != 1:
                raise ValueError("unsupported semantic output projection version")
            return cls(tuple(data["names"]), data["embeddings"], data["loadings"])


def _cache_path(directory, names, texts, rows, model, variance_cutoff, encoder):
    if directory is None or encoder is not None:
        return None
    key = json.dumps([2, names, texts, list(map(int, rows)), model, variance_cutoff])
    return Path(directory).joinpath(f"{sha256(key.encode()).hexdigest()}.npz")
