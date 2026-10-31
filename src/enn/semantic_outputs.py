import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

from enn._rust import semantic_encode, semantic_output_projection, semantic_texts


def _output_names(names):
    names = tuple(names)
    if (
        len(names) < 2
        or any(not isinstance(n, str) or not n.strip() for n in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("at least two unique nonempty output names are required")
    return names


def _cutoff(value):
    value = float(value)
    if not np.isfinite(value) or not 0 < value <= 1:
        raise ValueError("variance_cutoff must be finite and in (0, 1]")
    return value


def _normalized(embeddings, outputs):
    values = np.array(embeddings, dtype=float, copy=True)
    if (
        values.ndim != 2
        or values.shape[0] != outputs
        or values.shape[1] == 0
        or not np.isfinite(values).all()
    ):
        raise ValueError("embeddings must be finite nonempty rows, one per output")
    values /= np.maximum(np.max(np.abs(values), axis=1, keepdims=True), 1e-300)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.where(norms > 0, norms, 1)


@dataclass(frozen=True, eq=False)
class SemanticOutputProjection:
    names: tuple[str, ...]
    embeddings: np.ndarray
    loadings: np.ndarray

    def __post_init__(self):
        names = _output_names(self.names)
        embeddings = _normalized(self.embeddings, len(names))
        loadings = np.array(self.loadings, dtype=float, copy=True)
        if (
            loadings.ndim != 2
            or loadings.shape[0] != len(names)
            or loadings.shape[1] >= len(names)
            or not np.isfinite(loadings).all()
        ):
            raise ValueError("invalid output loadings")
        if not np.allclose(loadings.T @ loadings, np.eye(loadings.shape[1])):
            raise ValueError("output loadings must have orthonormal columns")
        embeddings.setflags(write=False)
        loadings.setflags(write=False)
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "embeddings", embeddings)
        object.__setattr__(self, "loadings", loadings)

    @classmethod
    def from_embeddings(cls, names, embeddings, *, variance_cutoff=0.9):
        names = _output_names(names)
        variance_cutoff = _cutoff(variance_cutoff)
        embeddings = _normalized(embeddings, len(names))
        loadings = semantic_output_projection(embeddings, variance_cutoff)
        return cls(names, embeddings, loadings)

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
        names = _output_names(names)
        variance_cutoff = _cutoff(variance_cutoff)
        texts, rows = semantic_texts(names, descriptions)
        path = _cache_path(
            cache_dir, names, texts, rows, model, variance_cutoff, encoder
        )
        if path is not None and path.exists():
            return cls.load(path)
        encoded = (
            semantic_encode(texts, model) if encoder is None else encoder.encode(texts)
        )
        encoded = np.asarray(encoded, dtype=float)
        if encoded.ndim != 2 or encoded.shape[0] != len(texts):
            raise ValueError("encoder must return one embedding row per text")
        result = cls.from_embeddings(
            names, encoded[rows], variance_cutoff=variance_cutoff
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
    return Path(directory) / (sha256(key.encode()).hexdigest() + ".npz")
