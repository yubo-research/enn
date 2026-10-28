"""Fixed, metadata-only output loadings, independent of observed targets."""

import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np

from enn._rust import semantic_encode, semantic_texts


def _output_names(names):
    names = tuple(names)
    if (
        len(names) < 2
        or any(not isinstance(n, str) or not n.strip() for n in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("at least two unique nonempty output names are required")
    return names


def _rank(rank, outputs):
    if rank is None:
        return min(2, outputs - 1)
    if isinstance(rank, bool) or not isinstance(rank, (int, np.integer)):
        raise TypeError("rank must be an integer")
    if not 0 < rank < outputs:
        raise ValueError("rank must be positive and smaller than the output count")
    return int(rank)


def _normalized(embeddings, outputs):
    """Rescale first to avoid overflow when normalizing supplied embeddings."""
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
    """Normalized sentence embeddings and orthonormal uncentered SVD loadings.

    Use ``from_metadata`` for pretrained embeddings; ``from_embeddings`` supports
    controlled ablations. Numerically zero singular directions are discarded.
    Omitted rank uses min(2, output_count - 1).
    Zero embeddings produce no factors and the wrapper falls back to independent
    outputs. Repeated descriptions retain their common direction.
    """

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
    def from_embeddings(cls, names, embeddings, *, rank=None):
        names = _output_names(names)
        rank = _rank(rank, len(names))
        embeddings = _normalized(embeddings, len(names))
        u, singular, _ = np.linalg.svd(embeddings, full_matrices=False)
        tolerance = np.finfo(float).eps * max(embeddings.shape) * singular[0]
        count = min(rank, int(np.sum(singular > tolerance)))
        return cls(names, embeddings, u[:, :count])

    @classmethod
    def from_metadata(
        cls,
        names,
        descriptions=None,
        *,
        rank=None,
        model="sentence-transformers/all-MiniLM-L6-v2",
        encoder=None,
        cache_dir=None,
    ):
        """Encode once; optional disk cache keys include texts, model, and rank.

        A custom encoder is never looked up in the automatic cache because its
        identity is unknown. Save/load its projection explicitly instead.
        Descriptions fall back to names and support file URIs like input metadata.
        """
        names = _output_names(names)
        rank = _rank(rank, len(names))
        texts, rows = semantic_texts(names, descriptions)
        path = _cache_path(cache_dir, names, texts, rows, model, rank, encoder)
        if path is not None and path.exists():
            return cls.load(path)
        encoded = (
            semantic_encode(texts, model) if encoder is None else encoder.encode(texts)
        )
        encoded = np.asarray(encoded, dtype=float)
        if encoded.ndim != 2 or encoded.shape[0] != len(texts):
            raise ValueError("encoder must return one embedding row per text")
        result = cls.from_embeddings(names, encoded[rows], rank=rank)
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


def _cache_path(directory, names, texts, rows, model, rank, encoder):
    if directory is None or encoder is not None:
        return None
    key = json.dumps([1, names, texts, list(map(int, rows)), model, rank])
    return Path(directory) / (sha256(key.encode()).hexdigest() + ".npz")
