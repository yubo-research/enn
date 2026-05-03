from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


def _pad_neighbor_cols_to_search_k(
    dist2s: np.ndarray,
    idx: np.ndarray,
    *,
    search_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np

    k_eff = int(dist2s.shape[1])
    if k_eff >= search_k:
        return dist2s, idx
    if k_eff == 0:
        n_query = int(dist2s.shape[0])
        return (
            np.full((n_query, search_k), np.inf, dtype=float),
            np.zeros((n_query, search_k), dtype=int),
        )
    n_query = int(dist2s.shape[0])
    pad_w = search_k - k_eff
    pad_dist = np.full((n_query, pad_w), np.inf, dtype=float)
    far_idx = idx[:, k_eff - 1 : k_eff]
    pad_idx = np.tile(far_idx, (1, pad_w))
    return (
        np.concatenate([dist2s, pad_dist], axis=1),
        np.concatenate([idx, pad_idx], axis=1),
    )


def _empty_train_neighbor_slots(
    n_query: int, search_k: int
) -> tuple[np.ndarray, np.ndarray]:
    import numpy as np

    return (
        np.full((n_query, search_k), np.inf, dtype=float),
        np.zeros((n_query, search_k), dtype=int),
    )


class ENNIndex:
    def __init__(
        self,
        train_x_scaled: np.ndarray,
        num_dim: int,
        x_scale: np.ndarray,
        scale_x: bool,
        driver: Any = None,
    ) -> None:
        from enn.turbo.config.enn_index_driver import ENNIndexDriver

        if driver is None:
            driver = ENNIndexDriver.FLAT
        self._train_x_scaled = train_x_scaled
        self._num_dim = num_dim
        self._x_scale = x_scale
        self._scale_x = scale_x
        self._driver = driver
        self._index: Any | None = None
        self._build_index()

    def _build_index(self) -> None:
        import numpy as np

        from enn.turbo.config.enn_index_driver import ENNIndexDriver

        if len(self._train_x_scaled) == 0:
            return
        import faiss

        x_f32 = self._train_x_scaled.astype(np.float32, copy=False)
        if self._driver == ENNIndexDriver.FLAT:
            index = faiss.IndexFlatL2(self._num_dim)
        elif self._driver == ENNIndexDriver.HNSW:
            index = faiss.IndexHNSWFlat(self._num_dim, 32)
        else:
            raise ValueError(f"Unknown driver: {self._driver}")
        index.add(x_f32)
        self._index = index

    def add(self, x: np.ndarray) -> None:
        import numpy as np

        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self._num_dim:
            raise ValueError(x.shape)
        x_scaled = x / self._x_scale if self._scale_x else x
        x_f32 = x_scaled.astype(np.float32, copy=False)
        self._train_x_scaled = np.concatenate([self._train_x_scaled, x_f32], axis=0)
        if self._index is not None:
            self._index.add(x_f32)
        elif len(self._train_x_scaled) > 0:
            self._build_index()

    def search(
        self,
        x: np.ndarray,
        *,
        search_k: int,
        exclude_nearest: bool,
    ) -> tuple[np.ndarray, np.ndarray]:
        import numpy as np

        search_k = int(search_k)
        if search_k <= 0:
            raise ValueError(search_k)
        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self._num_dim:
            raise ValueError(x.shape)
        x_scaled = x / self._x_scale if self._scale_x else x
        x_f32 = x_scaled.astype(np.float32, copy=False)
        n_query = int(x_f32.shape[0])
        n_train = len(self._train_x_scaled)
        if self._index is not None:
            k_eff = min(search_k, n_train)
            dist2s_full, idx_full = self._index.search(x_f32, k_eff)
            dist2s_full = dist2s_full.astype(float)
            idx_full = idx_full.astype(int)
            dist2s_full, idx_full = _pad_neighbor_cols_to_search_k(
                dist2s_full, idx_full, search_k=search_k
            )
        else:
            dist2s_full, idx_full = _empty_train_neighbor_slots(n_query, search_k)
        if exclude_nearest:
            dist2s_full = dist2s_full[:, 1:]
            idx_full = idx_full[:, 1:]
        return dist2s_full, idx_full
