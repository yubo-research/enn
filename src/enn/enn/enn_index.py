from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


def _use_faiss() -> bool:
    import os
    import sys

    if sys.platform == "darwin":
        return os.environ.get("ENN_USE_FAISS", "0") == "1"
    return True


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
        if _use_faiss():
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
        else:
            self._index = None

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
        if self._index is not None:
            dist2s_full, idx_full = self._index.search(x_f32, search_k)
            dist2s_full = dist2s_full.astype(float)
            idx_full = idx_full.astype(int)
        else:
            n_query = x_f32.shape[0]
            n_train = len(self._train_x_scaled)
            k = min(search_k, n_train)
            dist2s_full = np.full((n_query, search_k), np.inf, dtype=float)
            idx_full = np.full((n_query, search_k), -1, dtype=int)
            if k:
                x2 = np.sum(x_f32**2, axis=1, keepdims=True)
                y2 = np.sum(self._train_x_scaled**2, axis=1, keepdims=True).T
                d2 = x2 + y2 - 2.0 * (x_f32 @ self._train_x_scaled.T)
                part = np.argpartition(d2, kth=k - 1, axis=1)[:, :k]
                rows = np.arange(n_query)[:, None]
                d2_part = d2[rows, part]
                order = np.argsort(d2_part, axis=1)
                part_sorted = part[rows, order]
                d2_sorted = d2_part[rows, order]
                idx_full[:, :k] = part_sorted.astype(int)
                dist2s_full[:, :k] = d2_sorted.astype(float)
        if exclude_nearest:
            dist2s_full = dist2s_full[:, 1:]
            idx_full = idx_full[:, 1:]
        return dist2s_full, idx_full
