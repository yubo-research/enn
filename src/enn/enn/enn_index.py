from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np


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
        import faiss
        import numpy as np

        from enn.turbo.config.enn_index_driver import ENNIndexDriver

        if len(self._train_x_scaled) == 0:
            return
        x_f32 = self._train_x_scaled.astype(np.float32, copy=False)
        if self._driver == ENNIndexDriver.FLAT:
            index = faiss.IndexFlatL2(self._num_dim)
        elif self._driver == ENNIndexDriver.HNSW:
            # TODO: Make M configurable
            index = faiss.IndexHNSWFlat(self._num_dim, 32)
        else:
            raise ValueError(f"Unknown driver: {self._driver}")
        index.add(x_f32)
        self._index = index

    def add(self, x: np.ndarray) -> None:
        import numpy as np

        from enn.turbo.config.enn_index_driver import ENNIndexDriver

        x = np.asarray(x, dtype=float)
        if x.ndim != 2 or x.shape[1] != self._num_dim:
            raise ValueError(x.shape)
        x_scaled = x / self._x_scale if self._scale_x else x
        x_f32 = x_scaled.astype(np.float32, copy=False)
        if self._index is None:
            import faiss

            if self._driver == ENNIndexDriver.FLAT:
                self._index = faiss.IndexFlatL2(self._num_dim)
            elif self._driver == ENNIndexDriver.HNSW:
                self._index = faiss.IndexHNSWFlat(self._num_dim, 32)
            else:
                raise ValueError(f"Unknown driver: {self._driver}")
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
        if self._index is None:
            raise RuntimeError("index is not initialized")
        x_scaled = x / self._x_scale if self._scale_x else x
        x_f32 = x_scaled.astype(np.float32, copy=False)
        dist2s_full, idx_full = self._index.search(x_f32, search_k)
        dist2s_full = dist2s_full.astype(float)
        idx_full = idx_full.astype(int)
        if exclude_nearest:
            dist2s_full = dist2s_full[:, 1:]
            idx_full = idx_full[:, 1:]
        return dist2s_full, idx_full
