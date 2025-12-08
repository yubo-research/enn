import torch

from sampling.sparse_jl_t import (
    _block_sparse_hash_scatter_from_nz_t,
    block_sparse_jl_transform_t,
)


class DeltaSparseJL_T:
    def __init__(
        self,
        num_dim_ambient: int,
        num_dim_embedding: int,
        s: int = 4,
        seed: int = 42,
        incremental: bool = False,
    ):
        assert isinstance(num_dim_ambient, int) and num_dim_ambient > 0
        assert isinstance(num_dim_embedding, int) and num_dim_embedding > 0
        assert isinstance(s, int) and s > 0
        if s > num_dim_embedding:
            raise ValueError("s must be <= num_dim_embedding")
        self.num_dim_ambient = num_dim_ambient
        self.num_dim_embedding = num_dim_embedding
        self.s = s
        self.seed = int(seed)
        self.incremental = bool(incremental)
        self._initialized = False
        self._x0 = None
        self._y0 = None

    def initialize(self, x_0: torch.Tensor):
        assert not self._initialized
        assert torch.is_tensor(x_0)
        assert x_0.ndim == 1
        assert x_0.shape[0] == self.num_dim_ambient
        self._x0 = x_0
        if self.incremental:
            self._y0 = block_sparse_jl_transform_t(
                x_0, d=self.num_dim_embedding, s=self.s, seed=self.seed
            )
        self._initialized = True

    def transform(self, d_x: torch.Tensor) -> torch.Tensor:
        assert self._initialized
        assert torch.is_tensor(d_x) and d_x.is_sparse
        assert d_x.ndim == 1
        assert d_x.shape[0] == self.num_dim_ambient
        assert d_x.device == self._x0.device
        assert d_x.dtype == self._x0.dtype
        if not self.incremental:
            x = self._x0 + d_x.to_dense()
            y = block_sparse_jl_transform_t(
                x, d=self.num_dim_embedding, s=self.s, seed=self.seed
            )
            assert y.shape == (self.num_dim_embedding,)
            return y
        if d_x._nnz() == 0:
            return self._y0.clone()
        idx = d_x.coalesce()._indices().squeeze(0)
        vals = d_x.coalesce()._values()
        y_delta = _block_sparse_hash_scatter_from_nz_t(
            nz_indices=idx,
            nz_values=vals,
            d=self.num_dim_embedding,
            s=self.s,
            seed=self.seed,
            dtype=self._x0.dtype,
            device=self._x0.device,
        )
        return self._y0 + y_delta
