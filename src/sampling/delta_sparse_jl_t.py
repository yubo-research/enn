import torch

from sampling.sparse_jl_t import (
    _block_sparse_hash_scatter_from_nz_t,
    block_sparse_jl_transform_t,
)


class DeltaSparseJL_T:
    _initialized: bool
    _x0: torch.Tensor | None
    _y0: torch.Tensor | None

    def __init__(
        self,
        num_dim_ambient: int,
        num_dim_embedding: int,
        s: int = 4,
        seed: int = 42,
        incremental: bool = False,
    ) -> None:
        if not (isinstance(num_dim_ambient, int) and num_dim_ambient > 0):
            raise ValueError(
                f"num_dim_ambient must be a positive integer, got {num_dim_ambient}"
            )
        if not (isinstance(num_dim_embedding, int) and num_dim_embedding > 0):
            raise ValueError(
                f"num_dim_embedding must be a positive integer, got {num_dim_embedding}"
            )
        if not (isinstance(s, int) and s > 0):
            raise ValueError(f"s must be a positive integer, got {s}")
        if s > num_dim_embedding:
            raise ValueError(
                f"s must be <= num_dim_embedding, got s={s}, num_dim_embedding={num_dim_embedding}"
            )
        self.num_dim_ambient = num_dim_ambient
        self.num_dim_embedding = num_dim_embedding
        self.s = s
        self.seed = int(seed)
        self.incremental = bool(incremental)
        self._initialized = False
        self._x0 = None
        self._y0 = None

    @property
    def x0(self) -> torch.Tensor | None:
        return self._x0

    @property
    def y0(self) -> torch.Tensor | None:
        return self._y0

    @property
    def initialized(self) -> bool:
        return self._initialized

    def initialize(self, x_0: torch.Tensor) -> None:
        if self._initialized:
            raise RuntimeError("Already initialized, cannot initialize twice")
        if not torch.is_tensor(x_0):
            raise TypeError(f"x_0 must be a torch.Tensor, got {type(x_0)}")
        if x_0.ndim != 1:
            raise ValueError(f"x_0 must be 1D, got ndim={x_0.ndim}")
        if x_0.shape[0] != self.num_dim_ambient:
            raise ValueError(
                f"x_0 shape mismatch: expected ({self.num_dim_ambient},), got {x_0.shape}"
            )
        self._x0 = x_0
        if self.incremental:
            self._y0 = block_sparse_jl_transform_t(
                x_0, d=self.num_dim_embedding, s=self.s, seed=self.seed
            )
        self._initialized = True

    def transform(self, d_x: torch.Tensor) -> torch.Tensor:
        if not self._initialized:
            raise RuntimeError("Must call initialize() before transform()")
        if not (torch.is_tensor(d_x) and d_x.is_sparse):
            raise TypeError(
                f"d_x must be a sparse torch.Tensor, got {type(d_x)}, is_sparse={getattr(d_x, 'is_sparse', False)}"
            )
        if d_x.ndim != 1:
            raise ValueError(f"d_x must be 1D, got ndim={d_x.ndim}")
        if d_x.shape[0] != self.num_dim_ambient:
            raise ValueError(
                f"d_x shape mismatch: expected ({self.num_dim_ambient},), got {d_x.shape}"
            )
        if d_x.device != self._x0.device:
            raise ValueError(
                f"d_x device mismatch: expected {self._x0.device}, got {d_x.device}"
            )
        if d_x.dtype != self._x0.dtype:
            raise ValueError(
                f"d_x dtype mismatch: expected {self._x0.dtype}, got {d_x.dtype}"
            )
        if not self.incremental:
            x = self._x0 + d_x.to_dense()
            y = block_sparse_jl_transform_t(
                x, d=self.num_dim_embedding, s=self.s, seed=self.seed
            )
            return y
        if d_x._nnz() == 0:
            return self._y0.clone()
        d_x_coalesced = d_x.coalesce()
        idx = d_x_coalesced._indices().squeeze(0)
        vals = d_x_coalesced._values()
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
