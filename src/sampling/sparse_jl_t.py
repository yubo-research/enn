import math

import torch

_SPLITMIX64_INCREMENT = 0x9E3779B97F4A7C15
_SPLITMIX64_MUL1 = 0xBF58476D1CE4E5B9
_SPLITMIX64_MUL2 = 0x94D049BB133111EB
_MASK_64 = (1 << 64) - 1


def _block_sparse_hash_scatter_from_nz_t(
    nz_indices: torch.Tensor,
    nz_values: torch.Tensor,
    d: int,
    s: int,
    seed: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    inv_sqrt_s = 1.0 / math.sqrt(float(s))
    y = torch.zeros(d, dtype=dtype, device=device)
    d_local = int(d)
    s_local = int(s)
    rows_accum: list[int] = []
    vals_accum: list[float] = []
    for j_t, v_t in zip(nz_indices, nz_values):
        j = int(j_t)
        v = float(v_t)
        if v == 0.0:
            continue
        acc = (
            _SPLITMIX64_INCREMENT ^ (int(seed) & _MASK_64) ^ ((int(j) & _MASK_64) << 1)
        )
        chosen_rows: list[int] = []
        while len(chosen_rows) < s_local:
            acc = (acc + _SPLITMIX64_INCREMENT) & _MASK_64
            z = acc
            z ^= z >> 30
            z = (z * _SPLITMIX64_MUL1) & _MASK_64
            z ^= z >> 27
            z = (z * _SPLITMIX64_MUL2) & _MASK_64
            z ^= z >> 31
            row = int(z % d_local)
            duplicate = False
            for rr in chosen_rows:
                if rr == row:
                    duplicate = True
                    break
            if duplicate:
                continue
            chosen_rows.append(row)
            acc = (acc + _SPLITMIX64_INCREMENT) & _MASK_64
            z2 = acc
            z2 ^= z2 >> 30
            z2 = (z2 * _SPLITMIX64_MUL1) & _MASK_64
            z2 ^= z2 >> 27
            z2 = (z2 * _SPLITMIX64_MUL2) & _MASK_64
            z2 ^= z2 >> 31
            sign = 1.0 if (z2 & 1) == 1 else -1.0
            rows_accum.append(row)
            vals_accum.append(sign * v * inv_sqrt_s)
    if rows_accum:
        rows_t = torch.tensor(rows_accum, dtype=torch.long, device=device)
        vals_t = torch.tensor(vals_accum, dtype=dtype, device=device)
        y.scatter_add_(0, rows_t, vals_t)
    return y


def block_sparse_jl_transform_t(
    x: torch.Tensor, d: int, s: int = 4, seed: int = 42
) -> torch.Tensor:
    if x.ndim != 1:
        raise ValueError(f"x must be 1D, got ndim={x.ndim}")
    if d <= 0:
        raise ValueError(f"d must be positive, got {d}")
    if s <= 0:
        raise ValueError(f"s must be positive, got {s}")
    if s > d:
        raise ValueError(f"s must be <= d, got s={s}, d={d}")
    if not torch.any(x):
        return torch.zeros(d, dtype=x.dtype, device=x.device)
    nz_indices = torch.nonzero(x, as_tuple=False).squeeze(-1)
    nz_values = x[nz_indices]
    return _block_sparse_hash_scatter_from_nz_t(
        nz_indices=nz_indices,
        nz_values=nz_values,
        d=d,
        s=s,
        seed=seed,
        dtype=x.dtype,
        device=x.device,
    )
