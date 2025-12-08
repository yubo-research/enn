import numpy as np
import torch
from scipy.stats import spearmanr

from sampling.sparse_jl_t import block_sparse_jl_transform_t


def test_sparse_jl_t_preserves_neighbors_and_correlations():
    N = 100
    D = 1000
    R = D // 10
    C = 30
    eps = 0.2
    s = 4
    d = int(8 * np.log(C + 1) / (eps**2) + 0.5)
    assert d < D
    assert s < d
    theta_base = torch.linspace(0.1, 1.0, D)
    gen0 = torch.Generator().manual_seed(2024)
    theta0 = torch.stack(
        [theta_base + torch.randn(D, generator=gen0) * 0.01 for _ in range(N)], dim=0
    )
    y0 = torch.stack(
        [block_sparse_jl_transform_t(theta0[i], d, s=s) for i in range(N)], dim=0
    )
    gen = torch.Generator().manual_seed(123)
    dists_orig = torch.zeros((C, N), dtype=torch.float64)
    dists_embed = torch.zeros((C, N), dtype=torch.float64)
    pearson_all = np.zeros(C)
    spearman_all = np.zeros(C)
    for c in range(C):
        indices = torch.randperm(D, generator=gen)[:R]
        delta = torch.zeros(D)
        delta[indices] = torch.randn(R, generator=gen) * 0.05
        theta_c = theta0[0] + delta
        y_c = block_sparse_jl_transform_t(theta_c, d, s=s)
        dists_orig[c] = torch.linalg.norm(theta0 - theta_c, dim=1).double()
        dists_embed[c] = torch.linalg.norm(y0 - y_c, dim=1).double()
        pearson_all[c] = np.corrcoef(dists_orig[c].numpy(), dists_embed[c].numpy())[
            0, 1
        ]
        spearman_all[c] = spearmanr(dists_orig[c].numpy(), dists_embed[c].numpy())[0]
    nn_orig = torch.argmin(dists_orig, dim=1).numpy()
    nn_embed = torch.argmin(dists_embed, dim=1).numpy()
    nn_match_frac = float(np.mean(nn_orig == nn_embed))
    assert nn_match_frac > 0.9
    assert float(np.mean(pearson_all)) > 0.6
    assert float(np.median(pearson_all)) > 0.6
    assert float(np.mean(spearman_all)) > 0.6
    assert float(np.median(spearman_all)) > 0.6


def test_sparse_jl_t_without_replacement_per_column():
    D = 32
    d = 16
    s = 8
    seed = 0
    j = 0
    x = torch.zeros(D)
    x[j] = 1.0
    y = block_sparse_jl_transform_t(x, d=d, s=s, seed=seed)
    idx = torch.nonzero(torch.abs(y) > 1e-12, as_tuple=False).squeeze(-1)
    assert int(idx.numel()) == s
    assert torch.allclose(
        torch.abs(y[idx]), torch.full((s,), 1.0 / np.sqrt(s), dtype=y.dtype)
    )


def test_sparse_jl_t_zero_maps_to_zero():
    D = 50
    d = 12
    s = 4
    x = torch.zeros(D)
    y = block_sparse_jl_transform_t(x, d=d, s=s, seed=7)
    assert int(torch.count_nonzero(y)) == 0
    assert torch.allclose(y, torch.zeros_like(y))


def test_sparse_jl_t_linearity_and_determinism():
    D = 64
    d = 16
    s = 8
    gen = torch.Generator().manual_seed(2025)
    x = torch.randn(D, generator=gen)
    z = torch.randn(D, generator=gen)
    seed = 123
    y_sum = block_sparse_jl_transform_t(x + z, d=d, s=s, seed=seed)
    y_x = block_sparse_jl_transform_t(x, d=d, s=s, seed=seed)
    y_z = block_sparse_jl_transform_t(z, d=d, s=s, seed=seed)
    assert torch.allclose(y_sum, y_x + y_z)
    y1 = block_sparse_jl_transform_t(x, d=d, s=s, seed=seed)
    y2 = block_sparse_jl_transform_t(x, d=d, s=s, seed=seed)
    assert torch.equal(y1, y2)
    y3 = block_sparse_jl_transform_t(x, d=d, s=s, seed=seed + 1)
    assert not torch.equal(y1, y3)


def test_sparse_jl_t_raises_when_s_exceeds_d():
    D = 10
    d = 4
    s = 5
    x = torch.ones(D)
    import pytest

    with pytest.raises(ValueError):
        block_sparse_jl_transform_t(x, d=d, s=s, seed=0)


def test_sparse_jl_t_timing_prints():
    import time

    gen = torch.Generator().manual_seed(2026)
    configs = [(5000, 256, 4), (20000, 512, 4)]
    for D, d, s in configs:
        x = torch.randn(D, generator=gen)
        t0 = time.perf_counter()
        y = block_sparse_jl_transform_t(x, d=d, s=s, seed=0)
        t1 = time.perf_counter()
        elapsed_ms = (t1 - t0) * 1000.0
        print(
            f"D={D}, d={d}, s={s}, time_ms={elapsed_ms:.2f}, y_norm={float(torch.linalg.norm(y)):.4f}"
        )


def test_sparse_jl_t_timing_sparse_x_prints():
    import time

    gen = torch.Generator().manual_seed(2027)
    D = 20000
    d = 512
    s = 4
    k = 100
    x = torch.zeros(D)
    nz_idx = torch.randperm(D, generator=gen)[:k]
    x[nz_idx] = torch.randn(k, generator=gen)
    t0 = time.perf_counter()
    y = block_sparse_jl_transform_t(x, d=d, s=s, seed=1)
    t1 = time.perf_counter()
    elapsed_ms = (t1 - t0) * 1000.0
    print(
        f"D={D}, d={d}, s={s}, k={k}, time_ms={elapsed_ms:.2f}, y_norm={float(torch.linalg.norm(y)):.4f}"
    )
