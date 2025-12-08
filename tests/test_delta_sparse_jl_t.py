import pytest
import torch

from sampling.delta_sparse_jl_t import DeltaSparseJL_T
from sampling.sparse_jl_t import block_sparse_jl_transform_t


def make_sparse_vector(indices, values, size, dtype=torch.float32, device="cpu"):
    if len(indices) == 0:
        idx_t = torch.zeros((1, 0), dtype=torch.long, device=device)
        val_t = torch.tensor([], dtype=dtype, device=device)
    else:
        idx_t = torch.tensor([indices], dtype=torch.long, device=device)
        val_t = torch.tensor(values, dtype=dtype, device=device)
    return torch.sparse_coo_tensor(idx_t, val_t, size=(size,), device=device).coalesce()


def test_requires_initialize_before_transform():
    D = 16
    d = 8
    dut = DeltaSparseJL_T(num_dim_ambient=D, num_dim_embedding=d, s=4, seed=0)
    dx = make_sparse_vector([], [], D)
    with pytest.raises(AssertionError):
        _ = dut.transform(dx)


def test_initialize_only_once():
    D = 10
    d = 6
    dtype = torch.float32
    x0 = torch.zeros(D, dtype=dtype)
    dut = DeltaSparseJL_T(num_dim_ambient=D, num_dim_embedding=d, s=3, seed=1)
    dut.initialize(x0)
    with pytest.raises(AssertionError):
        dut.initialize(x0)


def test_initialize_shape_checks():
    D = 12
    d = 5
    dut = DeltaSparseJL_T(num_dim_ambient=D, num_dim_embedding=d, s=2, seed=2)
    with pytest.raises(AssertionError):
        dut.initialize(torch.zeros((D, 1)))
    with pytest.raises(AssertionError):
        dut.initialize(torch.zeros(D + 1))


def test_transform_requires_sparse_and_shape():
    D = 20
    d = 7
    dut = DeltaSparseJL_T(num_dim_ambient=D, num_dim_embedding=d, s=3, seed=3)
    dut.initialize(torch.zeros(D))
    with pytest.raises(AssertionError):
        _ = dut.transform(torch.zeros(D))
    with pytest.raises(AssertionError):
        _ = dut.transform(make_sparse_vector([0], [1.0], D + 1))


def test_parity_with_reference_transform():
    D = 32
    d = 16
    s = 4
    seed = 7
    gen = torch.Generator().manual_seed(2025)
    x0 = torch.randn(D, generator=gen)
    k = 5
    idx = torch.randperm(D, generator=gen)[:k]
    vals = torch.randn(k, generator=gen)
    dx = make_sparse_vector(
        idx.tolist(), vals.tolist(), D, dtype=x0.dtype, device=x0.device
    )
    dut = DeltaSparseJL_T(num_dim_ambient=D, num_dim_embedding=d, s=s, seed=seed)
    dut.initialize(x0)
    y = dut.transform(dx)
    y_ref = block_sparse_jl_transform_t(x0 + dx.to_dense(), d=d, s=s, seed=seed)
    assert torch.is_tensor(y)
    assert y.shape == (d,)
    torch.testing.assert_close(y, y_ref, atol=0, rtol=0)


def test_zero_maps_to_zero():
    D = 40
    d = 12
    dut = DeltaSparseJL_T(num_dim_ambient=D, num_dim_embedding=d, s=3, seed=0)
    x0 = torch.zeros(D)
    dx = make_sparse_vector([], [], D, dtype=x0.dtype, device=x0.device)
    dut.initialize(x0)
    y = dut.transform(dx)
    assert int(torch.count_nonzero(y)) == 0
    torch.testing.assert_close(y, torch.zeros_like(y), atol=0, rtol=0)


def test_seed_determinism():
    D = 24
    d = 8
    s = 4
    seed = 123
    g = torch.Generator().manual_seed(4242)
    x0 = torch.randn(D, generator=g)
    idx = torch.randperm(D, generator=g)[:6]
    vals = torch.randn(6, generator=g)
    dx = make_sparse_vector(
        idx.tolist(), vals.tolist(), D, dtype=x0.dtype, device=x0.device
    )
    a = DeltaSparseJL_T(num_dim_ambient=D, num_dim_embedding=d, s=s, seed=seed)
    b = DeltaSparseJL_T(num_dim_ambient=D, num_dim_embedding=d, s=s, seed=seed)
    a.initialize(x0)
    b.initialize(x0.clone())
    ya = a.transform(dx)
    yb = b.transform(dx)
    assert torch.equal(ya, yb)
    c = DeltaSparseJL_T(num_dim_ambient=D, num_dim_embedding=d, s=s, seed=seed + 1)
    c.initialize(x0)
    yc = c.transform(dx)
    assert not torch.equal(ya, yc)


def test_s_exceeds_embedding_raises():
    D = 10
    d = 4
    s = 5
    with pytest.raises(ValueError):
        _ = DeltaSparseJL_T(num_dim_ambient=D, num_dim_embedding=d, s=s, seed=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_device_dtype_gpu_paths():
    D = 48
    d = 12
    s = 3
    device = "cuda"
    dtype = torch.float32
    g = torch.Generator(device="cpu").manual_seed(2026)
    x0 = torch.randn(D, generator=g, dtype=dtype, device="cpu").to(device)
    idx = torch.randperm(D, generator=g)[:8].to(device)
    vals = torch.randn(8, generator=g, dtype=dtype, device="cpu").to(device)
    dx = make_sparse_vector(idx.tolist(), vals.tolist(), D, dtype=dtype, device=device)
    dut = DeltaSparseJL_T(num_dim_ambient=D, num_dim_embedding=d, s=s, seed=9)
    dut.initialize(x0)
    y = dut.transform(dx)
    assert y.device.type == "cuda"
    assert y.dtype == dtype


def test_delta_sparse_jl_t_timing_prints():
    import time

    gen = torch.Generator().manual_seed(3033)
    D = 5000
    d = 256
    s = 4
    k = 50
    x0 = torch.randn(D, generator=gen)
    idx = torch.randperm(D, generator=gen)[:k]
    vals = torch.randn(k, generator=gen) * 0.05
    dx = make_sparse_vector(
        idx.tolist(), vals.tolist(), D, dtype=x0.dtype, device=x0.device
    )
    dut = DeltaSparseJL_T(num_dim_ambient=D, num_dim_embedding=d, s=s, seed=0)
    dut.initialize(x0)
    times_ms = []
    y = None
    for _ in range(3):
        t0 = time.perf_counter()
        y = dut.transform(dx)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    mean_ms = sum(times_ms) / len(times_ms)
    print(
        f"D={D}, d={d}, s={s}, k={k}, time_ms_mean={mean_ms:.2f}, y_norm={float(torch.linalg.norm(y)):.4f}"
    )


def test_delta_sparse_jl_t_timing_sparse_dx_prints():
    import time

    gen = torch.Generator().manual_seed(3034)
    D = 20000
    d = 512
    s = 4
    k = 100
    x0 = torch.zeros(D)
    idx = torch.randperm(D, generator=gen)[:k]
    vals = torch.randn(k, generator=gen)
    dx = make_sparse_vector(
        idx.tolist(), vals.tolist(), D, dtype=x0.dtype, device=x0.device
    )
    dut = DeltaSparseJL_T(num_dim_ambient=D, num_dim_embedding=d, s=s, seed=1)
    dut.initialize(x0)
    times_ms = []
    y = None
    for _ in range(3):
        t0 = time.perf_counter()
        y = dut.transform(dx)
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)
    mean_ms = sum(times_ms) / len(times_ms)
    print(
        f"D={D}, d={d}, s={s}, k={k}, time_ms_mean={mean_ms:.2f}, y_norm={float(torch.linalg.norm(y)):.4f}"
    )
