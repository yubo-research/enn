from __future__ import annotations

import functools
from dataclasses import dataclass

import numpy as np

CHUNK_ROWS = 1000
NOISE_STD = 0.1
NUM_RELEVANT = 5
SUBSPACE_NOISE = 1e-3
TRAJ_DECAY = 10.0
TRAJ_JITTER = 0.1
NUM_CLUSTERS = 4
CLUSTER_STD = 0.01
MIX_WEIGHTS = (0.3, 0.4, 0.3)
KINDS = ("uniform", "subspace", "trajectory", "mixed")
QUERY_MODES = ("data", "incumbent")


@dataclass(frozen=True)
class Workload:
    kind: str = "uniform"
    num_dim: int = 10
    intrinsic: int = 3
    seed: int = 0

    def label(self) -> str:
        if self.kind == "subspace":
            return f"subspace{self.intrinsic}_d{self.num_dim}"
        return f"{self.kind}_d{self.num_dim}"


@dataclass(frozen=True)
class Geometry:
    basis: np.ndarray
    x0: np.ndarray
    x_star: np.ndarray
    centers: np.ndarray


@functools.lru_cache(maxsize=16)
def geometry(workload: Workload) -> Geometry:
    if workload.kind not in KINDS:
        raise ValueError(f"unknown workload kind: {workload.kind!r}")
    rng = np.random.default_rng([workload.seed, 7919])
    d = workload.num_dim
    m = min(workload.intrinsic, d)
    basis = np.linalg.qr(rng.standard_normal((d, m)))[0]
    return Geometry(
        basis=basis,
        x0=rng.random(d),
        x_star=rng.random(d),
        centers=rng.random((NUM_CLUSTERS, d)),
    )


def objective(workload: Workload, x: np.ndarray, rng) -> np.ndarray:
    rel = min(NUM_RELEVANT, x.shape[1])
    weights = 1.0 / np.arange(1, rel + 1)
    f = np.sin(2 * np.pi * x[:, :rel]) @ weights
    if workload.kind in ("trajectory", "mixed"):
        dist = np.linalg.norm(x - geometry(workload).x_star, axis=1)
        f = f + np.log10(dist + 1e-12)
    return (f + NOISE_STD * rng.standard_normal(len(x))).reshape(-1, 1)


def trajectory_radius(positions: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.asarray(positions, dtype=float) / TRAJ_DECAY)


def _trajectory(geo: Geometry, positions: np.ndarray, rng) -> np.ndarray:
    d = len(geo.x0)
    r = trajectory_radius(positions)[:, None]
    jitter = TRAJ_JITTER * rng.standard_normal((len(positions), d)) / np.sqrt(d)
    return geo.x_star + r * ((geo.x0 - geo.x_star) + jitter)


def _subspace(geo: Geometry, count: int, rng) -> np.ndarray:
    d, m = geo.basis.shape
    coords = rng.random((count, m)) - 0.5
    noise = SUBSPACE_NOISE * rng.standard_normal((count, d))
    return 0.5 + coords @ geo.basis.T + noise


def _mixed(geo: Geometry, positions: np.ndarray, rng) -> np.ndarray:
    count, d = len(positions), len(geo.x0)
    comp = rng.choice(len(MIX_WEIGHTS), size=count, p=MIX_WEIGHTS)
    x = rng.random((count, d))
    traj = comp == 1
    x[traj] = _trajectory(geo, positions[traj], rng)
    clus = comp == 2
    which = rng.integers(NUM_CLUSTERS, size=int(clus.sum()))
    x[clus] = geo.centers[which] + CLUSTER_STD * rng.standard_normal((len(which), d))
    return x


def sample_at(workload: Workload, positions: np.ndarray, rng) -> np.ndarray:
    geo = geometry(workload)
    positions = np.asarray(positions)
    if workload.kind == "uniform":
        return rng.random((len(positions), workload.num_dim))
    if workload.kind == "subspace":
        return _subspace(geo, len(positions), rng)
    if workload.kind == "trajectory":
        return _trajectory(geo, positions, rng)
    return _mixed(geo, positions, rng)


@functools.lru_cache(maxsize=4)
def chunk(workload: Workload, index: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng([workload.seed, 104729, index])
    lo = index * CHUNK_ROWS
    x = sample_at(workload, np.arange(lo, lo + CHUNK_ROWS), rng)
    return x, objective(workload, x, rng)


def rows(workload: Workload, lo: int, hi: int):
    pos = lo
    while pos < hi:
        c, off = divmod(pos, CHUNK_ROWS)
        x, y = chunk(workload, c)
        take = min(CHUNK_ROWS - off, hi - pos)
        yield pos, x[off : off + take], y[off : off + take]
        pos += take


def queries(workload: Workload, num_seen: int, num: int, mode: str, seed: int):
    if mode not in QUERY_MODES:
        raise ValueError(f"unknown query mode: {mode!r}")
    rng = np.random.default_rng([workload.seed, 15485863, num_seen, len(mode)])
    if mode == "data":
        positions = rng.integers(num_seen, size=num)
    else:
        positions = np.full(num, num_seen - 1)
    x = sample_at(workload, positions, rng)
    return x, objective(workload, x, rng)
