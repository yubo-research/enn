from pathlib import Path

import numpy as np

from enn import SemanticOutputProjection

GROUPS = ("temperature", "pressure", "voltage", "speed")
DESCRIPTIONS = (
    "Temperature measured by the inlet heating-system sensor, in degrees Celsius.",
    "Temperature measured by the central heating-system sensor, in degrees Celsius.",
    "Temperature measured by the outlet heating-system sensor, in degrees Celsius.",
    "Temperature measured by the backup heating-system sensor, in degrees Celsius.",
    "Pressure measured by the inlet hydraulic-system gauge, in pascals.",
    "Pressure measured by the central hydraulic-system gauge, in pascals.",
    "Pressure measured by the outlet hydraulic-system gauge, in pascals.",
    "Pressure measured by the backup hydraulic-system gauge, in pascals.",
    "Electrical voltage measured at the inlet power-supply terminal, in volts.",
    "Electrical voltage measured at the central power-supply terminal, in volts.",
    "Electrical voltage measured at the outlet power-supply terminal, in volts.",
    "Electrical voltage measured at the backup power-supply terminal, in volts.",
    "Rotational speed measured by the inlet motor tachometer, in revolutions per minute.",
    "Rotational speed measured by the central motor tachometer, in revolutions per minute.",
    "Rotational speed measured by the outlet motor tachometer, in revolutions per minute.",
    "Rotational speed measured by the backup motor tachometer, in revolutions per minute.",
)
NAMES = tuple(f"{group}_{sensor}" for group in GROUPS for sensor in range(4))
CASES = {
    "aligned_noisy": 1.0,
    "aligned_moderate": 0.3,
    "aligned_clean": 0.03,
    "opposing": 0.3,
    "unrelated": 0.3,
    "misleading": 1.0,
    "correlated_noise": 1.0,
}
ALPHAS = (0.0, 0.1, 0.3, 1.0)


def metadata(directory):
    directory = Path(directory)
    paraphrase = SemanticOutputProjection.from_metadata(
        NAMES, DESCRIPTIONS, cache_dir=directory / "cache"
    )
    exact_texts = tuple(DESCRIPTIONS[4 * (j // 4)] for j in range(16))
    exact = SemanticOutputProjection.from_metadata(
        NAMES, exact_texts, cache_dir=directory / "cache"
    )
    paraphrase.save(directory / "paraphrase_projection.npz")
    exact.save(directory / "repeated_projection.npz")
    return paraphrase, exact


def problem(seed, n, case):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, (n + 100 + 512, 4))
    latent = np.column_stack(
        (
            np.sin(2 * x[:, 0]),
            np.cos(2 * x[:, 1]),
            np.tanh(2 * x[:, 2]),
            x[:, 3] + 0.3 * np.sin(3 * x[:, 3]),
        )
    )
    clean = np.repeat(latent, 4, axis=1)
    if case == "opposing":
        clean *= np.tile([1, -1, 1, -1], 4)
    if case == "unrelated":
        weights = np.random.default_rng(1701).normal(size=(4, 16))
        weights /= np.linalg.norm(weights, axis=0)
        clean = np.sin(3 * (x @ weights))
    noise = rng.normal(size=clean.shape)
    if case == "correlated_noise":
        noise = np.repeat(noise[:, ::4], 4, axis=1)
    return x, clean + CASES[case] * noise, clean


def projections(paraphrase, exact, y, seed, case, variance_cutoff=0.9):
    rng = np.random.default_rng(seed + 10000)
    order = np.arange(16).reshape(4, 4).T.ravel()
    semantic = paraphrase.embeddings
    if case == "misleading":
        semantic = semantic[order]
    z = (y - y.mean(axis=0)) / y.std(axis=0)
    embeddings = {
        "semantic": semantic,
        "shuffled": semantic[rng.permutation(16)],
        "random": rng.normal(size=semantic.shape),
        "pca": z.T,
        "repeated": exact.embeddings[order]
        if case == "misleading"
        else exact.embeddings,
    }
    return {
        name: SemanticOutputProjection.from_embeddings(
            NAMES, values, variance_cutoff=variance_cutoff
        )
        for name, values in embeddings.items()
    }
