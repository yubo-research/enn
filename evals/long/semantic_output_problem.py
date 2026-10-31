import numpy as np

from enn import SemanticOutputProjection

DESCRIPTIONS = (
    "Temperature of the heating system in degrees Celsius",
    "Pressure of the hydraulic system in pascals",
    "Electrical voltage of the power supply in volts",
    "Speed of the rotating motor in revolutions per minute",
)


def output_projection(outputs, directory):
    return SemanticOutputProjection.from_metadata(
        [f"metric_{j}" for j in range(outputs)],
        [DESCRIPTIONS[j % len(DESCRIPTIONS)] for j in range(outputs)],
        variance_cutoff=0.9,
        cache_dir=directory,
    )


def synthetic_problem(seed, dimensions, projection, scenario, count):
    rng = np.random.default_rng(seed)
    outputs = len(projection.names)
    x = rng.uniform(-1, 1, (count, dimensions))
    weights = rng.normal(size=(dimensions, outputs)) / np.sqrt(dimensions / 3)
    functions = np.sin(x @ weights) + 0.3 * np.cos(2 * x @ weights)
    loadings = projection.loadings.copy()
    if scenario == "C_opposing":
        signs = np.arange(outputs) // len(DESCRIPTIONS)
        loadings *= np.where(signs % 2, -1, 1)[:, None]
    if scenario == "D_misleading":
        loadings = _misleading_loadings(loadings, rng)
    if scenario == "B_independent":
        y = functions
    else:
        y = functions[:, : loadings.shape[1]] @ loadings.T
        y /= np.maximum(np.linalg.norm(loadings, axis=1), 1e-12)
    return x, y + rng.normal(scale=0.1, size=y.shape)


def method_embeddings(projection, seed):
    rng = np.random.default_rng(seed + 10000)
    return {
        "independent": projection.embeddings,
        "semantic": projection.embeddings,
        "shuffled": projection.embeddings[rng.permutation(len(projection.names))],
        "random": rng.normal(size=projection.embeddings.shape),
    }


def _misleading_loadings(loadings, rng):
    shuffled = loadings[rng.permutation(len(loadings))]
    if np.allclose(shuffled @ shuffled.T, loadings @ loadings.T):
        order = np.arange(len(loadings))
        order[:2] = [1, 0]
        shuffled = loadings[order]
    return shuffled
