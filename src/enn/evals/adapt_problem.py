from dataclasses import dataclass

import numpy as np

from enn import SemanticInputTransform, SemanticOutputProjection


class _Encoder:
    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, texts):
        return np.array([self.vectors[text] for text in texts])


@dataclass(frozen=True)
class AdaptProblem:
    x: np.ndarray
    y: np.ndarray
    input_transform: object
    output_projection: object


def make_problem(kind, case, seed, count=428):
    rng = np.random.default_rng(seed)
    dimensions = 3 if kind == "yy" else 16
    x = rng.uniform(size=(count, dimensions))
    active = (
        np.array([1, -1, 1, -1] + [0] * (dimensions - 4)) / 2
        if dimensions > 3
        else np.array([1, 0, 0])
    )
    t = (x - 0.5) @ active
    signal = np.sin(3 * t) + 0.3 * t**2
    names = tuple(f"x{i}" for i in range(dimensions))
    if kind == "yy":
        other = np.cos(4 * (x[:, 1] - 0.5))
        y = np.column_stack([signal] * 3 + [other] * 3)
        if case == "opposing":
            y[:, [1, 4]] *= -1
        if case == "independent":
            y = np.column_stack([np.sin((i + 1) * x[:, i % 3] * 3) for i in range(6)])
        embeddings = [[1, 0]] * 3 + [[0, 1]] * 3
        projection = SemanticOutputProjection.from_embeddings(
            [f"y{i}" for i in range(6)], embeddings
        )
        return AdaptProblem(x, y + rng.normal(0, 0.1, y.shape), None, projection)
    nuisance = np.roll(active, 8)
    prior = active if case == "aligned" else nuisance
    vectors = {name: np.eye(dimensions + 1)[i + 1] for i, name in enumerate(names)}
    for i in np.flatnonzero(prior):
        vectors[names[i]] = np.eye(dimensions + 1)[0] * np.sign(prior[i])
    vectors["response"] = np.eye(dimensions + 1)[0]
    kwargs = {"relative_weight": 3, "encoder": _Encoder(vectors)}
    bounds = np.tile([0.0, 1.0], (dimensions, 1))
    transform = (
        SemanticInputTransform(names, bounds, variance_cutoff=0.1, **kwargs)
        if kind == "xx"
        else SemanticInputTransform.from_input_output(
            names, bounds, ["response"], **kwargs
        )
    )
    y = (signal + rng.normal(0, 0.1, count))[:, None]
    return AdaptProblem(x, y, transform, None)
