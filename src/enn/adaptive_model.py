"""Training-only geometry and output operators for online semantic ENNs."""

from dataclasses import dataclass

import numpy as np

from enn import EpistemicNearestNeighbors, enn_rust
from enn.enn.enn_params import ENNParams
from enn.semantic_multi_output import OutputPosterior


@dataclass(frozen=True)
class AdaptChoice:
    geometry: str
    weight: float
    output: str
    neighbors: int


def _quadratic_direction(z, target):
    design = np.column_stack((z, z**2 - 1))
    coefficients = np.linalg.solve(
        design.T @ design + len(z) * 0.03 * np.eye(design.shape[1]),
        design.T @ target,
    )
    linear, quadratic = np.split(coefficients, 2)
    gradient = linear[None, :, :] + 2 * z[:, :, None] * quadratic[None, :, :]
    flat = gradient.transpose(1, 0, 2).reshape(z.shape[1], -1)
    moment = flat @ flat.T / len(z)
    values, vectors = np.linalg.eigh(moment)
    direction = vectors * np.sqrt(np.maximum(values, 0))
    return direction / max(np.linalg.norm(direction), 1e-12) * np.sqrt(z.shape[1])


def _fixed_rank(direction, rank, dimensions):
    """Keep the metadata-selected factor count, including the zero-factor case."""
    u, singular, _ = np.linalg.svd(direction, full_matrices=False)
    count = min(rank, len(singular))
    retained = np.zeros((dimensions, rank))
    retained[:, :count] = u[:, :count] * singular[:count]
    return retained / max(np.linalg.norm(retained), 1e-12) * np.sqrt(dimensions)


class AdaptiveBasis:
    """Learn ridge directions and signed output factors from fit rows.

    ``geometry`` and ``output`` restrict final refits to the selected maps.
    ``linear_mean=False`` also skips an unused ridge trend. Selection bases leave
    the map restrictions unset to prepare the complete candidate family.
    Exactly constant fit columns use unit scale so later changes are not
    magnified by the numerical standard-deviation floor.
    """

    def __init__(
        self,
        x,
        y,
        input_transform,
        output_projection,
        *,
        geometry=None,
        output=None,
        linear_mean=True,
    ):
        self.input_transform = input_transform
        self.x_mean = x.mean(axis=0)
        centered_x = x - self.x_mean
        self.x_scale = np.where(
            np.ptp(x, axis=0) == 0,
            1.0,
            np.maximum(np.sqrt(np.mean(centered_x**2, axis=0)), 1e-12),
        )
        self.y_mean = y.mean(axis=0)
        centered_y = y - self.y_mean
        self.constant = np.ptp(y, axis=0) == 0
        self.y_scale = np.where(
            self.constant, 1, np.maximum(np.sqrt(np.mean(centered_y**2, axis=0)), 1e-12)
        )
        z = centered_x / self.x_scale
        target = centered_y / self.y_scale
        self.ridge, self.directions = _input_maps(z, target, geometry, linear_mean)
        if input_transform is not None:
            rank = input_transform.projection.shape[1]
            self.directions = {
                name: _fixed_rank(direction, rank, x.shape[1])
                for name, direction in self.directions.items()
            }
        self.direction = self.directions.get("linear")
        self._feature_roots = {}
        self.operators = _output_operators(target, output_projection, output)

    def features(self, x, choice):
        if choice.geometry in ("raw", "fixed"):
            if self.input_transform is None:
                return x
            lower, upper = self.input_transform.bounds.T
            z = (x - lower) / (upper - lower)
            if choice.geometry == "raw":
                return z
            direction = self.input_transform.weighted_projection
            return z @ self._root("fixed", 1, direction)
        z = (x - self.x_mean) / self.x_scale
        return z @ self._root(
            choice.geometry, choice.weight, self.directions[choice.geometry]
        )

    def _root(self, geometry, weight, direction):
        key = geometry, weight
        if key not in self._feature_roots:
            gram = np.eye(len(self.x_mean)) + weight * direction @ direction.T
            values, vectors = np.linalg.eigh(gram)
            self._feature_roots[key] = (
                vectors * np.sqrt(np.maximum(values, 0))
            ) @ vectors.T
        return self._feature_roots[key]

    def trend(self, x):
        if self.ridge is None:
            raise RuntimeError("this basis was fitted without a linear trend")
        return (x - self.x_mean) / self.x_scale @ self.ridge


class AdaptivePredictor:
    """A fixed epoch: a geometry, an ENN, and a correlated-error SD envelope."""

    def __init__(self, basis, x, y, choice, linear_mean):
        self.basis, self.choice, self.linear_mean = basis, choice, linear_mean
        target = (y - basis.y_mean) / basis.y_scale
        if linear_mean:
            target = target - basis.trend(x)
        self.model = EpistemicNearestNeighbors(basis.features(x, choice), target)
        self.params = ENNParams(min(choice.neighbors, len(x)), 1, 0)

    def posterior(self, x):
        posterior = self.model.posterior(
            self.basis.features(x, self.choice), params=self.params
        )
        return self._restore(x, posterior)

    def selection_predictions(self, x, neighbors):
        """Share the index and features across all candidate neighbor counts."""
        params = [ENNParams(min(k, len(self.model)), 1, 0) for k in neighbors]
        posterior = self.model.batch_posterior(
            self.basis.features(x, self.choice), params
        )
        return self._restore(x, posterior)

    def _restore(self, x, posterior):
        operator = self.basis.operators[self.choice.output]
        mean = posterior.mu
        if self.linear_mean:
            mean = mean + self.basis.trend(x)
        mu = (mean @ operator) * self.basis.y_scale + self.basis.y_mean
        se = (posterior.se @ np.abs(operator)) * self.basis.y_scale
        mu[..., self.basis.constant] = self.basis.y_mean[self.basis.constant]
        se[..., self.basis.constant] = 0
        return OutputPosterior(mu, se)


def select_predictor(
    x, y, transform, projection, linear_mean, adaptation="all", geometry_weight=None
):
    """Fit on first 60%; select on next 20%; reserve final 20% for calibration.

    Return a new basis refitted on fit+selection rows. Calibration rows never
    influence geometry, scaling, candidate choice, or ENN training in this epoch.
    """
    geometry_weight = validate_geometry_weight(geometry_weight)
    fit_end, calibration_start = int(len(x) * 0.6), int(len(x) * 0.8)
    basis = AdaptiveBasis(x[:fit_end], y[:fit_end], transform, projection)
    choice = _select_choice(
        basis,
        x[:calibration_start],
        y[:calibration_start],
        fit_end,
        linear_mean,
        adaptation,
        geometry_weight,
    )
    basis = AdaptiveBasis(
        x[:calibration_start],
        y[:calibration_start],
        transform,
        projection,
        geometry=choice.geometry,
        output=choice.output,
        linear_mean=linear_mean,
    )
    model = AdaptivePredictor(
        basis, x[:calibration_start], y[:calibration_start], choice, linear_mean
    )
    posterior = model.posterior(x[calibration_start:])
    floor = np.maximum(basis.y_scale * 0.03, 1e-12)
    variance = np.maximum(posterior.se**2, floor**2)
    calibration = np.maximum(
        np.mean((y[calibration_start:] - posterior.mu) ** 2 / variance, axis=0), 1e-12
    )
    return model, floor, calibration


def _select_choice(basis, x, y, fit_end, linear_mean, adaptation, geometry_weight):
    scores = []
    weights = (3, 30) if geometry_weight is None else (geometry_weight,)
    geometries = [("fixed", 0)] + [
        (geometry, weight) for geometry in ("linear", "quadratic") for weight in weights
    ]
    if basis.input_transform is not None:
        geometries.insert(0, ("raw", 0))
    if adaptation in ("outputs", "none"):
        geometries = [("fixed", 0)]
    outputs = _output_candidates(basis, adaptation)
    neighbors = (3, 10, 30)
    target = (y[fit_end:] - basis.y_mean) / basis.y_scale
    for geometry, weight in geometries:
        choice = AdaptChoice(geometry, weight, "independent", neighbors[0])
        mean = _selection_means(
            basis,
            (x[:fit_end], y[:fit_end], x[fit_end:]),
            choice,
            linear_mean,
            neighbors,
        )
        standardized = (mean - basis.y_mean) / basis.y_scale
        losses = _selection_scores(standardized, target, outputs)
        i, j = np.unravel_index(np.argmin(losses), losses.shape)
        candidate = AdaptChoice(geometry, weight, tuple(outputs)[j], neighbors[i])
        scores.append((float(losses[i, j]), candidate))
    return min(scores, key=lambda item: item[0])[1]


def _output_candidates(basis, adaptation):
    if adaptation == "inputs":
        return {"independent": basis.operators["independent"]}
    if adaptation == "none":
        name = "semantic" if "semantic" in basis.operators else "independent"
        return {name: basis.operators[name]}
    return basis.operators


def _selection_means(basis, rows, choice, linear_mean, neighbors):
    """Use certified fit-row means, falling back to the full predictor if needed."""
    x, y, query = rows
    target = (y - basis.y_mean) / basis.y_scale
    if linear_mean:
        target = target - basis.trend(x)
    means = enn_rust.model.small_flat_selection_means(
        basis.features(x, choice),
        target,
        basis.features(query, choice),
        list(neighbors),
    )
    if means is None:
        model = AdaptivePredictor(basis, x, y, choice, linear_mean)
        return model.selection_predictions(query, neighbors).mu
    if linear_mean:
        means = means + basis.trend(query)
    mu = (means @ basis.operators["independent"]) * basis.y_scale + basis.y_mean
    mu[..., basis.constant] = basis.y_mean[basis.constant]
    return mu


def _selection_scores(means, target, outputs):
    """Batch counts while preserving each candidate's reduction and tie order."""
    operators = np.stack(tuple(outputs.values()))
    errors = means[:, None] @ operators[None] - target
    return np.mean(errors**2, axis=(2, 3))


def validate_geometry_weight(weight):
    """Allow a fixed positive finite geometry weight without reducing factor ranks."""
    if weight is None:
        return None
    if (
        isinstance(weight, (bool, np.bool_))
        or not isinstance(weight, (int, float, np.integer, np.floating))
        or weight <= 0
        or weight > float(np.finfo(float).max)
        or not np.isfinite(float(weight))
    ):
        raise ValueError("geometry_weight must be a positive finite number or None")
    return float(weight)


def _input_maps(z, target, geometry, linear_mean):
    ridge, directions = None, {}
    if geometry in (None, "linear") or linear_mean:
        ridge = np.linalg.solve(
            z.T @ z + len(z) * 0.03 * np.eye(z.shape[1]), z.T @ target
        )
    if geometry in (None, "linear"):
        directions["linear"] = (
            ridge / max(np.linalg.norm(ridge), 1e-12) * np.sqrt(z.shape[1])
        )
    if geometry in (None, "quadratic"):
        directions["quadratic"] = _quadratic_direction(z, target)
    return ridge, directions


def _output_operators(target, projection, output):
    identity = np.eye(target.shape[1])
    operators = {"independent": identity}
    if projection is None:
        return operators
    loading = projection.loadings
    if output in (None, "semantic"):
        operators["semantic"] = (
            identity
            if not loading.shape[1]
            else 0.7 * identity + 0.3 * loading @ loading.T
        )
    if output in (None, "learned", "lowrank"):
        _, _, vt = np.linalg.svd(target, full_matrices=False)
        rank = loading.shape[1]
        learned = vt[:rank].T @ vt[:rank] if rank else identity
        if output in (None, "learned"):
            operators["learned"] = 0.7 * identity + 0.3 * learned
        if output in (None, "lowrank"):
            operators["lowrank"] = learned
    return operators
