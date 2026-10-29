"""Bounded replay adaptation of input geometry and output relationships."""

import numpy as np

from enn.adaptive_model import select_predictor, validate_geometry_weight
from enn.semantic_multi_output import OutputPosterior


def _rows(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (
        x.ndim != 2
        or y.ndim != 2
        or not x.shape[1]
        or not y.shape[1]
        or len(x) != len(y)
        or not len(x)
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise ValueError("observations must be finite, nonempty matching matrices")
    return x, y


class AdaptiveSemanticENN:
    """Optional online wrapper with bounded uniform replay and epoch refits.

    Call ``add`` with at least 12 rows initially, then any nonempty batch. Every
    call refits geometry, output sharing, ENNs, and marginal uncertainty using
    at most ``capacity`` retained rows. Random priority sampling is uniform over
    all observations seen; it does not favor recent data. Each epoch uses a fresh
    fit/selection/calibration split, with complete output rows kept together.

    This is bounded replay, not an exact update of a full-history index. It
    supports finite continuous numerical inputs, complete output matrices, and
    marginal predictions only. Metadata transforms/projections remain immutable.
    Input geometry adapts in a separate supervised linear map. Optional linear
    mean correction is disabled by default to isolate geometry adaptation.
    ``adaptation`` restricts the candidate family for controlled ablations:
    all, inputs only, outputs only, or none (fixed prior with neighbor tuning).
    Predictive variances are calibrated on withheld replay rows; no joint
    covariance or coverage guarantee under adaptive sampling is claimed.
    """

    def __init__(
        self,
        *,
        input_transform=None,
        output_projection=None,
        capacity=300,
        seed=0,
        linear_mean=False,
        adaptation="all",
    ):
        if (
            isinstance(capacity, bool)
            or not isinstance(capacity, (int, np.integer))
            or capacity < 12
        ):
            raise ValueError("capacity must be an integer at least 12")
        if adaptation not in ("all", "inputs", "outputs", "none"):
            raise ValueError("adaptation must be all, inputs, outputs, or none")
        self.adaptation = adaptation
        self.input_transform = input_transform
        self.output_projection = output_projection
        self.capacity = int(capacity)
        self.linear_mean = linear_mean
        self._rng = np.random.default_rng(seed)
        self._sample = None
        self._model = None
        self.num_seen = 0
        self.num_refits = 0

    @property
    def num_retained(self):
        return 0 if self._sample is None else len(self._sample[0])

    @property
    def choice(self):
        """Selected geometry, weight, output operator, and neighbor count, or None before first add."""
        return None if self._model is None else self._model.choice

    def add(self, x, y, *, geometry_weight=None):
        """Refit an epoch, optionally holding the learned geometry weight fixed.

        None tunes weights 3 and 30, alongside a zero-mixing raw candidate. A positive finite value evaluates that one
        weight for both learned geometries; metadata and factor ranks are retained.
        Pass the value on each update to keep it fixed throughout a stream.
        """
        geometry_weight = validate_geometry_weight(geometry_weight)
        options = (
            {} if geometry_weight is None else {"geometry_weight": geometry_weight}
        )
        x, y = _rows(x, y)
        self._validate_shape(x, y)
        sample = self._updated_sample(x, y)
        order = self._rng.permutation(len(sample[0]))
        model, floor, calibration = select_predictor(
            sample[0][order],
            sample[1][order],
            self.input_transform,
            self.output_projection,
            self.linear_mean,
            self.adaptation,
            **options,
        )
        self._sample = sample
        self._model, self._floor, self._calibration = model, floor, calibration
        self.num_seen += len(x)
        self.num_refits += 1
        return self

    def _validate_shape(self, x, y):
        if self._sample is None and len(x) < 12:
            raise ValueError("initial batch must have at least 12 rows")
        if self._sample is not None and (
            x.shape[1] != self._sample[0].shape[1]
            or y.shape[1] != self._sample[1].shape[1]
        ):
            raise ValueError("observation dimensions must match previous batches")
        if self.output_projection is not None and y.shape[1] != len(
            self.output_projection.names
        ):
            raise ValueError("output columns must match the semantic projection")
        if self.input_transform is not None and x.shape[1] != len(
            self.input_transform.names
        ):
            raise ValueError("input columns must match the semantic transform")

    def _updated_sample(self, x, y):
        keys = self._rng.random(len(x))
        if self._sample is not None:
            x = np.concatenate((self._sample[0], x))
            y = np.concatenate((self._sample[1], y))
            keys = np.concatenate((self._sample[2], keys))
        keep = np.argsort(keys)[: self.capacity]
        return x[keep].copy(), y[keep].copy(), keys[keep]

    def posterior(self, x):
        if self._model is None:
            raise RuntimeError("add must be called before posterior")
        x = np.asarray(x, dtype=float)
        if (
            x.ndim != 2
            or x.shape[1] != self._sample[0].shape[1]
            or not np.isfinite(x).all()
        ):
            raise ValueError("query must be finite with the fitted input dimension")
        if not len(x):
            shape = (0, self._sample[1].shape[1])
            return OutputPosterior(np.empty(shape), np.empty(shape))
        posterior = self._model.posterior(x)
        variance = np.maximum(posterior.se**2, self._floor**2) * self._calibration
        return OutputPosterior(posterior.mu, np.sqrt(variance))
