"""Python API for Rust-owned bounded replay semantic adaptation."""

import numpy as np

from enn._rust import AdaptiveSemanticENN as _NativeAdaptive
from enn.adaptive_model import AdaptChoice
from enn.semantic_multi_output import OutputPosterior


class AdaptiveSemanticENN:
    """Fit geometry, output sharing, and uncertainty in bounded replay epochs.

    The initial batch needs at least 12 rows. Uniform priority sampling retains
    at most ``capacity`` rows. Each refit uses a fresh 60/20/20 fit, selection,
    and calibration split; calibration labels never influence the fitted mean.
    Semantic and learned output projectors mix with identity using a weight
    fitted on selection rows. ``choice.alpha`` reports that weight.
    Metadata remains immutable; learned input maps retain its factor width.
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
        self.input_transform = input_transform
        self.output_projection = output_projection
        self.capacity = int(capacity)
        self.linear_mean = linear_mean
        self.adaptation = adaptation
        self._native = _NativeAdaptive(
            None if input_transform is None else input_transform.bounds,
            None if input_transform is None else input_transform.weighted_projection,
            None if output_projection is None else output_projection.loadings,
            self.capacity,
            seed,
            linear_mean,
            adaptation,
        )

    @property
    def num_seen(self):
        return self._native.num_seen

    @property
    def num_refits(self):
        return self._native.num_refits

    @property
    def num_retained(self):
        return self._native.num_retained

    @property
    def choice(self):
        value = self._native.choice
        return None if value is None else AdaptChoice(*value)

    def add(self, x, y, *, geometry_weight=None):
        """Refit, tuning geometry weights 3 and 30 or using a supplied weight."""
        if isinstance(geometry_weight, (bool, np.bool_)) or (
            geometry_weight is not None
            and not isinstance(geometry_weight, (int, float, np.integer, np.floating))
        ):
            raise ValueError("geometry_weight must be a positive finite number or None")
        try:
            weight = None if geometry_weight is None else float(geometry_weight)
        except OverflowError as error:
            raise ValueError("geometry_weight must be finite") from error
        self._native.add(np.asarray(x, dtype=float), np.asarray(y, dtype=float), weight)
        return self

    def posterior(self, x):
        if self.choice is None:
            raise RuntimeError("add must be called before posterior")
        return OutputPosterior(*self._native.posterior(np.asarray(x, dtype=float)))
