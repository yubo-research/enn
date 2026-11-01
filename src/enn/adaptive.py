import numpy as np

from enn._rust import AdaptiveSemanticENN as _NativeAdaptive
from enn.adaptive_model import AdaptChoice
from enn.semantic_multi_output import OutputPosterior


class AdaptiveSemanticENN:
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
        self.input_transform = input_transform
        self.output_projection = output_projection
        self.linear_mean = linear_mean
        self.adaptation = adaptation
        self._native = _NativeAdaptive(
            None if input_transform is None else input_transform.bounds,
            None if input_transform is None else input_transform.weighted_projection,
            None if output_projection is None else output_projection.loadings,
            capacity,
            seed,
            linear_mean,
            adaptation,
        )

    @property
    def capacity(self):
        return self._native.capacity

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
        self._native.add(
            np.asarray(x, dtype=float), np.asarray(y, dtype=float), geometry_weight
        )
        return self

    def posterior(self, x):
        return OutputPosterior(*self._native.posterior(np.asarray(x, dtype=float)))
