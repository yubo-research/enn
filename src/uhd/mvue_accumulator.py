import math


class MVUE:
    def __init__(self, decay: float) -> None:
        if not (0.0 < decay <= 1.0):
            raise ValueError("decay must be in (0, 1]")
        self._decay = decay
        self._sum_precision: float = 0.0
        self._sum_weighted_y: float = 0.0
        self._n: int = 0

    def update(self, y: float, y_var: float) -> None:
        if y_var <= 0:
            raise ValueError("y_var must be positive")
        precision = 1.0 / y_var
        self._sum_precision = self._decay * self._sum_precision + precision
        self._sum_weighted_y = self._decay * self._sum_weighted_y + y * precision
        self._n += 1

    @property
    def n(self) -> int:
        return self._n

    @property
    def mean(self) -> float:
        if self._sum_precision == 0:
            raise ValueError("No observations yet")
        return self._sum_weighted_y / self._sum_precision

    @property
    def var(self) -> float:
        if self._sum_precision == 0:
            raise ValueError("No observations yet")
        return 1.0 / self._sum_precision

    @property
    def se(self) -> float:
        return math.sqrt(self.var)

    def confidence_bounds(self, k: float) -> tuple[float, float]:
        lcb = self.mean - k * self.se
        ucb = self.mean + k * self.se
        return lcb, ucb
