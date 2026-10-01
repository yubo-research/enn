from __future__ import annotations

from typing import Any

import numpy as np


class ENNAddToken:
    """One-shot proof that these rows were just appended to one model.

    ``model.add`` is the only constructor that ``enn_fit`` will accept.
    A second ``add``, a different model, or a second ``enn_fit`` invalidates it.
    """

    def __init__(
        self,
        model: Any,
        x: np.ndarray,
        y: np.ndarray,
        yvar: np.ndarray | None,
    ) -> None:
        self._model = model
        self._x = x
        self._y = y
        self._yvar = yvar
        self._pending = True

    def take(self, model: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        if self._model is not model:
            raise ValueError("fit token belongs to a different model")
        if not self._pending or getattr(model, "_pending_fit_token", None) is not self:
            raise ValueError("fit token is stale or already used")
        self._pending = False
        model._pending_fit_token = None
        return self._x, self._y, self._yvar
