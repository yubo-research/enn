class FixedStepSize:
    def __init__(self, step_size: float) -> None:
        self._step_size = step_size

    def ask(self) -> float:
        return self._step_size

    def tell(self, success: bool) -> None:
        pass

    @property
    def step_size(self) -> float:
        return self._step_size
