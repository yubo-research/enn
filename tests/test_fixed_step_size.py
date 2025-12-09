from uhd.fixed_step_size import FixedStepSize


def test_fixed_step_size_ask_returns_step_size():
    adapter = FixedStepSize(0.1)
    assert adapter.ask() == 0.1


def test_fixed_step_size_tell_does_not_change_step_size():
    adapter = FixedStepSize(0.1)

    adapter.tell(success=True)
    assert adapter.ask() == 0.1

    adapter.tell(success=False)
    assert adapter.ask() == 0.1


def test_fixed_step_size_property():
    adapter = FixedStepSize(0.5)
    assert adapter.step_size == 0.5
