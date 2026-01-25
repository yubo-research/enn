import numpy as np
from enn.turbo.types import (
    GPDataPrep,
    GPFitResult,
    ObsLists,
    PrepareAskResult,
    TellInputs,
)


def test_prepare_ask_result():
    result = PrepareAskResult(
        model=None, y_mean=1.0, y_std=0.5, lengthscales=np.array([1.0, 2.0])
    )
    assert result.model is None
    assert result.y_mean == 1.0
    assert result.y_std == 0.5
    assert result.lengthscales is not None


def test_tell_inputs():
    x = np.array([[1.0, 2.0]])
    y = np.array([3.0])
    inputs = TellInputs(x=x, y=y, y_var=None, num_metrics=1)
    assert inputs.x.shape == (1, 2)
    assert inputs.y.shape == (1,)
    assert inputs.y_var is None
    assert inputs.num_metrics == 1


def test_obs_lists():
    obs = ObsLists(x_obs=[[1.0, 2.0]], y_obs=[3.0], y_tr=[3.0], yvar_obs=[0.1])
    assert len(obs.x_obs) == 1
    assert len(obs.y_obs) == 1
    assert len(obs.y_tr) == 1
    assert len(obs.yvar_obs) == 1


def test_gp_fit_result():
    result = GPFitResult(model="mock", likelihood="mock_lh", y_mean=0.0, y_std=1.0)
    assert result.model == "mock"
    assert result.likelihood == "mock_lh"
    assert result.y_mean == 0.0
    assert result.y_std == 1.0


def test_gp_data_prep():
    prep = GPDataPrep(
        train_x=np.array([[1.0]]),
        train_y=np.array([2.0]),
        is_multi=False,
        y_mean=0.0,
        y_std=1.0,
        y_raw=np.array([2.0]),
    )
    assert prep.is_multi is False
    assert prep.train_x.shape == (1, 1)
