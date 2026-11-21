import numpy as np
import pytest

from enn.benchmarks.ackley_class import Ackley
from enn.benchmarks.double_ackley import DoubleAckley
from enn.enn.cat_ties import attach_tied_dims
from enn.enn.enn_params_class import ENNParams
from enn.enn.metric_stream import Reservoir
from enn.turbo.config import (
    CandidateGenConfig,
    ENNFitConfig,
    ENNSurrogateConfig,
    HybridInit,
    InitConfig,
    MultiObjectiveConfig,
    OptimizerConfig,
    TRLengthConfig,
)


@pytest.mark.parametrize("benchmark", [Ackley, DoubleAckley])
def test_callable_promotes_a_single_row(benchmark):
    objective = benchmark(noise=0, seed=1)
    point = np.array([0.25, 0.75])
    np.testing.assert_array_equal(objective.__call__(point), objective(point[None, :]))


@pytest.mark.parametrize(
    "config",
    [
        CandidateGenConfig(),
        ENNFitConfig(),
        ENNSurrogateConfig(tied_dims=((0, 1),)),
        InitConfig(),
        MultiObjectiveConfig(2),
        OptimizerConfig(),
        TRLengthConfig(),
        ENNParams(3, 1, 1),
    ],
)
def test_normalized_configs_pass_revalidation(config):
    config.__post_init__()


def test_fit_and_initialization_defaults():
    assert ENNFitConfig().infers_aleatoric_variance
    assert not ENNFitConfig(
        num_fit_samples=10, infer_aleatoric_variance_scale=False
    ).infers_aleatoric_variance
    assert isinstance(InitConfig().get_init_strategy(), HybridInit)


def test_array_finalize_clears_ties_without_a_source():
    values = attach_tied_dims(np.ones((2, 2)), ((0, 1),))
    view = values[:1]
    assert view.tied_dims == ((0, 1),)
    view.__array_finalize__(None)
    assert view.tied_dims == () and values.tied_dims == ((0, 1),)


def test_reservoir_length_stops_at_capacity():
    reservoir = Reservoir(1, 1, seed=0)
    reservoir.add([[0], [1]], [0, 1])
    assert reservoir.__len__() == 1 and reservoir.num_seen == 2


def test_native_standard_normals_resume_numpy_state():
    from enn.enn_rust import util

    rng = np.random.default_rng(10)
    state = rng.bit_generator.state["state"]
    words = [divmod(state[key], 2**64) for key in ("state", "inc")]
    native = util.NumpyNormal(*words[0], *words[1])
    np.testing.assert_array_equal(native.standard_normals(12), rng.standard_normal(12))
