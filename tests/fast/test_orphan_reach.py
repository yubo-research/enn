from __future__ import annotations

import numpy as np
import pytest

from enn.benchmarks import Ackley, DoubleAckley
from enn.enn.cat_ties import _EncodedRows, _groups_if_same_columns
from enn.enn.enn_class import EpistemicNearestNeighbors
from enn.enn.enn_params import ENNParams
from enn.enn.mbpann import MBPANNMetric
from enn.enn.metric_stream import Reservoir
from enn.turbo.config.candidate_gen_config import CandidateGenConfig
from enn.turbo.config.enn_fit_config import ENNFitConfig
from enn.turbo.config.enn_surrogate_config import ENNSurrogateConfig
from enn.turbo.config.init_config import InitConfig
from enn.turbo.config.multi_objective_config import MultiObjectiveConfig
from enn.turbo.config.optimizer_config import OptimizerConfig
from enn.turbo.config.tr_length_config import TRLengthConfig
from evals.harness.work_dirs import fresh_work_dir


def _model() -> EpistemicNearestNeighbors:
    rng = np.random.default_rng(0)
    return EpistemicNearestNeighbors(
        rng.random((6, 3)), rng.random((6, 1)), work_dir=fresh_work_dir()
    )


@pytest.mark.parametrize("cls", [Ackley, DoubleAckley])
def test_benchmark_call_matches_evaluate(cls):
    f = cls(noise=0.0, seed=1)
    x = np.zeros((2, 2))
    assert f.__call__(x).shape[0] == 2
    assert f.__call__(x[0]).shape[0] == 1


def test_encoded_rows_finalize_keeps_tied_dims_for_same_width_views():
    rows = _EncodedRows(np.zeros((3, 2)), ((0, 1),))
    assert rows[:2].tied_dims == ((0, 1),)
    assert _groups_if_same_columns(rows[:2], rows) == ((0, 1),)
    assert _groups_if_same_columns(rows[:, :1], rows) == ()
    assert _groups_if_same_columns(rows, np.zeros((3, 2))) == ()
    rows.__array_finalize__(rows)
    assert rows.tied_dims == ((0, 1),)


def test_model_private_shape_properties_and_len():
    model = _model()
    assert model._num_dim == 3
    assert model._num_metrics == 1
    assert model.__len__() == 6


def test_reservoir_len_is_the_number_seen_up_to_capacity():
    assert Reservoir(4, 2, 0).__len__() == 0


def test_metric_view_forwards_unknown_attributes_to_the_snapshot():
    rng = np.random.default_rng(0)
    model = EpistemicNearestNeighbors(
        rng.random((30, 3)),
        rng.random((30, 1)),
        weighting=True,
        work_dir=fresh_work_dir(),
    )
    metric = MBPANNMetric(model)
    assert np.asarray(metric.__getattr__("weights")).shape == (3,)
    with pytest.raises(AttributeError, match="no attribute"):
        metric.__getattr__("no_such_field")


def test_params_post_init_revalidates():
    params = ENNParams(
        k_num_neighbors=2, epistemic_variance_scale=1.0, aleatoric_variance_scale=1.0
    )
    params.__post_init__()


def test_config_post_init_methods_revalidate():
    for config in (
        CandidateGenConfig(),
        ENNFitConfig(),
        ENNSurrogateConfig(tied_dims=((0, 1),)),
        InitConfig(),
        MultiObjectiveConfig(num_metrics=2),
        OptimizerConfig(),
        TRLengthConfig(),
    ):
        config.__post_init__()


def test_config_helpers():
    assert ENNFitConfig().infers_aleatoric_variance is True
    assert (
        ENNFitConfig(
            num_fit_samples=2, infer_aleatoric_variance_scale=False
        ).infers_aleatoric_variance
        is False
    )
    ENNSurrogateConfig(tied_dims=((0, 1),))._check_tied_dims()
    assert InitConfig().get_init_strategy() is not None
