from unittest.mock import patch

import numpy as np
import pytest

from enn import (
    CandidateGenConfig,
    ENNFitConfig,
    ENNParams,
    ENNSurrogateConfig,
    MultiObjectiveConfig,
    TRLengthConfig,
)
from enn.enn.cat_ties import attach_tied_dims
from enn.turbo.config.init_config import InitConfig
from enn.turbo.config.init_strategies.hybrid_init import HybridInit
from enn.turbo.config.init_strategies.lhd_only_init import LHDOnlyInit
from enn.turbo.config.optimizer_config import OptimizerConfig


def test_dataclass_constructors_dispatch_validation():
    cases = (
        (CandidateGenConfig, CandidateGenConfig.__post_init__, {}),
        (ENNFitConfig, ENNFitConfig.__post_init__, {}),
        (
            ENNParams,
            ENNParams.__post_init__,
            {
                "k_num_neighbors": 3,
                "epistemic_variance_scale": 1.0,
                "aleatoric_variance_scale": 0.1,
            },
        ),
        (
            ENNSurrogateConfig,
            ENNSurrogateConfig.__post_init__,
            {"tied_dims": ((0, 1),)},
        ),
        (InitConfig, InitConfig.__post_init__, {}),
        (MultiObjectiveConfig, MultiObjectiveConfig.__post_init__, {"num_metrics": 2}),
        (OptimizerConfig, OptimizerConfig.__post_init__, {}),
        (TRLengthConfig, TRLengthConfig.__post_init__, {}),
    )
    for cls, validate, kwargs in cases:
        with patch.object(
            cls, validate.__name__, autospec=True, side_effect=validate
        ) as hook:
            config = cls(**kwargs)
        hook.assert_called_once_with(config)


def test_config_accessors_report_defaults_and_explicit_choices():
    assert isinstance(InitConfig().get_init_strategy(), HybridInit)
    strategy = LHDOnlyInit()
    assert InitConfig(init_strategy=strategy).get_init_strategy() is strategy
    assert ENNFitConfig().infers_aleatoric_variance
    assert not ENNFitConfig(
        num_fit_samples=10, infer_aleatoric_variance_scale=False
    ).infers_aleatoric_variance


def test_array_finalize_preserves_groups_for_row_views():
    groups = ((0, 1),)
    encoded = attach_tied_dims(np.zeros((3, 2)), groups)
    with patch.object(
        type(encoded),
        "__array_finalize__",
        autospec=True,
        side_effect=type(encoded).__array_finalize__,
    ) as finalize:
        row_view = encoded[1:]
        column_view = encoded[:, :1]
    assert finalize.call_count == 2
    assert row_view.tied_dims == groups
    assert column_view.tied_dims == ()


def test_metric_forwards_known_attributes_and_names_missing_ones(tmp_path):
    from enn import EpistemicNearestNeighbors

    rows = np.arange(3.0)[:, None]
    model = EpistemicNearestNeighbors(rows, rows, weighting=True, work_dir=tmp_path)
    metric = model.metric
    assert metric.num_seen == metric.__getattr__("num_seen") == len(rows)
    with pytest.raises(AttributeError, match="missing_metric_attribute"):
        metric.__getattr__("missing_metric_attribute")


def test_native_normal_draws_match_the_numpy_generator():
    from enn._rust import NumpyNormal

    bit_generator = np.random.PCG64(7)
    state = bit_generator.state["state"]
    initial, increment = state["state"], state["inc"]
    mask = (1 << 64) - 1
    native = NumpyNormal(
        initial >> 64, initial & mask, increment >> 64, increment & mask
    )
    numpy = np.random.Generator(bit_generator)
    np.testing.assert_array_equal(
        native.standard_normals(30), numpy.standard_normal(30)
    )
