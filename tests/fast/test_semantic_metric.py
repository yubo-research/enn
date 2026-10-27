"""Semantic-only AUTO keeps controller coordinates through fitting and restart."""

import numpy as np
import pytest

from enn import SemanticInputTransform, create_optimizer
from enn.turbo.config import ENNIndexDriver, ENNSurrogateConfig
from enn.turbo.config.factory import turbo_enn_config


class Encoder:
    def encode(self, texts):
        return np.eye(len(texts))


@pytest.mark.parametrize("only", [False, True])
def test_semantic_auto_optimizer_observations_and_predictions(tmp_path, only):
    transform = SemanticInputTransform(["a", "b", "c"], [[0, 2]] * 3, encoder=Encoder())
    config = turbo_enn_config(
        enn=ENNSurrogateConfig(
            index_driver=ENNIndexDriver.BPANN_DISK,
            work_dir=tmp_path / "index",
            weighting=True,
            input_transform=transform,
            semantic_only=only,
        ),
        num_init=4,
    )
    optimizer = create_optimizer(
        bounds=transform.bounds, config=config, rng=np.random.default_rng(12)
    )
    x = np.random.default_rng(13).uniform(0, 2, (120, 3))
    optimizer.tell(x, np.sin(x[:, 0]))
    query = optimizer.ask(2)
    assert query.shape == (2, 3)
    assert np.all((query >= 0) & (query <= 2))
    np.testing.assert_allclose(optimizer._x_obs, x)
    optimizer.tell(query, np.sin(query[:, 0]))
    assert np.isfinite(optimizer.ask(1)).all()


def test_semantic_only_requires_projection():
    with pytest.raises(ValueError, match="requires input_transform"):
        ENNSurrogateConfig(semantic_only=True)
