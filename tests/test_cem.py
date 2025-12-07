import numpy as np
import pytest

from turbo.cem import CEMConfig, CEMSampler


def test_cem_sampler_ask_tell_basic():
    rng = np.random.default_rng(42)

    def sphere_score(x: np.ndarray) -> np.ndarray:
        return -np.sum((x - 0.5) ** 2, axis=1)

    config = CEMConfig(num_samples=50, elite_frac=0.2)
    sampler = CEMSampler(
        num_dim=3, x_center=np.array([0.1, 0.1, 0.1]), config=config, rng=rng
    )

    for _ in range(5):
        samples = sampler.ask()
        scores = sphere_score(samples)
        sampler.tell(scores)

    assert sampler.best_x.shape == (3,)
    assert sampler.mu.shape == (3,)
    assert np.all(sampler.best_x >= 0) and np.all(sampler.best_x <= 1)
    assert sampler.iteration == 5


def test_cem_sampler_converges_to_optimum():
    rng = np.random.default_rng(123)

    def sphere_score(x: np.ndarray) -> np.ndarray:
        return -np.sum((x - 0.7) ** 2, axis=1)

    config = CEMConfig(num_samples=200, elite_frac=0.1)
    sampler = CEMSampler(
        num_dim=2, x_center=np.array([0.2, 0.2]), config=config, rng=rng
    )

    for _ in range(20):
        samples = sampler.ask()
        scores = sphere_score(samples)
        sampler.tell(scores)

    np.testing.assert_allclose(sampler.best_x, [0.7, 0.7], atol=0.1)
    np.testing.assert_allclose(sampler.mu, [0.7, 0.7], atol=0.1)


def test_cem_sampler_respects_bounds():
    rng = np.random.default_rng(456)

    def score_fn(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=1)

    lb = np.array([0.2, 0.3])
    ub = np.array([0.6, 0.7])
    config = CEMConfig(num_samples=100, elite_frac=0.1)
    sampler = CEMSampler(num_dim=2, lb=lb, ub=ub, config=config, rng=rng)

    for _ in range(10):
        samples = sampler.ask()
        assert np.all(samples >= lb) and np.all(samples <= ub)
        scores = score_fn(samples)
        sampler.tell(scores)

    assert np.all(sampler.best_x >= lb) and np.all(sampler.best_x <= ub)


def test_cem_sampler_with_custom_bounds():
    rng = np.random.default_rng(789)

    def linear_score(x: np.ndarray) -> np.ndarray:
        return np.sum(x, axis=1)

    lb = np.array([-1.0, -1.0])
    ub = np.array([2.0, 2.0])
    config = CEMConfig(num_samples=100, elite_frac=0.1)
    sampler = CEMSampler(num_dim=2, lb=lb, ub=ub, config=config, rng=rng)

    for _ in range(15):
        samples = sampler.ask()
        scores = linear_score(samples)
        sampler.tell(scores)

    assert np.all(sampler.best_x >= lb) and np.all(sampler.best_x <= ub)
    np.testing.assert_allclose(sampler.best_x, [2.0, 2.0], atol=0.2)


def test_cem_sampler_tell_without_ask_raises():
    rng = np.random.default_rng(101)
    config = CEMConfig(num_samples=10)
    sampler = CEMSampler(num_dim=2, config=config, rng=rng)

    with pytest.raises(RuntimeError, match="Must call ask"):
        sampler.tell(np.zeros(10))


def test_cem_sampler_tell_wrong_shape_raises():
    rng = np.random.default_rng(202)
    config = CEMConfig(num_samples=10)
    sampler = CEMSampler(num_dim=2, config=config, rng=rng)

    sampler.ask()
    with pytest.raises(ValueError, match="scores shape"):
        sampler.tell(np.zeros(5))


def test_cem_sampler_invalid_bounds():
    rng = np.random.default_rng(303)
    lb = np.array([0.8, 0.8])
    ub = np.array([0.2, 0.2])

    with pytest.raises(ValueError, match="Lower bounds must be strictly less"):
        CEMSampler(num_dim=2, lb=lb, ub=ub, rng=rng)


def test_cem_config_frozen():
    config = CEMConfig()
    with pytest.raises(Exception):
        config.num_samples = 200


def test_cem_sampler_with_smoothing():
    rng = np.random.default_rng(404)

    def sphere_score(x: np.ndarray) -> np.ndarray:
        return -np.sum((x - 0.5) ** 2, axis=1)

    config = CEMConfig(num_samples=50, elite_frac=0.2, smoothing=0.5)
    sampler = CEMSampler(
        num_dim=2, x_center=np.array([0.1, 0.1]), config=config, rng=rng
    )

    for _ in range(10):
        samples = sampler.ask()
        scores = sphere_score(samples)
        sampler.tell(scores)

    assert sampler.best_x.shape == (2,)
    assert np.all(sampler.best_x >= 0) and np.all(sampler.best_x <= 1)


def test_cem_sampler_random_initial_center():
    rng = np.random.default_rng(505)
    config = CEMConfig(num_samples=50)
    sampler = CEMSampler(num_dim=3, config=config, rng=rng)

    assert sampler.mu.shape == (3,)
    assert np.all(sampler.mu >= 0) and np.all(sampler.mu <= 1)


def test_cem_sampler_properties():
    rng = np.random.default_rng(606)
    config = CEMConfig(num_samples=20, elite_frac=0.2)
    sampler = CEMSampler(
        num_dim=2, x_center=np.array([0.5, 0.5]), config=config, rng=rng
    )

    assert sampler.iteration == 0
    assert sampler.best_score == float("-inf")

    samples = sampler.ask()
    scores = -np.sum((samples - 0.5) ** 2, axis=1)
    sampler.tell(scores)

    assert sampler.iteration == 1
    assert sampler.best_score > float("-inf")


def test_cem_sampler_fixed_center_keeps_mu_constant():
    rng = np.random.default_rng(707)
    x_center = np.array([0.3, 0.7])

    config = CEMConfig(num_samples=50, elite_frac=0.2, fixed_center=True)
    sampler = CEMSampler(num_dim=2, x_center=x_center, config=config, rng=rng)

    initial_mu = sampler.mu.copy()

    for _ in range(10):
        samples = sampler.ask()
        scores = -np.sum((samples - 0.5) ** 2, axis=1)
        sampler.tell(scores)

    np.testing.assert_array_equal(sampler.mu, initial_mu)
    assert sampler.iteration == 10


def test_cem_sampler_fixed_center_updates_std():
    rng = np.random.default_rng(808)
    x_center = np.array([0.5, 0.5])

    config = CEMConfig(num_samples=100, elite_frac=0.1, fixed_center=True, init_std=0.3)
    sampler = CEMSampler(num_dim=2, x_center=x_center, config=config, rng=rng)

    initial_std = sampler.std.copy()

    for _ in range(5):
        samples = sampler.ask()
        scores = -np.sum((samples - x_center) ** 2, axis=1)
        sampler.tell(scores)

    assert not np.allclose(sampler.std, initial_std)
    assert np.all(sampler.std > 0)


def test_cem_sampler_fixed_center_with_thompson_noise():
    rng = np.random.default_rng(909)
    x_center = np.array([0.5, 0.5])

    config = CEMConfig(num_samples=100, elite_frac=0.1, fixed_center=True)
    sampler = CEMSampler(num_dim=2, x_center=x_center, config=config, rng=rng)

    for _ in range(10):
        samples = sampler.ask()
        mu = -np.sum((samples - x_center) ** 2, axis=1)
        se = np.full(len(samples), 0.1)
        thompson_scores = mu + se * rng.standard_normal(len(samples))
        sampler.tell(thompson_scores)

    np.testing.assert_array_equal(sampler.mu, x_center)
    assert np.all(sampler.std > config.min_std)
