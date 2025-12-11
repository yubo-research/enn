import numpy as np
import torch
from torch import nn

from uhd.enn_perturbator import ENNPerturbator, embed_perturbation_streaming


def _make_simple_model() -> nn.Module:
    return nn.Linear(10, 5)


def _get_params_flat(module: nn.Module) -> torch.Tensor:
    return torch.cat([p.data.flatten() for p in module.parameters()])


def test_embed_perturbation_streaming_deterministic():
    model = _make_simple_model()

    emb1 = embed_perturbation_streaming(
        model, seed=42, step_size=0.1, num_dim_embed=20, embed_seed=123, s=4
    )
    emb2 = embed_perturbation_streaming(
        model, seed=42, step_size=0.1, num_dim_embed=20, embed_seed=123, s=4
    )

    assert np.allclose(emb1, emb2)


def test_embed_perturbation_streaming_different_seeds_differ():
    model = _make_simple_model()

    emb1 = embed_perturbation_streaming(
        model, seed=42, step_size=0.1, num_dim_embed=20, embed_seed=123, s=4
    )
    emb2 = embed_perturbation_streaming(
        model, seed=43, step_size=0.1, num_dim_embed=20, embed_seed=123, s=4
    )

    assert not np.allclose(emb1, emb2)


def test_embed_perturbation_streaming_scales_with_step_size():
    model = _make_simple_model()

    emb1 = embed_perturbation_streaming(
        model, seed=42, step_size=0.1, num_dim_embed=20, embed_seed=123, s=4
    )
    emb2 = embed_perturbation_streaming(
        model, seed=42, step_size=0.2, num_dim_embed=20, embed_seed=123, s=4
    )

    assert np.allclose(emb2, 2 * emb1)


def test_enn_perturbator_ask_perturbs_module():
    model = _make_simple_model()
    params_before = _get_params_flat(model).clone()

    perturbator = ENNPerturbator(num_dim_embed=20, num_candidates=5, seed=42)
    seed = perturbator.ask(model, step_size=0.1)

    assert isinstance(seed, int)
    params_after = _get_params_flat(model)
    assert not torch.allclose(params_before, params_after)


def test_enn_perturbator_tell_stores_observation():
    model = _make_simple_model()

    perturbator = ENNPerturbator(num_dim_embed=20, num_candidates=5, seed=42)
    seed = perturbator.ask(model, step_size=0.1)
    perturbator.tell(model, seed, y=1.0, y_var=0.1)

    assert len(perturbator._observations) == 1
    assert perturbator._observations[0][0] == seed
    assert perturbator._observations[0][2] == 1.0


def test_enn_perturbator_tell_accepts_improvement():
    model = _make_simple_model()

    perturbator = ENNPerturbator(num_dim_embed=20, num_candidates=5, seed=42)
    seed = perturbator.ask(model, step_size=0.1)
    params_after_ask = _get_params_flat(model).clone()

    perturbator.tell(model, seed, y=1.0, y_var=0.1)

    params_after_tell = _get_params_flat(model)
    assert torch.allclose(params_after_ask, params_after_tell)


def test_enn_perturbator_tell_rejects_and_unperturbs():
    model = _make_simple_model()

    perturbator = ENNPerturbator(num_dim_embed=20, num_candidates=5, seed=42)

    seed1 = perturbator.ask(model, step_size=0.1)
    perturbator.tell(model, seed1, y=1.0, y_var=0.1)
    params_after_first = _get_params_flat(model).clone()

    seed2 = perturbator.ask(model, step_size=0.1)
    perturbator.tell(model, seed2, y=0.5, y_var=0.1)

    params_after_reject = _get_params_flat(model)
    assert torch.allclose(params_after_first, params_after_reject)


def test_enn_perturbator_uses_enn_for_selection():
    model = _make_simple_model()

    perturbator = ENNPerturbator(num_dim_embed=20, num_candidates=5, seed=42, k=3)

    for i in range(5):
        seed = perturbator.ask(model, step_size=0.1)
        perturbator.tell(model, seed, y=float(i), y_var=0.1)

    assert len(perturbator._observations) == 5

    seed = perturbator.ask(model, step_size=0.1)
    assert isinstance(seed, int)
