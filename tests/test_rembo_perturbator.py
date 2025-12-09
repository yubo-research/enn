import torch
from torch import nn

from uhd.rembo import rembo_perturb, rembo_unperturb
from uhd.rembo_perturbator import REMBOPerturbator


class SimpleModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear1 = nn.Linear(10, 5)
        self.linear2 = nn.Linear(5, 2)


def test_rembo_perturb_changes_parameters():
    module = SimpleModule()
    original_params = [p.clone() for p in module.parameters()]

    z = torch.randn(8)
    rembo_perturb(module, seed=42, z=z)

    for orig, current in zip(original_params, module.parameters()):
        assert not torch.allclose(orig, current.data)


def test_rembo_unperturb_reverses():
    module = SimpleModule()
    original_params = [p.clone() for p in module.parameters()]

    z = torch.randn(8)
    rembo_perturb(module, seed=42, z=z)
    rembo_unperturb(module, seed=42, z=z)

    for orig, current in zip(original_params, module.parameters()):
        assert torch.allclose(orig, current.data, atol=1e-6)


def test_rembo_perturb_deterministic():
    module1 = SimpleModule()
    module2 = SimpleModule()
    for p1, p2 in zip(module1.parameters(), module2.parameters()):
        p2.data.copy_(p1.data)

    z = torch.randn(8)
    rembo_perturb(module1, seed=42, z=z)
    rembo_perturb(module2, seed=42, z=z)

    for p1, p2 in zip(module1.parameters(), module2.parameters()):
        assert torch.allclose(p1.data, p2.data)


def test_rembo_perturb_different_seeds_differ():
    module1 = SimpleModule()
    module2 = SimpleModule()
    for p1, p2 in zip(module1.parameters(), module2.parameters()):
        p2.data.copy_(p1.data)

    z = torch.randn(8)
    rembo_perturb(module1, seed=42, z=z)
    rembo_perturb(module2, seed=43, z=z)

    for p1, p2 in zip(module1.parameters(), module2.parameters()):
        assert not torch.allclose(p1.data, p2.data)


def test_rembo_perturb_different_z_differ():
    module1 = SimpleModule()
    module2 = SimpleModule()
    for p1, p2 in zip(module1.parameters(), module2.parameters()):
        p2.data.copy_(p1.data)

    z1 = torch.randn(8)
    z2 = torch.randn(8)
    rembo_perturb(module1, seed=42, z=z1)
    rembo_perturb(module2, seed=42, z=z2)

    for p1, p2 in zip(module1.parameters(), module2.parameters()):
        assert not torch.allclose(p1.data, p2.data)


def test_rembo_perturb_zero_z_no_change():
    module = SimpleModule()
    original_params = [p.clone() for p in module.parameters()]

    z = torch.zeros(8)
    rembo_perturb(module, seed=42, z=z)

    for orig, current in zip(original_params, module.parameters()):
        assert torch.allclose(orig, current.data)


def test_rembo_perturb_scales_with_z_magnitude():
    module1 = SimpleModule()
    module2 = SimpleModule()
    for p1, p2 in zip(module1.parameters(), module2.parameters()):
        p2.data.copy_(p1.data)

    original_params = [p.clone() for p in module1.parameters()]

    z = torch.randn(8)
    rembo_perturb(module1, seed=42, z=z)
    rembo_perturb(module2, seed=42, z=2 * z)

    for orig, p1, p2 in zip(
        original_params, module1.parameters(), module2.parameters()
    ):
        delta1 = p1.data - orig
        delta2 = p2.data - orig
        assert torch.allclose(delta2, 2 * delta1, atol=1e-6)


def test_rembo_perturb_preserves_l2_norm():
    module = SimpleModule()
    original_params = [p.clone() for p in module.parameters()]

    z = torch.randn(100)
    rembo_perturb(module, seed=42, z=z)

    delta_x = torch.cat(
        [
            (p.data - orig).flatten()
            for orig, p in zip(original_params, module.parameters())
        ]
    )

    z_norm = z.norm().item()
    delta_x_norm = delta_x.norm().item()
    assert abs(delta_x_norm - z_norm) / z_norm < 0.1


def test_rembo_perturbator_ask_perturbs_module():
    module = SimpleModule()
    original_params = [p.clone() for p in module.parameters()]

    perturbator = REMBOPerturbator(num_dim_z=10, seed_A=42, seed=17)
    seed = perturbator.ask(module, step_size=0.1)

    assert isinstance(seed, int)
    for orig, current in zip(original_params, module.parameters()):
        assert not torch.allclose(orig, current.data)


def test_rembo_perturbator_tell_accepts_improvement():
    module = SimpleModule()

    perturbator = REMBOPerturbator(num_dim_z=10, seed_A=42, seed=17)
    seed = perturbator.ask(module, step_size=0.1)
    params_after_ask = [p.clone() for p in module.parameters()]

    perturbator.tell(module, seed, y=1.0, y_var=0.1)

    for after_ask, current in zip(params_after_ask, module.parameters()):
        assert torch.allclose(after_ask, current.data)


def test_rembo_perturbator_tell_rejects_and_unperturbs():
    module = SimpleModule()

    perturbator = REMBOPerturbator(num_dim_z=10, seed_A=42, seed=17)
    seed1 = perturbator.ask(module, step_size=0.1)
    perturbator.tell(module, seed1, y=1.0, y_var=0.1)
    params_after_first = [p.clone() for p in module.parameters()]

    seed2 = perturbator.ask(module, step_size=0.1)
    perturbator.tell(module, seed2, y=0.5, y_var=0.1)

    for after_first, current in zip(params_after_first, module.parameters()):
        assert torch.allclose(after_first, current.data, atol=1e-6)


def test_rembo_perturbator_deterministic_with_seed():
    module1 = SimpleModule()
    module2 = SimpleModule()
    for p1, p2 in zip(module1.parameters(), module2.parameters()):
        p2.data.copy_(p1.data)

    perturbator1 = REMBOPerturbator(num_dim_z=10, seed_A=42, seed=17)
    perturbator2 = REMBOPerturbator(num_dim_z=10, seed_A=42, seed=17)

    perturbator1.ask(module1, step_size=0.1)
    perturbator2.ask(module2, step_size=0.1)

    for p1, p2 in zip(module1.parameters(), module2.parameters()):
        assert torch.allclose(p1.data, p2.data)


def test_rembo_perturbator_counter_increments():
    module = SimpleModule()

    perturbator = REMBOPerturbator(num_dim_z=10, seed_A=42, seed=17)

    seed1 = perturbator.ask(module, step_size=0.1)
    perturbator.tell(module, seed1, y=1.0, y_var=0.1)

    seed2 = perturbator.ask(module, step_size=0.1)

    assert seed2 == seed1 + 1
