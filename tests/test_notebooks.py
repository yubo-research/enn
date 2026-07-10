import importlib.util
import os
from pathlib import Path

import pytest

FAST_TEST_SMOKE = {
    "examples/demo_enn.ipynb": """
import numpy as np
from enn import ENNStatefulFitter, EpistemicNearestNeighbors
from enn.enn.enn_params import PosteriorFlags

rng = np.random.default_rng(0)
train_x = np.linspace(0, 1, 5)[:, None]
train_y = np.sin(train_x)
train_yvar = 0.01 * np.ones_like(train_y)
model = EpistemicNearestNeighbors(train_x, train_y, train_yvar)
fitter = ENNStatefulFitter(k=3, rng=rng)
fitter.tell(train_x, train_y, train_yvar)
params = fitter.ask(model, num_fit_candidates=10, num_fit_samples=2)
posterior = model.posterior(np.array([[0.5]]), params=params, flags=PosteriorFlags())
assert posterior.mu.shape == (1, 1)
""",
    "examples/demo_turbo_enn.ipynb": """
import numpy as np
import torch
from enn import create_optimizer
from enn.benchmarks import Ackley
from enn.turbo.optimizer_config import CandidateRV, turbo_zero_config

rng = np.random.default_rng(18)
torch.manual_seed(17)
objective = Ackley(noise=0.1, rng=rng)
bounds = np.array([objective.bounds] * 2, dtype=float)
optimizer = create_optimizer(
    bounds=bounds,
    config=turbo_zero_config(candidate_rv=CandidateRV.UNIFORM),
    rng=rng,
)
x_arms = optimizer.ask(num_arms=1)
optimizer.tell(x_arms, objective(x_arms))
""",
    "examples/demo_morbo_enn.ipynb": """
import numpy as np
import torch
from enn import create_optimizer
from enn.benchmarks import DoubleAckley
from enn.turbo.config import MorboTRConfig, MultiObjectiveConfig, turbo_zero_config

rng = np.random.default_rng(18)
torch.manual_seed(17)
objective = DoubleAckley(noise=0.1, rng=rng)
bounds = np.array([objective.bounds] * 6, dtype=float)
config = turbo_zero_config(
    trust_region=MorboTRConfig(
        multi_objective=MultiObjectiveConfig(num_metrics=2)
    )
)
optimizer = create_optimizer(bounds=bounds, config=config, rng=rng)
x_arms = optimizer.ask(num_arms=1)
optimizer.tell(x_arms, objective(x_arms))
""",
}


def _run_fast_test_smoke(notebook_path: str) -> None:
    smoke = FAST_TEST_SMOKE[notebook_path]
    repo_root = Path(__file__).resolve().parent.parent
    shim_dir = Path(__file__).resolve().parent / "_nbmake_sitecustomize"
    src_dir = repo_root / "src"
    pythonpath_parts = [str(shim_dir), str(src_dir)]
    existing_pythonpath = os.environ.get("PYTHONPATH", "")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
    exec(compile(smoke.strip(), notebook_path, "exec"), {"__name__": "__main__"})


def _execute_with_kernel(nb, repo_root: Path, kernel_manager) -> None:
    from nbclient import NotebookClient

    client = NotebookClient(
        nb,
        timeout=600,
        kernel_manager=kernel_manager,
        resources={"metadata": {"path": str(repo_root)}},
    )
    client.allow_errors = False
    client.execute()


def run_notebook(notebook_path: str) -> None:
    if os.environ.get("FAST_TEST", "0") == "1" and notebook_path in FAST_TEST_SMOKE:
        _run_fast_test_smoke(notebook_path)
        return

    if importlib.util.find_spec("nbclient") is None:
        pytest.skip("nbclient is not installed")

    import nbformat
    from jupyter_client import KernelManager

    repo_root = Path(__file__).resolve().parent.parent
    nb_path = repo_root / notebook_path
    if not nb_path.exists():
        raise FileNotFoundError(nb_path)
    with nb_path.open(encoding="utf-8") as handle:
        nb = nbformat.read(handle, as_version=4)

    km = KernelManager(kernel_name="python3")
    km.start_kernel(env=os.environ.copy())
    try:
        _execute_with_kernel(nb, repo_root, km)
    finally:
        km.shutdown_kernel(now=True)


@pytest.fixture(autouse=True)
def set_fast_test():
    os.environ["FAST_TEST"] = "1"
    yield
    os.environ.pop("FAST_TEST", None)


def test_demo_enn_notebook():
    run_notebook("examples/demo_enn.ipynb")


def test_demo_turbo_enn_notebook():
    run_notebook("examples/demo_turbo_enn.ipynb")


def test_demo_morbo_enn_notebook():
    run_notebook("examples/demo_morbo_enn.ipynb")
