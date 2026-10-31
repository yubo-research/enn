from enn.evals import metric_12d, metric_ranges
from enn.evals.metric_12d import Metric12dConfig

SHORT_CONFIG = Metric12dConfig(
    n_grid=(30, 100, 300),
    num_test=128,
    num_seeds=3,
    num_fit_candidates=16,
    num_fit_samples=32,
)
LONG_CONFIG = Metric12dConfig(n_grid=(100, 1000, 10000, 100000))


def run_prediction(config: Metric12dConfig) -> None:
    for problem, data in (
        ("sparse", metric_12d.make_data),
        ("unequal_ranges", metric_ranges.make_data),
    ):
        metric_12d.run_eval(config, data=data, problem=problem)
