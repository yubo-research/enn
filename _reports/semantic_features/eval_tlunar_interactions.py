"""Falsify controller scale symmetries and additive pair effects.

Run: OPENBLAS_NUM_THREADS=1 python -m _reports.semantic_features.eval_tlunar_interactions
Outputs are independent of the historical semantic benchmark caches.
"""

import argparse
import json
from itertools import combinations, product
from pathlib import Path

import numpy as np

from _reports.semantic_features.tlunar import NAMES, TurboLunarPolicy, _rollout

ROOT = Path(__file__).resolve().parent
PAIRS = tuple(combinations(range(12), 2))
CORNERS = tuple(product((-0.5, 0.5), repeat=2))
SCALE_GROUPS = {
    "all_commands_and_thresholds": tuple(range(4, 12)),
    "angle_only": (4, 5, 8, 11),
    "hover_only": (6, 7, 9, 10),
    "targets_only": (0, 1, 2, 3),
    "thresholds_only": (10, 11),
}


def scaled_point(point, indices, factor):
    """Scale physical controller weights, then return search coordinates."""
    weights = np.asarray(point, dtype=float).copy() + 1
    weights[list(indices)] *= factor
    scaled = weights - 1
    if not np.isfinite(scaled).all() or np.any(np.abs(scaled) > 1) or factor <= 0:
        raise ValueError("positive scale must preserve controller bounds")
    return scaled


def mixed_difference(corners):
    """Corner order is low-low, low-high, high-low, high-high."""
    values = np.asarray(corners)
    return values[..., 3] - values[..., 2] - values[..., 1] + values[..., 0]


def state_bank(rng, count=256):
    states = rng.uniform(-1, 1, (count, 8))
    states[:, 1] = rng.uniform(0, 1.5, count)
    states[:, 6:] = rng.integers(0, 2, (count, 2))
    states[: count // 2, 6:] = 0
    return states


def action_vector(point, states):
    policy = TurboLunarPolicy()
    policy.set_params(point)
    return np.array([policy(state) for state in states])


def symmetry_experiment(env, backgrounds, seeds):
    rng = np.random.default_rng(20261028)
    states = state_bank(rng)
    points = rng.uniform(-0.8, 0.2, (backgrounds, 12))
    actions = np.array([action_vector(p, states) for p in points])
    baseline = episode_matrix(env, points, seeds)
    records = []
    for name, indices in SCALE_GROUPS.items():
        for factor in (0.5, 1.5):
            scaled = np.array([scaled_point(p, indices, factor) for p in points])
            changed = np.array([action_vector(p, states) for p in scaled])
            returns = episode_matrix(env, scaled, seeds)
            records.append(
                {
                    "group": name,
                    "factor": factor,
                    "action_mismatches": int(np.count_nonzero(changed != actions)),
                    "actions_tested": int(actions.size),
                    "changed_returns": int(np.count_nonzero(returns != baseline)),
                    "episodes_tested": int(baseline.size),
                    "max_return_difference": float(np.max(np.abs(returns - baseline))),
                }
            )
    return points, states, baseline, records


def episode_matrix(env, points, seeds):
    policy = TurboLunarPolicy()
    result = np.empty((len(points), len(seeds)))
    for i, point in enumerate(points):
        policy.set_params(point)
        for j, seed in enumerate(seeds):
            result[i, j] = _rollout(env, policy, seed)
    return result


def pair_experiment(env, backgrounds, seeds):
    points = np.random.default_rng(20261027).uniform(-1, 1, (backgrounds, 12))
    returns = np.empty((backgrounds, len(PAIRS), len(seeds), 4))
    for b, point in enumerate(points):
        for p, pair in enumerate(PAIRS):
            corners = np.tile(point, (4, 1))
            corners[:, list(pair)] = CORNERS
            returns[b, p] = episode_matrix(env, corners, seeds).T
        print(f"pair background {b + 1}/{backgrounds} complete", flush=True)
    return points, returns


def evaluate(output, backgrounds=32):
    import gymnasium

    if backgrounds < 2:
        raise ValueError("at least two backgrounds are required")
    output.mkdir(parents=True, exist_ok=True)
    seeds = np.arange(3)
    env = gymnasium.make("LunarLander-v3", continuous=False)
    try:
        points, returns = pair_experiment(env, backgrounds, seeds)
        symmetry = symmetry_experiment(env, backgrounds, seeds)
    finally:
        env.close()
    np.savez_compressed(
        output / "interactions.npz",
        points=points,
        returns=returns,
        pairs=PAIRS,
        corners=CORNERS,
        seeds=seeds,
        symmetry_points=symmetry[0],
        symmetry_states=symmetry[1],
        symmetry_baseline=symmetry[2],
    )
    metadata = {
        "names": NAMES,
        "backgrounds": backgrounds,
        "rng_seed": 20261027,
        "environment": "LunarLander-v3",
        "gymnasium_version": gymnasium.__version__,
        "numpy_version": np.__version__,
        "max_steps": 500,
        "scale_tests": symmetry[3],
    }
    (output / "protocol.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "tlunar_interactions")
    parser.add_argument("--backgrounds", type=int, default=32)
    args = parser.parse_args()
    evaluate(args.output, args.backgrounds)


if __name__ == "__main__":
    main()
