"""Frozen-noise TuRBO LunarLander controller.

`tlunar_fn` is yubo's `tlunar` preset with the `:fn` tag: a 12-parameter
heuristic on Gymnasium `LunarLander-v3`, evaluated as a deterministic function
of those parameters.
"""

import numpy as np

_DIM = 12
_X_LOW = -1.0
_X_WIDTH = 2.0
_MAX_STEPS = 500
_ENV_ID = "LunarLander-v3"


def controller_weights(x):
    """Map a search point in [-1, 1]^12 onto controller weights in [0, 2]^12.

    This is the affine map in yubo's `TurboLunarPolicy.set_params`:
    `2 * (x - x_low) / x_width` with `x_low = -1` and `x_width = 2`.
    """
    point = np.asarray(x, dtype=float)
    if point.shape != (_DIM,):
        raise ValueError(f"expected shape ({_DIM},), got {point.shape}")
    if not np.isfinite(point).all() or np.any(np.abs(point) > 1):
        raise ValueError("controller parameters must be finite and in [-1, 1]")
    return 2.0 * (point - _X_LOW) / _X_WIDTH


class TurboLunarPolicy:
    """12-weight heuristic from yubo's `problems/turbo_lunar_policy.py`."""

    def __init__(self):
        self._w_orig = np.zeros(_DIM)
        self._w = np.zeros(_DIM)

    def num_params(self):
        return _DIM

    def set_params(self, x):
        self._w_orig = np.asarray(x, dtype=float).copy()
        self._w = controller_weights(self._w_orig)

    def get_params(self):
        return self._w_orig

    def __call__(self, state):
        assert self._w.min() >= 0, self._w
        assert self._w.max() <= 2, self._w
        angle_targ = state[0] * self._w[0] + state[2] * self._w[1]
        angle_targ = min(angle_targ, self._w[2])
        angle_targ = max(angle_targ, -self._w[2])
        hover_targ = self._w[3] * np.abs(state[0])

        angle_todo = (angle_targ - state[4]) * self._w[4] - (state[5]) * self._w[5]
        hover_todo = (hover_targ - state[1]) * self._w[6] - (state[3]) * self._w[7]

        if state[6] or state[7]:
            angle_todo = self._w[8]
            hover_todo = -(state[3]) * self._w[9]

        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > self._w[10]:
            a = 2
        elif angle_todo < -self._w[11]:
            a = 3
        elif angle_todo > +self._w[11]:
            a = 1
        return a


def _episode_return(policy, seed):
    import gymnasium

    env = gymnasium.make(_ENV_ID, continuous=False)
    try:
        return _rollout(env, policy, seed)
    finally:
        env.close()


def _rollout(env, policy, seed):
    state, _ = env.reset(seed=int(seed))
    total = 0.0
    for _ in range(_MAX_STEPS):
        action = policy(np.asarray(state, dtype=np.float32))
        state, reward, terminated, truncated, _ = env.step(action)
        total += float(reward)
        if terminated or truncated:
            break
    return total


def tlunar_fn(x, *, num_denoise, noise_seed_0=0):
    """Mean return of `num_denoise` episodes with frozen simulator seeds.

    Episode `i` uses `reset(seed=noise_seed_0 + i)`. The same arguments
    always yield the same value.
    """
    if int(num_denoise) < 1:
        raise ValueError(f"num_denoise must be >= 1, got {num_denoise}")
    policy = TurboLunarPolicy()
    policy.set_params(np.asarray(x, dtype=float))
    returns = [
        _episode_return(policy, int(noise_seed_0) + i) for i in range(int(num_denoise))
    ]
    return float(np.mean(returns))


NAMES = (
    "angle_position_gain",
    "angle_velocity_gain",
    "angle_target_limit",
    "hover_position_gain",
    "angle_error_gain",
    "angular_velocity_damping",
    "hover_error_gain",
    "vertical_velocity_damping",
    "contact_angle_command",
    "contact_vertical_damping",
    "main_engine_threshold",
    "side_engine_threshold",
)
DESCRIPTIONS = (
    "Gain from horizontal position to target lander angle for lateral correction",
    "Gain from horizontal velocity to target lander angle for lateral correction",
    "Symmetric clipping limit on target lander angle for lateral correction",
    "Gain from absolute horizontal position to target hover altitude",
    "Gain from target minus actual lander angle to angle correction command",
    "Damping gain from angular velocity to angle correction command",
    "Gain from target minus actual altitude to hover correction command",
    "Damping gain from vertical velocity to hover correction command",
    "Constant angle correction command overriding flight control when either leg contacts ground",
    "Damping gain from vertical velocity to hover command when either leg contacts ground",
    "Positive hover command threshold for firing the main vertical engine",
    "Symmetric angle command threshold for firing left or right lateral engine",
)
